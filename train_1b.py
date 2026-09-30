# -*- coding: utf-8 -*-
# NANO-GPT 1B: корпус corpus-700m (книги Либрусека, 12+ млрд BPE-токенов).
# Фичи против train_bpe.py: bf16 autocast, градиентный чекпоинтинг,
# 8-битный AdamW (bitsandbytes) с фолбэком на обычный, gradient accumulation,
# длинный warmup, резюм с весами + состоянием оптимизатора.
# env: DM NL NH CTX BS ACCUM LR STEPS TOKENS TOK CKPT GCK SAVE_EVERY START
import os, time, math, json
from functools import partial
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D  = int(os.environ.get('DM', '2048'))
NL = int(os.environ.get('NL', '20'))
NH = int(os.environ.get('NH', '16'))
T  = int(os.environ.get('CTX', '1024'))
BS = int(os.environ.get('BS', '8'))
ACCUM = int(os.environ.get('ACCUM', '16'))
LR = float(os.environ.get('LR', '2e-4'))
STEPS = int(os.environ.get('STEPS', '95000'))
TOKENS = os.environ.get('TOKENS', 'tokens.bin')
TOKJSON = os.environ.get('TOK', 'bpe_tokenizer.json')
CKPT = os.environ.get('CKPT', '')
GCK = os.environ.get('GCK', '1') == '1'
SAVE_EVERY = int(os.environ.get('SAVE_EVERY', '500'))
WARMUPT = 2000

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tokenizers import Tokenizer
tk = Tokenizer.from_file(TOKJSON)
V = tk.get_vocab_size()

tokens = np.memmap(TOKENS, dtype=np.uint16, mode='r')
print('токенов: %.2f млрд, vocab %d, ctx %d' % (len(tokens)/1e9, V, T), flush=True)

START = int(os.environ.get('START', '0'))
if os.path.exists('step.txt'):
    try:
        START = max(START, int(open('step.txt').read().strip()))
    except ValueError:
        pass

class NanoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.we = nn.Parameter(torch.randn(V, D) * 0.02)
        self.wpe = nn.Parameter(torch.randn(T, D) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(NL):
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(D), ln2=nn.LayerNorm(D),
                q=nn.Linear(D, D, bias=False), k=nn.Linear(D, D, bias=False),
                v=nn.Linear(D, D, bias=False), o=nn.Linear(D, D, bias=False),
                fc=nn.Linear(D, 4 * D, bias=False), fc2=nn.Linear(4 * D, D, bias=False))))
        self.lnf = nn.LayerNorm(D)
    def _block(self, b, x):
        B, Tc = x.shape[0], x.shape[1]
        h1 = b['ln1'](x)
        q = b['q'](h1).view(B, Tc, NH, -1).transpose(1, 2)
        k = b['k'](h1).view(B, Tc, NH, -1).transpose(1, 2)
        v = b['v'](h1).view(B, Tc, NH, -1).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(B, Tc, D)
        x = x + b['o'](att)
        return x + b['fc2'](F.gelu(b['fc'](b['ln2'](x))))
    def forward(self, x):
        x = self.we[x] + self.wpe[:x.shape[1]]
        for b in self.blocks:
            if GCK and self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(partial(self._block, b), x, use_reentrant=False)
            else:
                x = self._block(b, x)
        return F.linear(self.lnf(x), self.we)  # tied head

model = NanoT().to(dev)
print('параметров: %.2fM' % (sum(p.numel() for p in model.parameters())/1e6), flush=True)

# резюм: веса + оптимизатор
if START > 0 and os.path.exists('weights_1b.npz'):
    z = np.load('weights_1b.npz')
    with torch.no_grad():
        model.we.copy_(torch.tensor(z['we'])); model.wpe.copy_(torch.tensor(z['wpe']))
        for l in range(NL):
            b = model.blocks[l]
            for nm in ('q', 'k', 'v', 'o', 'fc', 'fc2'):
                getattr(b, nm).weight.copy_(torch.tensor(z[f'{l}{nm}']))
            b['ln1'].weight.copy_(torch.tensor(z[f'{l}ln1g'])); b['ln1'].bias.copy_(torch.tensor(z[f'{l}ln1b']))
            b['ln2'].weight.copy_(torch.tensor(z[f'{l}ln2g'])); b['ln2'].bias.copy_(torch.tensor(z[f'{l}ln2b']))
        model.lnf.weight.copy_(torch.tensor(z['lnfg'])); model.lnf.bias.copy_(torch.tensor(z['lnfb']))
    print('веса загружены, старт с шага %d' % START, flush=True)

try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    OPTK = 'bnb8'
except Exception:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    OPTK = 'fp32'
if START > 0 and os.path.exists('opt_1b.pt'):
    opt.load_state_dict(torch.load('opt_1b.pt', map_location=dev))
    print('состояние оптимизатора (%s) загружено' % OPTK, flush=True)

def get_batch():
    ix = np.random.randint(0, len(tokens) - T - 1, BS)
    x = np.stack([tokens[i: i + T] for i in ix]).astype(np.int64)
    y = np.stack([tokens[i + 1: i + T + 1] for i in ix]).astype(np.int64)
    return torch.tensor(x, device=dev), torch.tensor(y, device=dev)

def save(step, lossv):
    z = {'we': model.we.detach().cpu().numpy().astype(np.float32),
         'wpe': model.wpe.detach().cpu().numpy().astype(np.float32)}
    for l in range(NL):
        b = model.blocks[l]
        for nm in ('q', 'k', 'v', 'o', 'fc', 'fc2'):
            z[f'{l}{nm}'] = getattr(b, nm).weight.detach().cpu().numpy().astype(np.float32)
        z[f'{l}ln1g'] = b['ln1'].weight.detach().cpu().numpy()
        z[f'{l}ln1b'] = b['ln1'].bias.detach().cpu().numpy()
        z[f'{l}ln2g'] = b['ln2'].weight.detach().cpu().numpy()
        z[f'{l}ln2b'] = b['ln2'].bias.detach().cpu().numpy()
    z['lnfg'] = model.lnf.weight.detach().cpu().numpy()
    z['lnfb'] = model.lnf.bias.detach().cpu().numpy()
    np.savez('weights_1b.npz', **z)
    torch.save(opt.state_dict(), 'opt_1b.pt')
    json.dump({'cfg': dict(d_model=D, n_layer=NL, n_head=NH, ctx=T),
              'vocab_size': V, 'opt': OPTK, 'step': step},
             open('meta_1b.json', 'w'))
    open('step.txt', 'w').write(str(step))
    if CKPT:
        os.makedirs(CKPT, exist_ok=True)
        for f in ('weights_1b.npz', 'opt_1b.pt', 'meta_1b.json', 'step.txt', TOKJSON):
            if os.path.exists(f):
                os.system('cp -f "%s" "%s/"' % (f, CKPT))
    print('  чекпоинт (шаг %d, loss %.3f, %s)%s' % (
        step, lossv, OPTK, ' + Drive' if CKPT else ''), flush=True)

@torch.no_grad()
def sample(prompt='В: как дела?\nО:', n=60, temp=0.8, topk=8):
    model.eval()
    ids = tk.encode(prompt).ids
    out = []
    for _ in range(n):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(torch.tensor([ids[-T:]], device=dev))[0, -1].float() / temp
        k = torch.topk(lg, topk)
        ni = k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)].item()
        ids.append(ni); out.append(ni)
    model.train()
    return tk.decode(out)

def main():
    t0 = time.time(); tks = 0
    for step in range(START + 1, STEPS + 1):
        for g in opt.param_groups:
            g['lr'] = LR * min(1.0, step / WARMUPT) * \
                      (0.5 + 0.5 * math.cos(math.pi * min(1.0, step / STEPS)))
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(ACCUM):
            x, y = get_batch()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).view(-1, V), y.view(-1)) / ACCUM
            loss.backward()
            acc += loss.item()
            tks += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == STEPS:
            dt = time.time() - t0
            print('step %6d | loss %.3f | lr %.2e | %5.0f к ток/с | %4.1fs' % (
                step, acc, opt.param_groups[0]['lr'], tks/dt/1000, dt), flush=True)
            t0 = time.time(); tks = 0
        if step % SAVE_EVERY == 0 or step == STEPS:
            save(step, acc)
    save(STEPS, acc)
    print(sample('В: привет\nО:', 40), flush=True)

if __name__ == "__main__":
    main()
