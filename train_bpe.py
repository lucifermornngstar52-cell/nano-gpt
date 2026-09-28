# -*- coding: utf-8 -*-
# NANO-GPT BPE: обучение 150M на токенах (tokens.bin, uint16) с резюм-чекпоинтами.
# Архитектура та же (NanoT), но вход — BPE-токены: контекст 512 токенов ~= 2500 символов.
# env: DM NL NH CTX BS LR STEPS TOKENS TOK CKPT WARM START
#   WARM=1 + weights_src.npz — тёплый старт: блоки/lnf/wpe из старой char-модели,
#   эмбеддинги токенов инициализируем средним старых (модель не с нуля).
import os, time, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
D  = int(os.environ.get('DM', '1024'))
NL = int(os.environ.get('NL', '12'))
NH = int(os.environ.get('NH', '16'))
T  = int(os.environ.get('CTX', '512'))
BS = int(os.environ.get('BS', '24'))
LR = float(os.environ.get('LR', '6e-4'))
STEPS = int(os.environ.get('STEPS', '40000'))
TOKENS = os.environ.get('TOKENS', 'tokens.bin')
TOKJSON = os.environ.get('TOK', 'bpe_tokenizer.json')
CKPT = os.environ.get('CKPT', '')
WARM = os.environ.get('WARM', '0')
START = int(os.environ.get('START', '0'))

from tokenizers import Tokenizer
tk = Tokenizer.from_file(TOKJSON)
V = tk.get_vocab_size()

tokens = np.memmap(TOKENS, dtype=np.uint16, mode='r')
print('токенов: %d (%.0fM), vocab %d, контекст %d (~%d символов)' % (
    len(tokens), len(tokens)/1e6, V, T, T*5), flush=True)

# автоподхват шага после обрыва
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
    def forward(self, x):
        B, Tc = x.shape
        x = self.we[x] + self.wpe[:Tc]
        for b in self.blocks:
            h1 = b['ln1'](x)
            q = b['q'](h1).view(B, Tc, NH, -1).transpose(1, 2)
            k = b['k'](h1).view(B, Tc, NH, -1).transpose(1, 2)
            v = b['v'](h1).view(B, Tc, NH, -1).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).reshape(B, Tc, D)
            x = x + b['o'](att)
            x = x + b['fc2'](F.gelu(b['fc'](b['ln2'](x))))
        return F.linear(self.lnf(x), self.we)  # tied head

model = NanoT().to(dev)
print('параметров: %d' % sum(p.numel() for p in model.parameters()), flush=True)

if WARM == '1' and os.path.exists('weights_src.npz'):
    z = np.load('weights_src.npz')
    with torch.no_grad():
        model.wpe.copy_(torch.tensor(z['wpe']))
        for l in range(NL):
            b = model.blocks[l]
            for nm in ('q', 'k', 'v', 'o'):
                getattr(b, nm).weight.copy_(torch.tensor(z[f'{l}{nm}']).T)   # char-модель хранит x@W
            b['fc'].weight.copy_(torch.tensor(z[f'{l}fc']).T)
            b['fc2'].weight.copy_(torch.tensor(z[f'{l}fc2']).T)
            b['ln1'].weight.copy_(torch.tensor(z[f'{l}ln1g'])); b['ln1'].bias.copy_(torch.tensor(z[f'{l}ln1b']))
            b['ln2'].weight.copy_(torch.tensor(z[f'{l}ln2g'])); b['ln2'].bias.copy_(torch.tensor(z[f'{l}ln2b']))
        model.lnf.weight.copy_(torch.tensor(z['lnfg'])); model.lnf.bias.copy_(torch.tensor(z['lnfb']))
        mean_w = torch.tensor(z['we']).mean(0)
        model.we.copy_(mean_w.unsqueeze(0).expand(V, D) + torch.randn(V, D) * 0.01)
    print('тёплый старт: трансформер перенесён, эмбеддинги токенов ~ среднее', flush=True)

if START > 0 and os.path.exists('weights_bpe.npz'):
    z = np.load('weights_bpe.npz')
    with torch.no_grad():
        model.we.copy_(torch.tensor(z['we'])); model.wpe.copy_(torch.tensor(z['wpe']))
        for l in range(NL):
            b = model.blocks[l]
            for nm in ('q', 'k', 'v', 'o', 'fc', 'fc2'):
                getattr(b, nm).weight.copy_(torch.tensor(z[f'{l}{nm}']))
            b['ln1'].weight.copy_(torch.tensor(z[f'{l}ln1g'])); b['ln1'].bias.copy_(torch.tensor(z[f'{l}ln1b']))
            b['ln2'].weight.copy_(torch.tensor(z[f'{l}ln2g'])); b['ln2'].bias.copy_(torch.tensor(z[f'{l}ln2b']))
        model.lnf.weight.copy_(torch.tensor(z['lnfg'])); model.lnf.bias.copy_(torch.tensor(z['lnfb']))
    print('резюм с шага %d' % START, flush=True)

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
    np.savez('weights_bpe.npz', **z)
    json.dump({'cfg': dict(d_model=D, n_layer=NL, n_head=NH, ctx=T), 'vocab_size': V},
              open('meta_bpe.json', 'w'))
    open('step.txt', 'w').write(str(step))
    if CKPT:
        os.makedirs(CKPT, exist_ok=True)
        for f in ('weights_bpe.npz', 'meta_bpe.json', 'step.txt', 'bpe_tokenizer.json'):
            if os.path.exists(f):
                os.system('cp -f "%s" "%s/"' % (f, CKPT))
    print('  сохранено (шаг %d, loss %.3f)%s' % (step, lossv, ' + на Drive' if CKPT else ''), flush=True)

@torch.no_grad()
def sample(prompt='В: как дела?\nО:', n=60, temp=0.8, topk=8):
    model.eval()
    ids = tk.encode(prompt).ids
    out = []
    for _ in range(n):
        x = torch.tensor([ids[-T:]], device=dev)
        lg = model(x)[0, -1] / temp
        k = torch.topk(lg, topk)
        ni = k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)].item()
        ids.append(ni); out.append(ni)
    model.train()
    return tk.decode(out)

def main():
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(START + 1, STEPS + 1):
        x, y = get_batch()
        lg = model(x)
        loss = F.cross_entropy(lg.view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g['lr'] = LR * min(1.0, step / 300) * (0.5 + 0.5 * math.cos(math.pi * min(1.0, step / STEPS)))
        opt.step()
        if step % 100 == 0 or step == STEPS:
            print('step %5d | loss %.3f | lr %.5f | %4.1fs' % (
                step, loss.item(), opt.param_groups[0]['lr'], time.time() - t0), flush=True)
            t0 = time.time()
        if step % 500 == 0 or step == STEPS:
            save(step, loss.item())
    save(STEPS, loss.item())
    print(sample('В: привет\nО:', 40), flush=True)

if __name__ == '__main__':
    main()
