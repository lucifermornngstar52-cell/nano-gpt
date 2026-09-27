# -*- coding: utf-8 -*-
# Обучение расширенной nano-GPT на PyTorch (для Colab GPU).
# Та же архитектура, что в nanogpt.py, но быстрее и больше.
# Экспорт весов совместим с JS-движком демо (те же имена параметров).
import os, json, gzip, math, time
import numpy as np
import torch
import torch.nn as nn

DM  = int(os.environ.get('DM', '256'))    # d_model
NL  = int(os.environ.get('NL', '6'))      # слоёв
NH  = int(os.environ.get('NH', '8'))      # голов
CTX = int(os.environ.get('CTX', '256'))   # контекст
STEPS = int(os.environ.get('STEPS', '6000'))
BS  = int(os.environ.get('BS', '64'))
LR  = float(os.environ.get('LR', '3e-3'))
START = int(os.environ.get('START', '0')) # резюм

# ── корпус ──
pairs = [json.loads(l) for l in gzip.open('pairs.jsonl.gz', 'rt', encoding='utf-8')]
corpus = ''.join("В: %s\nО: %s\n\n" % (p['q'], p['a']) for p in pairs)
vocab = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(vocab)}
print('корпус: %d символов, vocab %d, пар %d' % (len(corpus), len(vocab), len(pairs)))

ids_cpu = torch.tensor([stoi[c] for c in corpus], dtype=torch.long)
V, T, hs = len(vocab), CTX, DM // NH
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev)

torch.manual_seed(1337)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        s = math.sqrt(2.0 / DM)
        for nm in ('q', 'k', 'v', 'o'):
            setattr(self, nm, nn.Parameter(torch.empty(DM, DM)))
            nn.init.uniform_(getattr(self, nm), -s, s)
        s2 = math.sqrt(2.0 / (4 * DM))
        self.fc = nn.Parameter(torch.empty(DM, 4 * DM)); nn.init.uniform_(self.fc, -s2, s2)
        self.fc2 = nn.Parameter(torch.empty(4 * DM, DM)); nn.init.uniform_(self.fc2, -s2, s2)
        self.ln1 = nn.LayerNorm(DM); self.ln2 = nn.LayerNorm(DM)
    def forward(self, x):
        B, Tt, _ = x.shape
        h1 = self.ln1(x)
        q = (h1 @ self.q).view(B, Tt, NH, hs).transpose(1, 2)
        k = (h1 @ self.k).view(B, Tt, NH, hs).transpose(1, 2)
        v = (h1 @ self.v).view(B, Tt, NH, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        mask = torch.triu(torch.full((Tt, Tt), -1e10, device=x.device), 1)
        a = torch.softmax(att + mask, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, Tt, DM) @ self.o
        x = x + o
        m = torch.nn.functional.gelu(self.ln2(x) @ self.fc) @ self.fc2
        return x + m

class NanoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.we = nn.Parameter(torch.empty(V, DM)); nn.init.uniform_(self.we, -3 / math.sqrt(DM), 3 / math.sqrt(DM))
        self.wpe = nn.Parameter(torch.empty(T, DM)); nn.init.uniform_(self.wpe, -0.02, 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.lnf = nn.LayerNorm(DM)
    def forward(self, idx):
        B, Tt = idx.shape
        x = self.we[idx] + self.wpe[:Tt]
        for b in self.blocks: x = b(x)
        return self.lnf(x) @ self.we.T   # tied эмбеддинг

model = NanoT().to(dev)
if START > 0 and os.path.exists('weights_big.npz'):
    z = np.load('weights_big.npz')
    with torch.no_grad():
        model.we.copy_(torch.tensor(z['we']))
        model.wpe.copy_(torch.tensor(z['wpe']))
        for l in range(NL):
            blk = model.blocks[l]
            blk.q.copy_(torch.tensor(z[f'{l}q'])); blk.k.copy_(torch.tensor(z[f'{l}k']))
            blk.v.copy_(torch.tensor(z[f'{l}v'])); blk.o.copy_(torch.tensor(z[f'{l}o']))
            blk.fc.copy_(torch.tensor(z[f'{l}fc'])); blk.fc2.copy_(torch.tensor(z[f'{l}fc2']))
            blk.ln1.weight.copy_(torch.tensor(z[f'{l}ln1g'])); blk.ln1.bias.copy_(torch.tensor(z[f'{l}ln1b']))
            blk.ln2.weight.copy_(torch.tensor(z[f'{l}ln2g'])); blk.ln2.bias.copy_(torch.tensor(z[f'{l}ln2b']))
        model.lnf.weight.copy_(torch.tensor(z['lnfg'])); model.lnf.bias.copy_(torch.tensor(z['lnfb']))
    print('резюм с шага %d' % START)

n_par = sum(p.numel() for p in model.parameters())
print('параметров: %d' % n_par)

opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)

def get_batch():
    ix = torch.randint(0, len(ids_cpu) - T - 1, (BS,))
    x = torch.stack([ids_cpu[i:i + T] for i in ix]).to(dev)
    y = torch.stack([ids_cpu[i + 1:i + T + 1] for i in ix]).to(dev)
    return x, y

@torch.no_grad()
def sample(prompt="В: как дела?\nО:", n=60, temp=0.8, topk=8):
    model.eval()
    idx = torch.tensor([[stoi[c] for c in prompt]], device=dev)
    out = prompt
    for _ in range(n):
        lg = model(idx[:, -T:])[0, -1] / temp
        k = torch.topk(lg, topk)
        probs = torch.softmax(k.values, dim=-1)
        ni = k.indices[torch.multinomial(probs, 1)]
        ch = vocab[ni.item()]
        out += ch
        if out.endswith('\n\n'): break
        idx = torch.cat([idx, ni.view(1, 1)], dim=1)
    model.train()
    return out

def export(step, lossv):
    sd_np = {'we': model.we.detach().cpu().numpy(), 'wpe': model.wpe.detach().cpu().numpy(),
             'lnfg': model.lnf.weight.detach().cpu().numpy(), 'lnfb': model.lnf.bias.detach().cpu().numpy()}
    for l in range(NL):
        b = model.blocks[l]
        sd_np.update({f'{l}q': b.q.detach().cpu().numpy(), f'{l}k': b.k.detach().cpu().numpy(),
                      f'{l}v': b.v.detach().cpu().numpy(), f'{l}o': b.o.detach().cpu().numpy(),
                      f'{l}fc': b.fc.detach().cpu().numpy(), f'{l}fc2': b.fc2.detach().cpu().numpy(),
                      f'{l}ln1g': b.ln1.weight.detach().cpu().numpy(), f'{l}ln1b': b.ln1.bias.detach().cpu().numpy(),
                      f'{l}ln2g': b.ln2.weight.detach().cpu().numpy(), f'{l}ln2b': b.ln2.bias.detach().cpu().numpy()})
    np.savez_compressed('weights_big.npz', **sd_np)
    json.dump(dict(vocab=vocab, cfg=dict(d_model=DM, n_layer=NL, n_head=NH, ctx=CTX)),
              open('meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
    ck = os.environ.get('CKPT', '')
    if ck:
        os.makedirs(ck, exist_ok=True)
        np.savez_compressed(os.path.join(ck, 'weights_big.npz'), **sd_np)
        json.dump(dict(vocab=vocab, cfg=dict(d_model=DM, n_layer=NL, n_head=NH, ctx=CTX)),
                  open(os.path.join(ck, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print('  сохранено (шаг %d, loss %.3f)%s' % (step, lossv, ' + на Drive' if ck else ''), flush=True)

def main():
    t0 = time.time()
    model.train()
    AMP = dev == 'cuda' and os.environ.get('AMP', '1') == '1'
    scaler = torch.amp.GradScaler(dev, enabled=AMP)
    print('amp:', AMP)
    for step in range(START + 1, STEPS + 1):
        lr = LR * min(1.0, step / 200) * (0.5 + 0.5 * math.cos(math.pi * min(1.0, step / STEPS)))
        for g in opt.param_groups: g['lr'] = lr
        x, y = get_batch()
        with torch.autocast(dev, enabled=AMP):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0 or step == 1:
            print('step %4d | loss %.3f | lr %.4f | %4.1fs' % (step, loss.item(), lr, time.time() - t0), flush=True)
        if step % 500 == 0 or step == STEPS:
            print('  пример:', sample().replace('\n', '⏎'), flush=True)
            export(step, loss.item())
    
    export(STEPS, loss.item())
    print('готово')

if __name__ == "__main__":
    main()
