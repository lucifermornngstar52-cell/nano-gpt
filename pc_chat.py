# -*- coding: utf-8 -*-
# NANO-GPT 150M — локальный чат на ПК (без GPU, чистый Python).
# Первый запуск сам скачает веса (~280 МБ fp16) с GitHub-релиза.
# Требования: pip install numpy torch
import os, sys, json, ssl, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

WURL = 'https://github.com/lucifermornngstar52-cell/nano-gpt/releases/download/char-150m/'
WFILE = os.environ.get('WFILE', 'weights_150m_fp16.npz')
MFILE = 'meta_150m.json'

def fetch(fn, must=True):
    if os.path.exists(fn):
        return
    print('качаю %s ...' % fn)
    try:
        req = urllib.request.Request(WURL + fn, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=_CTX, timeout=120) as r, open(fn, 'wb') as out:
            total = int(r.headers.get('Content-Length', 0))
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print('\r  %.0f%%' % (100 * done / total), end='')
        print('\r  ок: %.0f МБ' % (os.path.getsize(fn) / 1e6))
    except Exception as e:
        if must:
            print('не скачалось (%s). Положи %s рядом со скриптом вручную.' % (e, fn))
            sys.exit(1)

fetch(MFILE)
fetch(WFILE)
WFILE = os.environ.get('WFILE', WFILE)
if not os.path.exists(WFILE):
    print('файл весов не найден: %s' % WFILE); sys.exit(1)

meta = json.load(open(MFILE, encoding='utf-8'))
c = meta['cfg']
vocab = meta['vocab']
V, D, NL, NH, T = len(vocab), c['d_model'], c['n_layer'], c['n_head'], c['ctx']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
stoi = {ch: i for i, ch in enumerate(vocab)}

class NanoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.we = nn.Parameter(torch.zeros(V, D))
        self.wpe = nn.Parameter(torch.zeros(T, D))
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
        return F.linear(self.lnf(x), self.we)

model = NanoT().to(dev).eval()
z = np.load(WFILE)
with torch.no_grad():
    model.we.copy_(torch.tensor(z['we']).float())
    model.wpe.copy_(torch.tensor(z['wpe']).float())
    for l in range(NL):
        b = model.blocks[l]
        for nm in ('q', 'k', 'v', 'o'):
            getattr(b, nm).weight.copy_(torch.tensor(z['%d%s' % (l, nm)]).T.float())
        b.fc.weight.copy_(torch.tensor(z['%dfc' % l]).T.float())   # (4096,1024)
        b.fc2.weight.copy_(torch.tensor(z['%dfc2' % l]).T.float()) # (1024,4096)
        b['ln1'].weight.copy_(torch.tensor(z['%dln1g' % l]).float())
        b['ln1'].bias.copy_(torch.tensor(z['%dln1b' % l]).float())
        b['ln2'].weight.copy_(torch.tensor(z['%dln2g' % l]).float())
        b['ln2'].bias.copy_(torch.tensor(z['%dln2b' % l]).float())
    model.lnf.weight.copy_(torch.tensor(z['lnfg']).float())
    model.lnf.bias.copy_(torch.tensor(z['lnfb']).float())
print('NANO-GPT 150M | %s | напиши что-нибудь, /exit чтобы выйти' % dev)

@torch.no_grad()
def reply(prompt, n=80, temp=0.7, topk=6):
    ids = [stoi.get(ch, 0) for ch in prompt]
    out = []
    for _ in range(n):
        x = torch.tensor([ids[-T:]], device=dev)
        lg = model(x)[0, -1] / temp
        k = torch.topk(lg, topk)
        ni = k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)].item()
        ids.append(ni); out.append(vocab[ni])
        if ''.join(out).endswith('\n\n'):
            break
    return ''.join(out).split('\n\n')[0].strip()

hist = ''
while True:
    try:
        q = input('\nТы: ').strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q == '/exit':
        break
    a = reply(hist + 'В: ' + q + '\nО:')
    print('Аика: ' + (a or '...'))
    hist += 'В: ' + q + '\nО: ' + a + '\n\n'
    if len(hist) > 800:
        hist = hist[-600:]
