# -*- coding: utf-8 -*-
# Проверка градиентов аналитического backward против конечных разностей.
import numpy as np
from nanogpt import NanoGPT

np.random.seed(0)
vocab = list('абвгдеж ')
m = NanoGPT(vocab, d_model=8, n_layer=2, n_head=2, ctx=8, seed=7)
B, T = 2, 8
idx = np.random.randint(0, len(vocab), (B, T))
tgt = np.random.randint(0, len(vocab), (B, T))

def f():
    lg, c = m.forward(idx)
    l, pr = m.loss(lg, tgt)
    return l

lg, c = m.forward(idx)
l0, pr = m.loss(lg, tgt)
grads = m.backward(tgt, pr, c)
print('loss:', l0)

eps = 1e-6
worst = 0.0
keys = list(grads.keys())
checked = 0
for k in keys:
    p = m.p[k]
    if p.ndim == 1 or 'ln' in k: continue  # bias/LN проверим выборочно
    flat = p.reshape(-1)
    n_check = min(3, flat.size)
    for i in np.random.choice(flat.size, n_check, replace=False):
        old = flat[i]
        flat[i] = old + eps; lp = f()
        flat[i] = old - eps; lm = f()
        flat[i] = old
        num = (lp-lm)/(2*eps)
        ana = grads[k].reshape(-1)[i]
        rel = abs(ana-num)/max(1e-8, abs(num)+abs(ana))
        worst = max(worst, rel)
        checked += 1
        if rel > 0.05 and abs(num) > 1e-9:
            print('MISMATCH', k, i, 'num', num, 'ana', ana, 'rel', rel)
print('checked', checked, 'worst rel err', worst)
print('OK' if worst < 0.05 else 'FAIL')
