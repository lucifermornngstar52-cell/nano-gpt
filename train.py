# -*- coding: utf-8 -*-
# Обучение nano-GPT на русских диалогах (char-level, чистый NumPy).
import numpy as np, time, os
from nanogpt import NanoGPT
from dialogues import Q

# ── корпус: диалоги в формате В:/О: ──
lines = []
for q, a in Q:
    lines.append("В: %s\nО: %s\n\n" % (q.strip(), a.strip()))
corpus = ''.join(lines)*3          # 3 прохода по данным внутри эпохи (мини-датасет)
vocab = sorted(set(corpus))
print('корпус: %d символов, %d уникальных' % (len(corpus), len(vocab)))

CFG = dict(d_model=96, n_layer=3, n_head=4, ctx=128)
model = NanoGPT(vocab, seed=1337, **CFG)
START = int(os.environ.get('START', '0'))
if START > 0 and os.path.exists('weights.npz'):
    z = np.load('weights.npz')
    for k in model.p:
        model.p[k] = z[k]
    model.step = START
    print('резюм с шага %d' % START)
print('параметров: %d' % sum(v.size for v in model.p.values()))

ids = np.array(model.encode(corpus), dtype=np.int64)
T = model.T

def batch(bs=32):
    ix = np.random.randint(0, len(ids)-T-1, bs)
    x = np.stack([ids[i:i+T] for i in ix])
    y = np.stack([ids[i+1:i+T+1] for i in ix])
    return x, y

STEPS = int(os.environ.get('STEPS', '2200'))
lr0, warm = 3e-3, 100
t0 = time.time()
for step in range(START+1, STEPS+1):
    x, y = batch()
    lr = lr0*min(1.0, step/warm)*(0.55**(step//700))
    logits, c = model.forward(x)
    loss, pr = model.loss(logits, y)
    grads = model.backward(y, pr, c)
    model.update(grads, lr)
    if step % 50 == 0 or step == 1:
        el = time.time()-t0
        print('step %4d | loss %.3f | lr %.4f | %4.1fs' % (step, loss, lr, el), flush=True)
    if step % 100 == 0 or step == STEPS:
        print('  пример:', model.generate("В: как дела?\nО:", n=40).replace('\n', '⏎'), flush=True)
        np.savez_compressed('weights.npz', **model.p)
        with open('meta.json','w',encoding='utf-8') as f:
            import json; json.dump(dict(vocab=vocab, cfg=CFG), f, ensure_ascii=False)

np.savez_compressed('weights.npz', **model.p)
with open('meta.json', 'w', encoding='utf-8') as f:
    import json
    json.dump(dict(vocab=vocab, cfg=CFG), f, ensure_ascii=False)
print('веса сохранены: weights.npz')
