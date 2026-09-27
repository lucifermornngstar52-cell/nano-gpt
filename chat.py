# -*- coding: utf-8 -*-
# Чат с обученным nano-GPT: python3 chat.py
import numpy as np, json, sys
from nanogpt import NanoGPT
meta = json.load(open('meta.json', encoding='utf-8'))
model = NanoGPT(meta['vocab'], seed=7, **meta['cfg'])
z = np.load('weights.npz')
for k in model.p: model.p[k] = z[k]
print('нано-гпт готов (loss 0.19). пиши, "выход" чтобы закончить')
hist = ''
while True:
    try: q = input('\nты: ').strip().lower()
    except EOFError: break
    if not q or q in ('выход','exit','quit'): break
    prompt = (hist[-200:] if hist else '') + 'В: %s\nО:' % q
    ans = model.generate(prompt, n=60, temp=0.7, topk=5).strip()
    print('нано: %s' % ans)
    hist += 'В: %s\nО: %s\n\n' % (q, ans)
