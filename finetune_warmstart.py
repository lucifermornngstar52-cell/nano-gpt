# -*- coding: utf-8 -*-
# Тёплый старт: переносим обученные 50M-веса в новый словарь корпуса.
# Старая модель знает 398 символов, новый корпус — 277. Совпадающие символы
# забирают свои выученные эмбеддинги (знание русского переносится),
# редкие новые символы получают среднюю строку + шум.
import json, os, numpy as np

OLD_W = os.environ.get('OLD_W', 'weights_old.npz')   # веса 50M (vocab 398)
OLD_META = os.environ.get('OLD_META', 'meta_old.json')
NEW_W = os.environ.get('NEW_W', 'weights_big.npz')  # куда писать совместимые веса

old_meta = json.load(open(OLD_META, encoding='utf-8'))
old_vocab = old_meta['vocab']

# новый словарь — из корпуса
import gzip
with gzip.open(os.environ.get('CORPUS', 'corpus_big.txt.gz'), 'rt', encoding='utf-8') as f:
    corpus_vocab = sorted(set(f.read()))

zo = np.load(OLD_W)
we_old = zo['we']
rng = np.random.default_rng(0)
mean_row = we_old.mean(axis=0)

we_new = np.empty((len(corpus_vocab), we_old.shape[1]), dtype=np.float32)
kept = 0
for i, ch in enumerate(corpus_vocab):
    if ch in old_meta['stoi'] if 'stoi' in old_meta else ch in old_vocab:
        we_new[i] = we_old[old_vocab.index(ch)]
        kept += 1
    else:
        we_new[i] = mean_row + rng.normal(0, 0.02, we_old.shape[1]).astype(np.float32)
print('перенесено %d/%d символов' % (kept, len(corpus_vocab)))

sd = {'we': we_new}
for k in zo.files:
    if k != 'we':
        sd[k] = zo[k]
np.savez_compressed(NEW_W, **sd)
json.dump(dict(vocab=corpus_vocab, cfg=dict(d_model=old_meta['cfg']['d_model'],
             n_layer=old_meta['cfg']['n_layer'], n_head=old_meta['cfg']['n_head'],
             ctx=old_meta['cfg']['ctx'])),
          open('meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('готово:', NEW_W, '(', len(corpus_vocab), 'символов)')
