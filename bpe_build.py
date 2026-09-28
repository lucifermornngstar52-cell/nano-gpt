# -*- coding: utf-8 -*-
# Этап 1 BPE: сборка корпуса (русская Википедия через HF + наши диалоги),
# обучение BPE-токенизатора и упаковка в tokens.bin (uint16).
# env:
#   NBYTES=1200000000  сколько символов вики взять (~1.2 ГБ)
#   WIKI=1             1 = качать вики, 0 = только диалоги (для тестов)
#   DIAL=corpus_big.txt.gz  файл диалогов
#   VOCAB=8192         размер BPE-словаря
#   OUT=corpus_mix.txt.gz   итоговый корпус
import os, gzip, re, struct

NBYTES = int(os.environ.get('NBYTES', '1200000000'))
WIKI   = os.environ.get('WIKI', '1')
DIAL   = os.environ.get('DIAL', 'corpus_big.txt.gz')
VOCAB  = int(os.environ.get('VOCAB', '8192'))
OUT    = os.environ.get('OUT', 'corpus_mix.txt.gz')

# ---------- 1. корпус ----------
if not os.path.exists(OUT):
    total = n = 0
    if WIKI == '1':
        from datasets import load_dataset
        ds = load_dataset('wikimedia/wikipedia', '20231101.ru', split='train', streaming=True)
        out = gzip.open(OUT, 'wt', encoding='utf-8')
        for art in ds:
            t = (art.get('title', '') and art['title'] + '. ') + art['text'].strip()
            if len(t) < 500:
                continue
            t = re.sub(r'\n{3,}', '\n\n', t)
            t = re.sub(r'[ \t]{2,}', ' ', t)
            out.write(t + '\n\n')
            total += len(t); n += 1
            if n % 20000 == 0:
                print('  вики: %d статей, %.0fM символов' % (n, total/1e6), flush=True)
            if total >= NBYTES:
                break
        out.close()
        print('вики готова: %d статей, %.0fM символов' % (n, total/1e6), flush=True)
    else:
        gzip.open(OUT, 'wt', encoding='utf-8').close()
if os.path.exists(DIAL):
    d = gzip.open(DIAL, 'rt', encoding='utf-8').read()
    with gzip.open(OUT, 'at', encoding='utf-8') as f:
        f.write('\n\n' + d + '\n\n' + d)   # x2: чат-навык должен остаться
    print('диалоги добавлены: %.0fM символов x2' % (len(d)/1e6), flush=True)

# ---------- 2. BPE-токенизатор ----------
from tokenizers import Tokenizer, pre_tokenizers, decoders, trainers, models

if not os.path.exists('bpe_tokenizer.json'):
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, show_progress=True, min_frequency=50,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[])
    sample, got = [], 0
    with gzip.open(OUT, 'rt', encoding='utf-8') as f:
        for line in f:
            sample.append(line); got += len(line)
            if got > 200_000_000:   # учим словарь на 200M — этого достаточно
                break
    print('учим BPE на %d строках (%.0fM символов)...' % (len(sample), got/1e6), flush=True)
    tk.train_from_iterator(sample, trainer)
    del sample
    tk.save('bpe_tokenizer.json')
    print('bpe_tokenizer.json готов', flush=True)

# ---------- 3. tokens.bin ----------
if not os.path.exists('tokens.bin'):
    tk = Tokenizer.from_file('bpe_tokenizer.json')
    V = tk.get_vocab_size()
    assert V < 65536, 'vocab не влезает в uint16'

    def ids2bytes(ids):
        return struct.pack('<%dH' % len(ids), *ids)

    out = open('tokens.bin', 'wb')
    toks = blocks = 0
    with gzip.open(OUT, 'rt', encoding='utf-8') as f:
        buf, bufl = [], 0
        for line in f:
            buf.append(line); bufl += len(line)
            if bufl >= 4_000_000:
                enc = tk.encode_batch(buf)
                out.write(b''.join(ids2bytes(e.ids) for e in enc))
                toks += sum(len(e.ids) for e in enc)
                buf, bufl = [], 0
                blocks += 1
                if blocks % 40 == 0:
                    print('  токенизировано %.0fM токенов' % (toks/1e6), flush=True)
        if buf:
            enc = tk.encode_batch(buf)
            out.write(b''.join(ids2bytes(e.ids) for e in enc))
            toks += sum(len(e.ids) for e in enc)
    out.close()
    print('tokens.bin: %d токенов (%.1f МБ)' % (toks, toks*2/1e6), flush=True)
print('готово')
