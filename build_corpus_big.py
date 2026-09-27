# -*- coding: utf-8 -*-
# Собирает большой мультидиалоговый корпус из 4 источников:
#   1. IlyaGusev/ru_turbo_saiga   (37.7k мультидиалогов, gpt-3.5-turbo)
#   2. 0x7o/saiga_scored_ru_chatml (22.5k chatml-диалогов)
#   3. OpenAssistant/oasst1 ru    (727 деревьев, настоящие мультидиалоги людей)
#   4. наш pairs.jsonl.gz          (131k пар, рольплей/аниме)
# Формат: В: ...\nО: ...\n\nВ: ...\nО: ...  (мультидиалоги учат связывать тему)
import json, gzip, re, io, os, sys, collections
import pyarrow.parquet as pq
import pyarrow.compute as pc
import zstandard as zstd

OUT = os.environ.get('OUT', 'corpus_big.txt')
conversations = 0
turns = 0
corpus = []

def clean(s):
    s = s.strip()
    return s if s else None

def add_dialog(pairs, src):
    global conversations, turns
    if not pairs: return
    conv = '\n\n'.join('В: %s\nО: %s' % (u, a) for u, a in pairs)
    corpus.append(conv + '\n\n\n')
    conversations += 1
    turns += len(pairs)

# ── 1. ru_turbo_saiga: мультидиалоги ──
d = zstd.ZstdDecompressor()
with open('data50/ru_turbo.zst','rb') as f:
    txt = d.stream_reader(f).read()
turbo_chats = 0
for line in txt.decode('utf-8').splitlines():
    r = json.loads(line)
    msgs = r['messages']
    pairs, pending = [], None
    for m in msgs:
        if m['role'] == 'user':
            if pending is not None: break
            pending = clean(m['content'])
        elif m['role'] == 'assistant' and pending:
            a = clean(m['content'])
            if a: pairs.append((pending, a))
            pending = None
    add_dialog(pairs, 'turbo')
    turbo_chats += 1
print('ru_turbo: %d диалогов' % turbo_chats)

# ── 2. saiga_scored (chatml в одной колонке text) ──
t = pq.read_table('data50/saiga_scored.parquet')
n_sc = 0
for row in t.column('text').to_pylist():
    # parse chatml
    toks = re.findall(r'<\|im_start\|>(user|assistant)(.*?)<\|im_end\|>', row, re.S)
    done = []
    cur_u = None
    for role, body in toks:
        body = body.strip()
        if role == 'user' and body: cur_u = body
        elif role == 'assistant' and body and cur_u:
            done.append((cur_u, body)); cur_u = None
    add_dialog(done, 'scored')
    n_sc += 1
print('saiga_scored: %d диалогов' % n_sc)

# ── 3. OASST ru: деревья → мультидиалоги по ветке ──
t = pq.read_table('data50/oasst.parquet')
t = t.filter(pc.equal(t['lang'], 'ru'))
msgs = {}
for r in t.to_pylist():
    msgs[r['message_id']] = r
children = collections.defaultdict(list)
for r in msgs.values():
    if r['parent_id']: children[r['parent_id']].append(r)
n_oa = 0
roots = [r for r in msgs.values() if r['parent_id'] is None]
def extract(r):
    pairs, pending = [], None
    stack = [(r, None)]
    # простой обход: очередь (message, pending_user)
    q = collections.deque([(r, None)])
    while q:
        m, pu = q.popleft()
        if m['role'] == 'prompter':
            if pu is None: pu = m['text'].strip()
        elif m['role'] == 'assistant' and pu:
            pairs.append((pu, m['text'].strip()))
            pu = None
        for c in children[m['message_id']]:
            if len(pairs) < 12: q.append((c, pu))
    return pairs
for root in roots:
    add_dialog(extract(root), 'oasst')
    n_oa += 1
print('oasst: %d деревьев' % n_oa)

# ── 4. наш pairs.jsonl ──
n_ours = 0
for line in gzip.open('pairs.jsonl.gz', 'rt', encoding='utf-8'):
    p = json.loads(line)
    u, a = clean(p['q']), clean(p['a'])
    if u and a:
        add_dialog([(u, a)], 'ours')
        n_ours += 1
print('наших пар: %d' % n_ours)

text = ''.join(corpus)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(text)
print('ИТОГО: %d диалогов, %d реплик, %d символов' % (conversations, turns, len(text)))
print('сохранено:', OUT)
