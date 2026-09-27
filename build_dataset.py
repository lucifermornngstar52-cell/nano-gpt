# -*- coding: utf-8 -*-
# Фильтрует два датасета с HuggingFace и собирает расширенный корпус:
#   1) Arketov/ru_roleplay_conversation  (русские ролевые диалоги)
#   2) coderian/anime-girl-chat-dataset-en (английский чат)
# Выход: pairs.jsonl - пары {"q": ..., "a": ...} в чистом виде В:/О:
import json, re, os

BAD = ['http', 'www.', '://', '```', '{{', '<<', '[', ']', '*', 'Сцена возвращ', 'Ролевой игре']
NSFW = ['секс', 'оргазм', 'стона', 'постель', 'постел', 'раздева', 'трус', 'лифчик',
        'грудь', 'член', 'вагин', 'порно', 'эроти', 'обнажён',
        'нага', 'нагой', 'штаны снял', 'запясть', 'облиз', 'подошв', 'фетиш',
        'душ вместе', 'порк', 'связ', 'хлест', 'наслажд', 'проник', 'влагал',
        'спальн', 'домина', 'подчинени', 'вкус тво', 'пахн', 'запах', 'потрога',
        'раздеть', 'разден', 'ласк', 'тело тво', 'шею', 'прижал к', 'стона']
NSFW2 = ['поцелу', 'целова', 'обним', 'мурл']
PROFA = ['бля', 'хуй', 'пизд', 'еба', 'ёб', 'сука', 'мраз', 'fuck', 'shit']

def clean(s):
    s = re.sub(r'\*[^*]{0,300}\*', ' ', s)
    s = re.sub(r'\([^)]{0,120}\)', ' ', s)
    s = s.replace('\\', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def cyr_ratio(s): return sum('а' <= c.lower() <= 'я' or c == 'ё' for c in s) / max(len(s), 1)
def lat_ratio(s): return sum('a' <= c.lower() <= 'z' for c in s) / max(len(s), 1)

def ok(pair_q, pair_a, lang):
    if not pair_q or not pair_a: return False
    if any(b in pair_q.lower() or b in pair_a.lower() for b in BAD): return False
    if any(b in pair_a.lower() for b in NSFW): return False
    if any(b in pair_a.lower() or b in pair_q.lower() for b in NSFW2): return False
    if any(b in pair_q.lower() or b in pair_a.lower() for b in PROFA): return False
    if not (5 <= len(pair_q) <= 200 and 10 <= len(pair_a) <= 280): return False
    r = cyr_ratio if lang == 'ru' else lat_ratio
    return r(pair_q) >= 0.6 and r(pair_a) >= 0.6

seen, pairs = set(), []

def add(q, a, lang):
    q, a = clean(q), clean(a)
    if not ok(q, a, lang): return
    h = (q[:80].lower(), a[:80].lower())
    if h in seen: return
    seen.add(h)
    pairs.append({'q': q, 'a': a})

n_conv = 0
with open('rp.jsonl', encoding='utf-8') as f:
    for line in f:
        n_conv += 1
        try: d = json.loads(line)
        except Exception: continue
        conv = d.get('conv') or []
        for i in range(len(conv) - 1):
            if conv[i].get('role') == 'user' and conv[i+1].get('role') == 'bot':
                add(conv[i]['content'], conv[i+1]['content'], 'ru')
print(f'ru_roleplay: {n_conv} диалогов просмотрено')

import pyarrow.parquet as pq
t = pq.read_table('ag.parquet')
users, girls = t.column('user').to_pylist(), t.column('anime_girl').to_pylist()
for u, g in zip(users, girls):
    add(str(u), str(g), 'en')
print(f'anime-girl: {len(users)} пар просмотрено')

ru = sum(1 for p in pairs if cyr_ratio(p['q']) >= 0.5)
with open('pairs.jsonl', 'w', encoding='utf-8') as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')
chars = sum(len(p['q']) + len(p['a']) for p in pairs)
print(f'итого пар: {len(pairs)} (русских ~{ru}, английских ~{len(pairs)-ru})')
print(f'символов: {chars:,}')
