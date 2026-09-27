# NANO-GPT — GPT с нуля на чистом NumPy

Трансформер 350k параметров, обученный с нуля (без PyTorch/TensorFlow):
ручной forward и backward, градиенты проверены gradcheck.py.

## Файлы
- `nanogpt.py` — модель: эмбеддинги, 3 слоя self-attention + MLP, LayerNorm
- `train.py` — обучение на датасете 722 диалогов, чекпоинты каждые 100 шагов
- `dialogues.py` — датасет (русские диалоги, char-level)
- `weights.npz` — обученные веса (loss 0.19, шаг ~1100)
- `chat.py` — чат в терминале
- `gradcheck.py` — проверка обратного прохода
- `index.html` — веб-демо, вся модель зашита внутрь, считает в браузере
- `engine.js`, `make_web.py` — JS-порт инференса и сборка демо

## Запуск
```bash
python3 chat.py          # чат с моделью
python3 train.py         # дообучение (STEPS=2600)
```
Модель: d_model=96, 3 слоя, 4 головы, ctx=128, char-level (vocab 66).

## Расширенная версия (BIG) — для Colab GPU

- `train_torch.py` — та же архитектура на PyTorch, конфиг d_model=256, 6 слоёв, 8 голов (~4.9M параметров), экспорт весов совместим с JS-демо
- `pairs.jsonl.gz` — отфильтрованный датасет: 131k пар (29 млн символов) из ru_roleplay_conversation + anime-girl-chat-dataset-en
- `build_dataset.py` — пересборка датасета из исходников HuggingFace (нужны rp.jsonl и ag.parquet, в репо не входят — качаются с HF)
- `colab_train_big.ipynb` — ноутбук для обучения в Google Colab на T4

Фильтр датасета: длина пар, язык (ru/en), дедупликация, вырезание *действий* и (ремарок), чёрные списки 18+ и мата, мусорные символы.

Обучение в Colab: Среда выполнения → GPU T4 → ячейки ноутбука по порядку. Чекпоинт каждые 500 шагов (weights_big.npz), резюм через START.
