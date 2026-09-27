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
