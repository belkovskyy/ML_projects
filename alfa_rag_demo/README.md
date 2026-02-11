# Alfa-Bank RAG (local demo)

Локальный прототип RAG-сервиса для FAQ/справки: **retrieval → decision layer → (clarify / answer)**.

Особенности:
- Dense retrieval по чанкам (FAISS) → агрегация в doc-level результаты.
- Decision layer: **правила** (и опционально ML-gate как эксперимент).
- (Опционально) Rerank через CrossEncoder — **выключен по умолчанию**.
- Need-clarify: "clarify bank" (embeddings + nearest examples) + переписывание уточняющего вопроса через локальную LLM (Ollama).
- OK: ответ через локальную LLM (Ollama) строго по контексту из найденных источников.

> Репозиторий рассчитан на локальный запуск (без внешних API). Для демо достаточно Ollama.

## Быстрый старт

### Вариант A (рекомендуется): conda env

```bash
conda create -n alfa-rag python=3.11 -y
conda activate alfa-rag
conda config --env --set channel_priority strict

# Бинарные зависимости ставим через conda-forge/pytorch, чтобы не ломать NumPy
conda install -y -c conda-forge streamlit pyarrow pandas "numpy<2" sentence-transformers transformers
conda install -y -c pytorch faiss-cpu
```

### Вариант B: pip

1) Установите зависимости:
```bash
pip install -r requirements.txt
```

2) Положите данные хакатона в `data/` (эти файлы **не** коммитятся в git):

- `data/websites.csv` (обязательно)
- `data/chunks_websites.parquet` (обязательно)
- `data/questions_clean.csv` (опционально)

Если у вас есть только `websites.csv`, сгенерируйте parquet:

```bash
python scripts/make_chunks_parquet.py --inp data/websites.csv --out data/chunks_websites.parquet
```

3) Постройте/загрузите индекс и запустите демо:
```bash
python scripts/run_demo.py "Где посмотреть все мои счета?"
```

### Streamlit UI

```bash
python -m streamlit run apps/streamlit_app.py
```

В интерфейсе можно:
- выбрать `data_dir` и parquet
- включить/выключить reranker
- включить LLM (Ollama)
- указать путь до `gold_labels.csv` (опционально) — используется как "clarify bank" для подсказок при `need_clarify`

## Ollama на Windows: запуск, выбор модели, перенос моделей на диск D

### Проверка, что сервер Ollama запущен

Ollama на Windows часто запускается в фоне автоматически. Если команда `ollama serve` ругается:

> `bind: Only one usage of each socket address...`

значит сервер **уже** слушает `127.0.0.1:11434` — это нормально.

Проверить можно так (PowerShell):

```powershell
curl.exe http://127.0.0.1:11434
# или
iwr -UseBasicParsing http://127.0.0.1:11434
```

> В PowerShell `curl` — это алиас `Invoke-WebRequest`, поэтому лучше использовать именно `curl.exe`.

### Какую модель ставить

Для ноутбука с RTX 4050 6GB самый простой и быстрый старт:

```powershell
ollama pull qwen2.5:3b
```

В Streamlit в поле **Ollama model** можно указать `qwen2.5:3b`.

Если захочешь “потяжелее” (качество обычно лучше, но медленнее) — попробуй `qwen2.5:7b-instruct`.
На 6GB VRAM может уйти в частичный offload на CPU — это нормально, но будет заметно медленнее.

### Перенос моделей с диска C на диск D

По умолчанию модели лежат в `%USERPROFILE%\.ollama\models` (обычно `C:\Users\<user>\.ollama\models`).

Чтобы хранить модели на другом диске, Ollama поддерживает переменную окружения `OLLAMA_MODELS`.

1) Закрой Ollama (если есть значок в трее — **Quit**), либо убей процесс `Ollama.exe` в диспетчере задач.

2) Перенеси текущую папку моделей на D (пример):

```powershell
mkdir D:\ollama\models
robocopy "$env:USERPROFILE\.ollama\models" "D:\ollama\models" /E
```

3) Задай переменную окружения (один раз):

```powershell
setx OLLAMA_MODELS "D:\\ollama\\models"
```

4) Перезапусти Ollama (через меню Пуск). После перезапуска:

```powershell
ollama list
```

Модели должны появиться без повторной скачки. Если ты поставил `setx`, но всё равно качается в C — почти всегда это потому, что Ollama была запущена **до** изменения переменной (нужен полный перезапуск трея/процесса).

4) Прогон gold-датасета (если есть `gold_labels.csv`):
```bash
python scripts/eval_gold.py
```

> Формат разметки: `label_status` строго `ok/need_clarify/no_answer`. Если `ok`, то `gold_web_ids` может содержать несколько `web_id`.

## Что считается "продом" здесь

Под "продом" в контексте хакатона/собеса — **стабильный локальный сервис**, который:
- возвращает статус (`ok` / `need_clarify` / `no_answer`)
- отдаёт top-docs (источники)
- формирует либо ответ, либо уточняющий вопрос

Для боевого продакшена добавляются: мониторинг, логирование, rate-limit, бэкенд-векторного хранилища, безопасность, тесты нагрузочные и т.д.

## Конфиг

Через переменные окружения:
- `USE_RERANK=0/1`
- `RERANK_MODEL=cross-encoder/...`
- `OLLAMA_MODEL=qwen2.5:3b`
- `E5_MODEL=intfloat/multilingual-e5-small`

## Структура

```
src/alfa_rag/
  config.py
  retrieval.py
  rerank.py
  decision.py
  clarify.py
  llm.py
  service.py
  eval.py
  labeling.py
scripts/
  run_demo.py
  eval_gold.py
notebooks/
  Untitled8 (10).ipynb   # исходный ноутбук
```

