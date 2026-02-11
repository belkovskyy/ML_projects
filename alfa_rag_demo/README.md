# Alfa-Bank RAG Demo (локальный прототип)

Локальный прототип **RAG-сервиса** для справки/FAQ по базе документов:

**retrieval (поиск) → decision layer (ok/need_clarify/no_answer) → (ответ / уточнение)**

Проект сделан как портфолио-демо: всё запускается **локально**, без внешних API.
Генерация текста ответа — **опционально** (через **Ollama**).

---

## Что умеет

- **Dense retrieval** по чанкам: эмбеддинги + **FAISS** → top источники (страницы/доки).
- **Агрегация** результатов на уровень документа (doc-level).
- **Decision layer** возвращает статус:
  - `ok` — можно отвечать по найденным источникам
  - `need_clarify` — не хватает контекста, формируем уточняющий вопрос
  - `no_answer` — в базе нет ответа
- (Опционально) **Rerank** через CrossEncoder (**выключен по умолчанию**).
- (Опционально) генерация **ответа/уточнения** через локальную LLM (**Ollama**) строго по найденному контексту.
- (Опционально) прогон по `gold_labels.csv` и отчёт по метрикам.

---

## Быстрый старт

> Если проект лежит внутри монорепозитория, просто сначала перейдите в папку проекта:
> `cd alfa_rag_demo`

### 1) Установка зависимостей

**Вариант A (рекомендуется): conda**
```bash
conda create -n alfa-rag python=3.11 -y
conda activate alfa-rag

# Бинарные зависимости ставим через conda, чтобы не ловить проблемы с NumPy/FAISS
conda install -y -c conda-forge streamlit pyarrow pandas "numpy<2" sentence-transformers transformers
conda install -y -c pytorch faiss-cpu
```

**Вариант B: pip**
```bash
pip install -r requirements.txt
```

---

### 2) Данные (в репозиторий не коммитятся)

Положите файлы в папку `data/`:

- `data/websites.csv` — исходные документы/страницы (**обязательно**)
- `data/chunks_websites.parquet` — чанки (**обязательно**, можно сгенерировать)
- `data/questions_clean.csv` — опционально (для экспериментов/подсказок)

Если у вас есть только `websites.csv`, сгенерируйте parquet:

```bash
python scripts/make_chunks_parquet.py --inp data/websites.csv --out data/chunks_websites.parquet
```

---

### 3) Запуск демо (CLI)

```bash
python scripts/run_demo.py "Где посмотреть все мои счета?"
```

---

### 4) Streamlit UI

```bash
python -m streamlit run apps/streamlit_app.py
```

В интерфейсе можно:
- выбрать `data_dir` и путь к parquet
- включить/выключить reranker
- включить LLM (Ollama) и указать модель
- указать путь до `gold_labels.csv` (опционально)

---

## (Опционально) LLM через Ollama

Если хотите, чтобы сервис **генерировал текст ответа/уточнения**, установите Ollama и скачайте модель.

Пример модели для старта:
```bash
ollama pull qwen2.5:3b
```

По умолчанию Ollama доступна по адресу: `http://127.0.0.1:11434`

> Без Ollama проект тоже работает: вернёт статус и источники, но без “красивого” текста ответа.

---

## Оценка на gold-разметке (если есть)

Если у вас есть `gold_labels.csv`, можно прогнать оценку:

```bash
python scripts/eval_gold.py
```

Формат разметки:
- `label_status` ∈ `ok / need_clarify / no_answer`
- если `ok`, то `gold_web_ids` может содержать несколько `web_id`

---

## Архитектура пайплайна

1) **Retrieval**: эмбеддинги чанков → поиск top-k через FAISS  
2) **Decision layer**: правила/эвристики → `ok / need_clarify / no_answer`  
3) **Действие**:
   - `ok`: (опц.) LLM формирует ответ по контексту
   - `need_clarify`: формируется уточняющий вопрос
   - `no_answer`: отказ + рекомендация уточнить запрос

---

## Структура проекта

```text
src/alfa_rag/
  config.py        # конфигурация
  retrieval.py     # поиск (FAISS)
  rerank.py        # (опц.) cross-encoder rerank
  decision.py      # decision layer: ok/need_clarify/no_answer
  clarify.py       # логика уточнения
  llm.py           # (опц.) Ollama
  service.py       # сборка пайплайна
  eval.py          # метрики/оценка
  labeling.py      # разметка/утилиты

apps/
  streamlit_app.py # UI

scripts/
  make_chunks_parquet.py  # сборка parquet из websites.csv
  run_demo.py             # CLI-демо
  eval_gold.py            # прогон по gold

notebooks/
  *.ipynb          # эксперименты
```

---

## Конфиг (переменные окружения)

Можно задавать переменные окружения (или через `.env`), например:

- `USE_RERANK=0/1`
- `RERANK_MODEL=cross-encoder/...`
- `OLLAMA_MODEL=qwen2.5:3b`
- `E5_MODEL=intfloat/multilingual-e5-small`

Пример (PowerShell):
```powershell
$env:USE_RERANK="0"
$env:OLLAMA_MODEL="qwen2.5:3b"
python -m streamlit run apps/streamlit_app.py
```

Пример (bash):
```bash
export USE_RERANK=0
export OLLAMA_MODEL=qwen2.5:3b
streamlit run apps/streamlit_app.py
```

---

## Примечания

- Папка `data/` и артефакты индекса **не коммитятся** (чтобы не тащить большие файлы и данные в GitHub).
- Если что-то не запускается, первым делом проверьте версии Python/зависимостей и что данные лежат в `data/`.
