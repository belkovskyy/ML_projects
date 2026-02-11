 Text-to-SQL Agent (Northwind + PostgreSQL)

Локальный демо-проект: **вопрос на русском → SQL (PostgreSQL) → выполнение → табличный результат**.

Цель — показать, как можно собрать безопасного NL→SQL агента с ограничениями (guardrails),
уменьшением галлюцинаций (schema search) и удобным UI (Streamlit) — без внешних API.

---

## Возможности

- **RU вопрос → SQL → выполнение** в PostgreSQL (Northwind).
- **Guardrails (безопасность)**:
  - разрешены только `SELECT` / `WITH`
  - блокируются DDL/DML (`DROP/DELETE/UPDATE/INSERT/ALTER/...`)
  - принудительный `LIMIT` (чтобы не уронить базу большими выборками)
- **Интроспекция схемы** из Postgres: таблицы / колонки / внешние ключи.
- **Schema search**: отбор релевантного подмножества схемы перед генерацией SQL (меньше ошибок).
- **Генератор документации по БД**: собирает Markdown-описание таблиц/полей/связей.
- **Few-shot prompting** под паттерны Northwind (выручка, топ клиентов/сотрудников и т.п.).
- **Self-healing**: повтор с учётом ошибки БД + детерминированные фиксы типичных “галлюцинаций” колонок.

---

## Быстрый старт (Windows)

### 0) Запуск PostgreSQL + Northwind через Docker

В корне проекта:
```bat
docker compose up -d
```

Проверка, что контейнер жив:
```bat
docker compose ps
```

Если нужно пересоздать базу “с нуля” (удалить данные):
```bat
docker compose down -v
docker compose up -d
```

По умолчанию Postgres доступен на:
- host: `localhost`
- port: `5433`
- db: `northwind`
- user/pass: `postgres` / `postgres`

---

### 1) Установка окружения (conda)

```bat
cd /d D:\DS\text2sql_agent

conda create -n text2sql python=3.11 -y
conda activate text2sql

pip install -r requirements.txt

REM чтобы импорты работали везде:
pip install -e .

copy .env.example .env
notepad .env
```

---

### 2) Проверка подключения и smoke-тесты

```bat
python scripts\test_db.py
python scripts\show_schema.py
python scripts\smoke_queries.py
```

---

### 3) Запуск Streamlit UI

```bat
python -m streamlit run apps\streamlit_app.py
```

В интерфейсе обычно есть вкладки:
- **NL→SQL** (вопрос → SQL → результат)
- **SQL runner** (ручной запуск SQL, тоже с guardrails)
- **Schema search** (поиск по схеме БД)
- **DB documentation** (документация, собранная из схемы)

---

## (Опционально) LLM через Ollama

Проект может использовать локальную модель через Ollama. Пример модели для старта:

```bat
ollama pull qwen2.5:3b
```

Можно попробовать модель больше (часто меньше SQL-ошибок, но медленнее):
```bat
ollama pull qwen2.5:7b-instruct-q4_K_S
```

Дальше укажи модель в `.env`, например:
```env
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_S
```

---

## FAQ

### Почему NL→SQL иногда ошибается?
LLM может сгенерировать невалидный SQL или перепутать имена колонок/таблиц.
В проекте это снижено за счёт:
- schema search (подсовываем модели только релевантные куски схемы)
- few-shot примеров
- self-healing (повтор с ошибкой БД и типовыми исправлениями)

### Что такое “Schema search”?
Это быстрый поиск по текстовому представлению схемы (таблицы, колонки, связи),
чтобы быстро понять правильные названия и связи (FK).

---

## Troubleshooting

### Ollama 404 на `/api/chat`
Если получаешь `404 Not Found` на `http://127.0.0.1:11434/api/chat`, возможны причины:
- устаревшая сборка Ollama
- на порту 11434 запущено что-то другое

Проект умеет **fallback** на `/api/generate`, но полезно проверить:
- `ollama list`
- `http://127.0.0.1:11434/api/tags`

### Комментарии в `.env`
Используй `#` для комментариев.
Не пиши комментарий на той же строке после значения — он может “прилипнуть” к значению.

Плохо:
```env
TRANSLATE_RU_EN=true # comment
```

Хорошо:
```env
# Перевод RU→EN (если включён)
TRANSLATE_RU_EN=true
```

---

## Структура проекта

```text
apps/
  streamlit_app.py        # UI
scripts/
  test_db.py              # проверка подключения
  show_schema.py          # вывод схемы
  smoke_queries.py        # быстрые тест-запросы
src/
  ...                     # логика агента (NL→SQL, guardrails, schema search)
docker/
  init/                   # northwind.sql (инициализация БД)
docker-compose.yml        # Postgres в контейнере
```
