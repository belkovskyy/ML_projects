# Text-to-SQL Agent (Northwind / PostgreSQL)

What it does:
- **RU question** → **SQL (PostgreSQL)** → **execution** → **table result**
- **Guardrails**: blocks DDL/DML, allows only `SELECT/WITH`, enforces `LIMIT`
- **Schema introspection** from Postgres (tables/columns/FKs)
- **Schema search** (subset of schema) to reduce hallucinations
- **DB docs generator**: builds Markdown documentation from schema (tables/columns/FKs)
- **Few-shot prompting** for Northwind patterns (revenue / top customers / top employees)
- **Self-healing**: retries with DB error + deterministic synonym fixes for common column hallucinations

## Quickstart (Windows / conda)

```bat
cd /d D:\DS\text2sql_agent_ready

conda create -n text2sql python=3.11 -y
conda activate text2sql

pip install -r requirements.txt
copy .env.example .env
notepad .env

REM install package so imports work everywhere:
pip install -e .

python scripts\test_db.py
python scripts\show_schema.py
python scripts\smoke_queries.py

python -m streamlit run apps\streamlit_app.py

In Streamlit you have 4 tabs: NL→SQL, SQL runner, schema search, and **DB documentation**.
```

## Ollama models
You can try a larger model (often fewer SQL mistakes):
```bat
ollama pull qwen2.5:7b-instruct-q4_K_S
```
Then set `OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_S` in `.env`.

## FAQ
- **Why NL→SQL sometimes errors?** The LLM can generate invalid SQL. This project reduces it via schema search + few-shot + self-healing retries.
- **What is “📚 Schema (search)”?** It's a keyword search over the schema text to quickly find table/column names and FK relations.


## Troubleshooting Ollama 404 on /api/chat
If you see `404 Not Found` for `http://127.0.0.1:11434/api/chat`, your Ollama build may be older or another service is bound to that port.
This project now **falls back to /api/generate** automatically. Still, you can verify:
- `ollama list`
- `http://127.0.0.1:11434/api/tags`

## .env comments
Use `#` for comments. Do NOT write comments after values with `-` on the same line (it becomes part of the value).
Example:
`TRANSLATE_RU_EN=true # comment`
