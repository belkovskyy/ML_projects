from __future__ import annotations
import os
import pandas as pd
import psycopg
from .config import load_env

load_env()

def get_conn() -> psycopg.Connection:
    return psycopg.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        dbname=os.getenv("PG_DB", "northwind"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", ""),
        connect_timeout=5,
    )

def execute_select(sql: str, max_rows: int = 200) -> pd.DataFrame:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            cols = [d.name for d in cur.description] if cur.description else []
    return pd.DataFrame(rows, columns=cols)
