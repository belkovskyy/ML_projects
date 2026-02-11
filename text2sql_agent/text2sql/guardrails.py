from __future__ import annotations
from dataclasses import dataclass
import re
import sqlparse

BAD = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|call|copy)\b", re.I)

@dataclass
class GuardrailResult:
    ok: bool
    reason: str | None = None
    sanitized_sql: str | None = None

def _ensure_limit(sql: str, max_rows: int) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, re.I):
        return sql.rstrip().rstrip(";")
    return sql.rstrip().rstrip(";") + f"\nLIMIT {int(max_rows)}"

def validate_and_sanitize_select(sql: str, max_rows: int = 200) -> GuardrailResult:
    if not sql or not sql.strip():
        return GuardrailResult(False, "empty_sql")

    parsed = sqlparse.parse(sql)
    if not parsed:
        return GuardrailResult(False, "parse_error")

    norm = sql.strip()
    if BAD.search(norm):
        return GuardrailResult(False, "ddl_dml_blocked")

    if not re.match(r"\s*(with|select)\b", norm, re.I):
        return GuardrailResult(False, "only_select_with_allowed")

    sanitized = _ensure_limit(norm, max_rows=max_rows)
    return GuardrailResult(True, None, sanitized)
