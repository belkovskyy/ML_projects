from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Simple character-based chunker.

    - chunk_size: max chars per chunk
    - overlap: chars to overlap between consecutive chunks

    Works well enough for a demo RAG index.
    """
    s = str(text or "").replace("\r", "")
    s = "\n".join([ln.strip() for ln in s.split("\n") if ln.strip()])
    if not s:
        return []

    step = max(1, chunk_size - overlap)
    out: list[str] = []
    for start in range(0, len(s), step):
        piece = s[start : start + chunk_size].strip()
        if piece:
            out.append(piece)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build chunks_websites.parquet from websites.csv")
    ap.add_argument("--inp", default="data/websites.csv", help="Path to websites.csv")
    ap.add_argument("--out", default="data/chunks_websites.parquet", help="Output parquet path")
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=150)

    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # required columns in source file
    for c in ["web_id", "title", "url", "text"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {inp}. Columns: {list(df.columns)}")

    rows = []
    for r in df.itertuples(index=False):
        web_id = getattr(r, "web_id")
        title = getattr(r, "title")
        url = getattr(r, "url")
        text = getattr(r, "text")

        parts = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not parts:
            continue
        for part in parts:
            rows.append({
                "web_id": web_id,
                "title": title,
                "url": url,
                "text": part,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out, index=False)
    print(f"Saved: {out} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
