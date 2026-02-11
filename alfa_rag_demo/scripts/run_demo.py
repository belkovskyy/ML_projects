from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from alfa_rag.config import E5_MODEL, USE_RERANK, RERANK_MODEL
from alfa_rag.retrieval import Retriever
from alfa_rag.rerank import Reranker
from alfa_rag.service import Pipeline
from alfa_rag.clarify import build_clarify_bank


def make_embed_fn(model: SentenceTransformer):
    def embed_fn(texts):
        texts2 = [f"query: {str(t)}" for t in texts]
        vecs = model.encode(texts2, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")
    return embed_fn


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_demo.py "ваш запрос"')
        sys.exit(1)

    query = sys.argv[1]
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    retriever = Retriever.from_data_dir(data_dir, embed_model=E5_MODEL)

    reranker = None
    if USE_RERANK:
        reranker = Reranker(RERANK_MODEL, device="cpu")

    pipeline = Pipeline(retriever=retriever, reranker=reranker)

    # optional clarify bank if you have gold_labels.csv
    gold_path = Path(os.getenv("GOLD_PATH", "gold_labels.csv"))
    clarify_bank = None
    embed_fn = None
    if gold_path.exists():
        e5 = SentenceTransformer(E5_MODEL)
        embed_fn = make_embed_fn(e5)
        clarify_bank = build_clarify_bank(gold_path, embed_fn=embed_fn)

    res = pipeline.ask(
        query,
        clarify_bank=clarify_bank,
        embed_fn=embed_fn,
        use_llm=True,
        llm_for="both",
    )

    print("\n=== QUERY ===")
    print(query)
    print("\n=== STATUS ===")
    print(f"[{res['out']['status']}] {res['final']}")

    print("\n=== TOP DOCS ===")
    df = res["out"]["results"]
    if df is not None and len(df):
        cols = [c for c in ["score","rerank_score","web_id","title","url"] if c in df.columns]
        print(df[cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
