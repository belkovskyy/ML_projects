from __future__ import annotations

import os
from pathlib import Path

from alfa_rag.config import E5_MODEL, USE_RERANK, RERANK_MODEL
from alfa_rag.retrieval import Retriever
from alfa_rag.rerank import Reranker
from alfa_rag.service import Pipeline
from alfa_rag.eval import build_gold_runs, eval_runs


def main():
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    gold_path = os.getenv("GOLD_PATH", "gold_labels.csv")

    retriever = Retriever.from_data_dir(data_dir, embed_model=E5_MODEL)

    reranker = None
    if USE_RERANK:
        reranker = Reranker(RERANK_MODEL, device="cpu")

    pipeline = Pipeline(retriever=retriever, reranker=reranker)

    runs = build_gold_runs(pipeline, gold_path=gold_path, k_docs=20, k_chunks=80, save_parquet="gold_runs.parquet")
    stats = eval_runs(runs, out_errors_csv="gold_errors.csv")

    print("\n=== CLASSIFICATION REPORT ===")
    rep = stats["report"]
    for lbl in ["ok","need_clarify","no_answer"]:
        if lbl in rep:
            print(lbl, {k: round(rep[lbl][k], 3) for k in ["precision","recall","f1-score"]})

    print("\nConfusion matrix [ok, need_clarify, no_answer] rows:")
    print(stats["confusion_matrix"])

    print("\n=== RETRIEVAL (ok subset) ===")
    for k,v in stats["retrieval_ok"].items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
