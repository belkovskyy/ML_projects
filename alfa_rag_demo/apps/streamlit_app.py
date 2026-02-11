from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# --- Make sure local package (src/) is importable even without "pip install -e ."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from alfa_rag.retrieval import Retriever
from alfa_rag.rerank import CrossEncoderReranker
from alfa_rag.service import Pipeline
from alfa_rag.clarify import build_clarify_bank


st.set_page_config(page_title="Alfa RAG demo", layout="wide")


def _list_parquets(data_dir: Path) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted([p.name for p in data_dir.glob("*.parquet")])


@st.cache_resource(show_spinner="Загружаю ретривер (эмбеддинги/FAISS)…")
def _load_retriever(data_dir: str, chunks_filename: str, embed_model: str, device: str) -> Retriever:
    d = Path(data_dir)
    return Retriever.from_data_dir(
        data_dir=d,
        chunks_path=chunks_filename,
        embed_model=embed_model,
        device=device,
    )


@st.cache_resource(show_spinner="Загружаю reranker…")
def _load_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)


def _make_embed_fn(retriever: Retriever):
    """Build an embedding function compatible with ClarifyBank.

    NOTE: Retriever has an internal SentenceTransformer embedder; we reuse it.
    """

    def embed_fn(texts: list[str]) -> np.ndarray:
        xs = [f"query: {str(t)}" for t in texts]
        vecs = retriever.embedder.encode(xs, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    return embed_fn


def _maybe_load_clarify_bank(path: str, embed_fn) -> Optional[object]:
    """clarify_bank опционален. Если файла нет или он кривой — просто вернём None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return build_clarify_bank(p, embed_fn)
    except Exception:
        return None


st.title("Alfa RAG demo")
st.caption("Демо ретривала + guardrails. Если страница пустая — значит в скрипте нет ни одного вызова `st.*` 🙂")

with st.sidebar:
    st.header("Настройки")

    data_dir_default = str((PROJECT_ROOT / "data").resolve())
    data_dir = st.text_input("Папка с данными (data_dir)", value=data_dir_default)

    data_dir_p = Path(data_dir)
    parquet_candidates = _list_parquets(data_dir_p)
    if parquet_candidates:
        chunks_filename = st.selectbox("Файл чанков (parquet)", parquet_candidates, index=0)
    else:
        chunks_filename = st.text_input("Файл чанков (parquet)", value="chunks_websites.parquet")
        st.info("В папке data_dir не нашёл ни одного *.parquet — укажи имя файла вручную.")

    embed_model = st.text_input("Embedding model (SentenceTransformers)", value="intfloat/multilingual-e5-base")
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)

    top_k = st.slider("top_k", min_value=1, max_value=20, value=8, step=1)

    use_rerank = st.checkbox("Использовать reranker (дороже, но точнее)", value=False)
    rerank_model = st.text_input("Reranker model", value="cross-encoder/ms-marco-MiniLM-L-6-v2", disabled=not use_rerank)

    use_llm = st.checkbox("Использовать LLM (Ollama на localhost:11434)", value=False)
    # По умолчанию — лёгкая, но адекватная модель. Можно поменять на любую установленную в Ollama.
    llm_model = st.text_input("Ollama model", value="qwen2.5:3b", disabled=not use_llm)

    clarify_mode = st.selectbox(
        "Need_clarify режим (как формулировать уточнение)",
        ["llm", "gold_then_llm", "gold", "off"],
        index=0,
        help=(
            "llm: LLM формулирует уточнение по подсказкам\n"
            "gold_then_llm: дополнительно подаём LLM текст подсказки из gold_labels.csv\n"
            "gold: показываем готовый текст уточнения из gold_labels.csv (если есть)\n"
            "off: показываем базовое сообщение из правил"
        ),
    )

    gold_use_min_sim = st.slider(
        "Порог похожести для использования gold-уточнения как final",
        min_value=0.70,
        max_value=0.99,
        value=0.86,
        step=0.01,
        help=(
            "Если запрос очень похож на пример из gold_labels.csv (need_clarify), "
            "то в режимах gold / gold_then_llm будем показывать это уточнение как основной final."
        ),
        disabled=clarify_mode not in {"gold", "gold_then_llm"},
    )

    clarify_bank_path = st.text_input(
        "Файл gold_labels.csv (опционально, для need_clarify подсказок)",
        value=str((PROJECT_ROOT / "gold_labels.csv").resolve()),
    )

    if st.button("Очистить кэш и перезагрузить"):
        st.cache_resource.clear()
        st.rerun()

chunks_path = Path(data_dir) / chunks_filename
if not chunks_path.exists():
    st.error(
        f"Не найден файл чанков: {chunks_path}\n\n"
        "Нужен parquet с колонками хотя бы: web_id, title, url, text.\n"
        "После этого перезапусти приложение."
    )
    st.stop()

retriever = _load_retriever(data_dir, chunks_filename, embed_model, device)
embed_fn = _make_embed_fn(retriever)

reranker = None
if use_rerank:
    try:
        reranker = _load_reranker(rerank_model)
    except Exception as e:
        st.warning("Не смог загрузить reranker — продолжаю без него.")
        st.exception(e)
        reranker = None

pipeline = Pipeline(retriever=retriever, reranker=reranker, decision_model=None)
clarify_bank = _maybe_load_clarify_bank(clarify_bank_path, embed_fn)

query = st.text_area("Запрос", height=90, placeholder="Например: Как восстановить доступ в Альфа-Онлайн?")
col1, col2 = st.columns([1, 3])
with col1:
    run = st.button("Запустить", type="primary")
with col2:
    st.caption("Если включил LLM, убедись что Ollama запущена и модель скачана (`ollama pull ...`).")

if run and query.strip():
    with st.spinner("Ищу и принимаю решение…"):
        res = pipeline.ask(
            query=query.strip(),
            k_docs=top_k,
            k_chunks=80,
            clarify_bank=clarify_bank,
            embed_fn=embed_fn,
            clarify_topk=4,
            clarify_min_sim=0.82,
            clarify_gold_use_min_sim=float(gold_use_min_sim),
            clarify_mode=clarify_mode,
            use_llm=use_llm,
            llm_for="both",
            llm_model=llm_model if use_llm else None,
        )

    out = res.get("out", {})
    final = res.get("final", "")
    hint = res.get("hint")

    status = out.get("status")
    message = out.get("message")

    st.subheader("Результат")
    st.write(f"**status:** `{status}`")
    if message:
        st.write(message)
    if final:
        st.markdown("### final")
        st.write(final)

    if hint:
        st.markdown("### hint (need_clarify)")
        st.info(hint.get("best_text", ""))
        with st.expander("Похожие примеры / коды"):
            st.write("**codes:**", hint.get("codes"))
            st.write("**examples:**", hint.get("examples"))
            st.write("**top_sim:**", hint.get("top_sim"))

    st.markdown("### Топ документов")
    results_df = out.get("results")
    if isinstance(results_df, pd.DataFrame) and len(results_df):
        show_cols = [c for c in ["score", "web_id", "title", "url", "text"] if c in results_df.columns]
        st.dataframe(results_df[show_cols], use_container_width=True)
    else:
        st.write("Документы не найдены / results пустой.")

    with st.expander("DEBUG: meta"):
        st.json(out.get("meta", {}))
else:
    st.info("Введи запрос и нажми **Запустить**.")

