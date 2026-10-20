"""Pipeline 2 — Dense RAG: FAISS + MiniLM semantic retrieval  (Task T14).

OVERVIEW
    Same end-to-end system as the baseline, but retrieval is *semantic* instead of
    lexical: every chunk is embedded with sentence-transformers all-MiniLM-L6-v2 and
    the top-k are retrieved by cosine similarity from a FAISS index over all 84
    filings. Generator, prompt, temperature, and output schema are identical to the
    baseline (T06) — the only thing that changes is retrieval.

ROLE IN THE PROJECT (RQ1)
    The middle rung of the retrieval comparison (BM25 -> Dense -> Enhanced). It also
    hosts the single-document *oracle* ablation via `--scope single_doc`, which limits
    retrieval to the question's own filing to give a known-filing upper bound.

PIPELINE FLOW (per question)
    question -> embed -> FAISS cosine top-k -> prompt(context + question) -> LLM -> row

EMBEDDINGS
    Chunk vectors are L2-normalized (inner product == cosine) and cached per
    (chunk-strategy, model) at data/processed/emb_*.npy, so the Enhanced pipeline (T15)
    and the chunk-size ablations reuse them without re-encoding.

INPUTS   data/processed/chunks_<strategy>.parquet
OUTPUTS  results/raw_outputs/dense_faiss.jsonl   (or dense_singledoc.jsonl for the oracle)

RUN
    python src/pipeline_dense.py                                     # all 150, top-3, GPT-3.5
    python src/pipeline_dense.py --scope single_doc                 # known-filing oracle ceiling
    python src/pipeline_dense.py --chunk-strategy fixed_256 --top-k 5 # retrieval ablations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from llm import LOCAL_EMBEDDING_MODEL, get_backend, make_client
from pipeline_baseline import PROMPT_TEMPLATE, _chat, load_questions
from preprocess import load_chunks
from schema import RunConfig, validate_row, write_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_OUT = REPO_ROOT / "results" / "raw_outputs" / "dense_faiss.jsonl"


class DenseRetriever:
    """MiniLM + FAISS cosine retriever over the full chunk corpus.

    Embeddings are L2-normalized and indexed with inner product, so inner product
    equals cosine similarity. Chunk embeddings are cached per (strategy, model).
    """

    def __init__(self, chunks_df, model_name: str = LOCAL_EMBEDDING_MODEL,
                 strategy: str = "fixed_512"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.ids = chunks_df["chunk_id"].tolist()
        self.texts = chunks_df["text"].tolist()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        cache = PROCESSED_DIR / f"emb_{strategy}_{model_name.split('/')[-1]}.npy"
        if cache.exists():
            print(f"loading cached chunk embeddings <- {cache.name}")
            emb = np.load(cache)
        else:
            print(f"embedding {len(self.texts)} chunks with {model_name} ...")
            emb = self.model.encode(self.texts, batch_size=64, show_progress_bar=True,
                                    convert_to_numpy=True, normalize_embeddings=True)
            np.save(cache, emb)
            print(f"cached chunk embeddings -> {cache.name}")
        self.emb = emb.astype("float32")

        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

        # doc_name -> row indices, for the single-document ("known filing") scope.
        self.doc_to_idx: dict[str, list[int]] = {}
        for i, cid in enumerate(self.ids):
            self.doc_to_idx.setdefault(cid.split("::")[0], []).append(i)

    def _embed_query(self, query: str):
        return self.model.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")

    def top_k(self, query: str, k: int, doc_name: str | None = None) -> list[tuple[str, str]]:
        q = self._embed_query(query)
        if doc_name is None:                      # full-corpus retrieval
            _scores, idx = self.index.search(q, k)
            idx = idx[0]
        else:                                     # restrict to one filing's chunks
            cand = self.doc_to_idx.get(doc_name, [])
            if not cand:
                return []
            sims = (self.emb[cand] @ q[0])
            order = np.argsort(sims)[::-1][:k]
            idx = [cand[j] for j in order]
        return [(self.ids[i], self.texts[i]) for i in idx]


def run(backend_name, chunk_strategy, top_k, model, temperature, limit, out_path,
        embedding_model=LOCAL_EMBEDDING_MODEL, scope="corpus"):
    out_path = Path(out_path).resolve()
    load_dotenv(REPO_ROOT / ".env")
    backend = get_backend(backend_name)
    model = model or backend.default_model
    client = make_client(backend)
    single_doc = scope == "single_doc"
    print(f"generator: {backend.name} / {model} (temp {temperature}) | scope: {scope}")

    chunks_df = load_chunks(chunk_strategy)
    retriever = DenseRetriever(chunks_df, embedding_model, chunk_strategy)
    questions = load_questions(limit)

    rows = []
    for q in tqdm(questions, desc="dense RAG"):
        retrieved = retriever.top_k(q["question"], top_k,
                                    doc_name=q["doc_name"] if single_doc else None)
        contexts = [t for _c, t in retrieved]
        chunk_ids = [c for c, _t in retrieved]
        prompt = PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(contexts),
                                        question=q["question"])
        answer = _chat(client, model, prompt, temperature)
        row = {
            "financebench_id": q["financebench_id"], "question": q["question"],
            "contexts": contexts, "retrieved_chunk_ids": chunk_ids,
            "generated_answer": answer, "gold_answer": q["answer"],
            "doc_name": q["doc_name"], "question_type": q["question_type"],
        }
        validate_row(row)
        rows.append(row)

    pipeline = "dense_singledoc" if single_doc else "dense_faiss"
    note = ("FAISS cosine over MiniLM; retrieval restricted to the question's own filing "
            "(known-document upper bound)." if single_doc else
            "FAISS cosine over MiniLM chunk embeddings; same prompt as baseline.")
    cfg = RunConfig(pipeline=pipeline, model=f"{backend.name}:{model}",
                    retrieval="dense_singledoc" if single_doc else "dense",
                    chunk_strategy=chunk_strategy, top_k=top_k,
                    temperature=temperature, embedding_model=embedding_model, notes=note)
    path = write_predictions(rows, cfg, out_path)
    print(f"wrote {len(rows)} predictions -> {path.relative_to(REPO_ROOT)}")

    hit = sum(1 for r in rows if any(cid.split("::")[0] == r["doc_name"]
                                     for cid in r["retrieved_chunk_ids"]))
    print(f"retrieval reached the correct filing for {hit}/{len(rows)} "
          f"({hit / len(rows):.1%})")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Dense FAISS RAG pipeline (Task T14).")
    p.add_argument("--backend", choices=("ollama", "openai"), default=None)
    p.add_argument("--chunk-strategy", choices=("fixed_512", "fixed_256", "sentence"), default="fixed_512")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--scope", choices=("corpus", "single_doc"), default="corpus",
                   help="'corpus' = retrieve over all filings; 'single_doc' = restrict to "
                        "the question's own filing (known-document upper bound)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or (DEFAULT_OUT.parent / ("dense_singledoc.jsonl" if a.scope == "single_doc"
                                          else "dense_faiss.jsonl"))
    run(a.backend, a.chunk_strategy, a.top_k, a.model, a.temperature, a.limit, out,
        scope=a.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
