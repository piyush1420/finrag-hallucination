"""Pipeline 3 — Enhanced RAG: LLM query rewriting + dense retrieval  (Task T15).

OVERVIEW
    The Dense pipeline (T14) with one extra step in front: an LLM rewrites each
    question into a retrieval query — spelling out abbreviations and adding the
    financial line-item terms that appear in filings — and dense retrieval then runs
    on the *rewritten* query. Everything downstream (generator, prompt, schema) is
    unchanged, so the only difference vs. Dense is the query used for retrieval.

ROLE IN THE PROJECT (RQ1 / RQ2 / RQ3)
    The strongest deployable retriever in the comparison. Its residual failures are
    the ones we taxonomize in the error analysis (RQ2), and its retrieved context is
    reused by the Mistral generator comparison (RQ3, T17).

PIPELINE FLOW (per question)
    question -> LLM rewrite -> embed -> FAISS top-k -> prompt -> LLM answer -> row
    (each row also logs the rewritten query, for inspection)

INPUTS   data/processed/chunks_<strategy>.parquet
OUTPUTS  results/raw_outputs/enhanced_rewrite.jsonl

RUN
    python src/pipeline_enhanced.py            # all 150, top-3, GPT-3.5
    python src/pipeline_enhanced.py --limit 3  # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from tqdm import tqdm

from llm import LOCAL_EMBEDDING_MODEL, get_backend, make_client
from pipeline_baseline import PROMPT_TEMPLATE, _chat, load_questions
from pipeline_dense import DenseRetriever
from preprocess import load_chunks
from schema import RunConfig, validate_row, write_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "raw_outputs" / "enhanced_rewrite.jsonl"

REWRITE_TEMPLATE = (
    "You are helping search a database of SEC filings (10-K, 10-Q, 8-K). "
    "Rewrite the user's question into a concise search query that will retrieve the "
    "passage containing the answer. Spell out abbreviations, add the specific financial "
    "line-item terms likely to appear in the filing, and drop conversational words. "
    "Return ONLY the rewritten query, nothing else.\n\n"
    "Question: {question}\n"
    "Search query:"
)


def _rewrite(client, model: str, question: str) -> str:
    out = _chat(client, model, REWRITE_TEMPLATE.format(question=question), temperature=0.0)
    out = (out or "").strip().strip('"')
    # Fall back to the original question if the rewrite came back empty.
    return out or question


def run(backend_name, chunk_strategy, top_k, model, temperature, limit, out_path,
        embedding_model=LOCAL_EMBEDDING_MODEL):
    out_path = Path(out_path).resolve()
    load_dotenv(REPO_ROOT / ".env")
    backend = get_backend(backend_name)
    model = model or backend.default_model
    client = make_client(backend)
    print(f"generator + rewriter: {backend.name} / {model} (temp {temperature})")

    chunks_df = load_chunks(chunk_strategy)
    retriever = DenseRetriever(chunks_df, embedding_model, chunk_strategy)
    questions = load_questions(limit)

    rows = []
    for q in tqdm(questions, desc="enhanced RAG"):
        rewritten = _rewrite(client, model, q["question"])
        retrieved = retriever.top_k(rewritten, top_k)
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
            "rewritten_query": rewritten,   # extra field for the comparison note
        }
        validate_row(row)
        rows.append(row)

    cfg = RunConfig(pipeline="enhanced_rewrite", model=f"{backend.name}:{model}",
                    retrieval="dense+rewrite", chunk_strategy=chunk_strategy, top_k=top_k,
                    temperature=temperature, embedding_model=embedding_model,
                    notes="LLM query rewriting before dense retrieval; same prompt as baseline.")
    path = write_predictions(rows, cfg, out_path)
    print(f"wrote {len(rows)} predictions -> {path.relative_to(REPO_ROOT)}")

    hit = sum(1 for r in rows if any(cid.split("::")[0] == r["doc_name"]
                                     for cid in r["retrieved_chunk_ids"]))
    print(f"retrieval reached the correct filing for {hit}/{len(rows)} "
          f"({hit / len(rows):.1%})")
    # Show a couple of rewrites as a sanity check.
    for r in rows[:2]:
        print(f"  original : {r['question'][:80]}")
        print(f"  rewritten: {r['rewritten_query'][:80]}")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Enhanced RAG: query rewrite + dense (Task T15).")
    p.add_argument("--backend", choices=("ollama", "openai"), default=None)
    p.add_argument("--chunk-strategy", choices=("fixed_512", "sentence"), default="fixed_512")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    run(a.backend, a.chunk_strategy, a.top_k, a.model, a.temperature, a.limit, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
