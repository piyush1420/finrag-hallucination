"""Pipeline 1 — Baseline RAG: BM25 sparse retrieval + chat LLM  (Task T06).

OVERVIEW
    The simplest end-to-end RAG system and the critical path for M2. For each of the
    150 FinanceBench questions it (1) retrieves the top-k chunks from a BM25 index
    built over the *entire* 84-filing corpus, then (2) asks the generator to answer
    using only those chunks, at temperature 0.

ROLE IN THE PROJECT (RQ1)
    The weakest retriever and the baseline that the Dense (T14) and Enhanced (T15)
    pipelines are measured against. Only *retrieval* changes across those three, which
    is what isolates retrieval quality as the driver of hallucination.

WHY FULL-CORPUS RETRIEVAL
    We index all 84 filings on purpose: a retriever that can surface the *wrong* filing
    is exactly where hallucination appears — the phenomenon this project studies.

PIPELINE FLOW (per question)
    question -> BM25 top-k chunks -> prompt(context + question) -> LLM answer -> row

INPUTS   data/processed/chunks_<strategy>.parquet     (built by preprocess.py, T05)
         data/raw/financebench_open_source.jsonl      (questions + gold answers)
OUTPUTS  results/raw_outputs/baseline_bm25.jsonl      (shared schema; scored by T08/T09)

SHARED HELPERS (defined here, imported by the other pipelines)
    PROMPT_TEMPLATE  .  _chat()  .  load_questions()

GENERATOR
    OpenAI GPT-3.5-turbo by default (key in .env); `--backend ollama` runs a local
    model at no cost. See src/llm.py.

RUN
    python src/pipeline_baseline.py                   # all 150, top-3, GPT-3.5
    python src/pipeline_baseline.py --limit 5         # quick smoke test
    python src/pipeline_baseline.py --backend ollama  # local, no key/cost
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Let the script find its sibling modules whether it's run as `python src/foo.py`
# from the repo root or imported as `src.foo`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from llm import get_backend, make_client
from preprocess import load_chunks
from schema import RunConfig, validate_row, write_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_JSONL = REPO_ROOT / "data" / "raw" / "financebench_open_source.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "raw_outputs" / "baseline_bm25.jsonl"

PROMPT_TEMPLATE = (
    "You are a financial analyst. Answer the question using ONLY the context below, "
    "which is taken from company SEC filings. If the answer is not in the context, "
    "say \"I don't know.\" Be concise and cite the figure exactly as it appears.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenizer for BM25. Deliberately simple."""
    return _WORD_RE.findall(text.lower())


def load_questions(limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(QA_JSONL, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


class BM25Retriever:
    """BM25 over the full chunk corpus. Returns ranked (chunk_id, text) results."""

    def __init__(self, chunks_df):
        self.ids = chunks_df["chunk_id"].tolist()
        self.texts = chunks_df["text"].tolist()
        print(f"tokenizing {len(self.texts)} chunks for BM25 index ...")
        tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def top_k(self, question: str, k: int) -> list[tuple[str, str]]:
        scores = self.bm25.get_scores(_tokenize(question))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.ids[i], self.texts[i]) for i in ranked]


def _chat(client, model: str, prompt: str, temperature: float, max_retries: int = 4) -> str:
    """One chat completion with a simple exponential-backoff retry."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # transient rate-limit / network / server errors
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {exc}")
            time.sleep(delay)
            delay *= 2
    return ""


def run(backend_name: str | None, chunk_strategy: str, top_k: int, model: str | None,
        temperature: float, limit: int | None, out_path: Path) -> Path:
    out_path = Path(out_path).resolve()
    load_dotenv(REPO_ROOT / ".env")
    backend = get_backend(backend_name)
    model = model or backend.default_model
    client = make_client(backend)
    print(f"generator: {backend.name} / {model} (temp {temperature})")

    chunks_df = load_chunks(chunk_strategy)
    retriever = BM25Retriever(chunks_df)
    questions = load_questions(limit)

    rows: list[dict] = []
    for q in tqdm(questions, desc="baseline RAG"):
        retrieved = retriever.top_k(q["question"], top_k)
        contexts = [text for _cid, text in retrieved]
        chunk_ids = [cid for cid, _text in retrieved]
        prompt = PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(contexts),
                                        question=q["question"])
        answer = _chat(client, model, prompt, temperature)

        row = {
            "financebench_id": q["financebench_id"],
            "question": q["question"],
            "contexts": contexts,
            "retrieved_chunk_ids": chunk_ids,
            "generated_answer": answer,
            "gold_answer": q["answer"],
            "doc_name": q["doc_name"],
            "question_type": q["question_type"],
        }
        validate_row(row)
        rows.append(row)

    cfg = RunConfig(
        pipeline="baseline_bm25",
        model=f"{backend.name}:{model}",
        retrieval="bm25",
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        temperature=temperature,
        embedding_model=None,
        notes="Full-corpus BM25 retrieval; answer-only-from-context prompt.",
    )
    path = write_predictions(rows, cfg, out_path)
    print(f"wrote {len(rows)} predictions -> {path.relative_to(REPO_ROOT)}")

    # How often did retrieval even surface the right filing? A cheap sanity signal.
    hit = sum(1 for r in rows if any(cid.split("::")[0] == r["doc_name"]
                                     for cid in r["retrieved_chunk_ids"]))
    print(f"retrieval reached the correct filing for {hit}/{len(rows)} questions "
          f"({hit / len(rows):.1%})")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline BM25 RAG pipeline (Task T06).")
    parser.add_argument("--backend", choices=("ollama", "openai"), default=None,
                        help="LLM backend (default: $LLM_BACKEND or openai)")
    parser.add_argument("--chunk-strategy", choices=("fixed_512", "sentence"),
                        default="fixed_512")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default=None,
                        help="generator model (default: backend's default)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (smoke test)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    run(args.backend, args.chunk_strategy, args.top_k, args.model, args.temperature,
        args.limit, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
