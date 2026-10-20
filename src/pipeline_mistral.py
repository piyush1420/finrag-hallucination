"""Pipeline 4 — Mistral-7B generator on the Enhanced retrieval  (Task T17, RQ3).

OVERVIEW
    A *generator* swap, not a retrieval change. RQ3 asks whether an open, instruction-
    tuned model hallucinates less than GPT-3.5 given the *same* retrieved context. To
    keep it clean we do NOT re-retrieve: we reuse the exact contexts the GPT-3.5
    Enhanced run (T15, enhanced_rewrite.jsonl) already retrieved and only regenerate
    the answer with Mistral-7B-Instruct, using the identical prompt and schema.

WHY THIS DESIGN
    Holding retrieval fixed isolates the generator — any metric difference is the
    model, not retrieval. It is the opposite control to the RQ1 pipelines (which hold
    the generator fixed and vary retrieval).

RUNTIME
    Mistral runs locally via Ollama (no GPU, no API key, no cost):
        ollama serve &                    # start once
        ollama pull mistral:7b-instruct

PIPELINE FLOW (per question)
    reuse Enhanced context -> prompt(context + question) -> Mistral answer -> row

INPUTS   results/raw_outputs/enhanced_rewrite.jsonl   (contexts from T15)
OUTPUTS  results/raw_outputs/enhanced_mistral.jsonl

RUN
    python src/pipeline_mistral.py             # all 150, reuse enhanced contexts
    python src/pipeline_mistral.py --limit 3   # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from llm import get_backend, make_client
from pipeline_baseline import PROMPT_TEMPLATE, _chat
from schema import RunConfig, read_predictions, validate_row, write_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "results" / "raw_outputs" / "enhanced_rewrite.jsonl"
DEFAULT_OUT = REPO_ROOT / "results" / "raw_outputs" / "enhanced_mistral.jsonl"


def run(model, temperature, limit, source, out_path):
    out_path = Path(out_path).resolve()
    backend = get_backend("ollama")
    client = make_client(backend)
    print(f"generator: {backend.name} / {model} (temp {temperature}); "
          f"retrieval reused from {Path(source).name}")

    source_rows = read_predictions(source)
    if limit:
        source_rows = source_rows[:limit]

    rows = []
    for src in tqdm(source_rows, desc="mistral (reused contexts)"):
        prompt = PROMPT_TEMPLATE.format(context="\n\n---\n\n".join(src["contexts"]),
                                        question=src["question"])
        answer = _chat(client, model, prompt, temperature)
        row = {
            "financebench_id": src["financebench_id"], "question": src["question"],
            "contexts": src["contexts"], "retrieved_chunk_ids": src["retrieved_chunk_ids"],
            "generated_answer": answer, "gold_answer": src["gold_answer"],
            "doc_name": src["doc_name"], "question_type": src["question_type"],
        }
        validate_row(row)
        rows.append(row)

    cfg = RunConfig(pipeline="enhanced_mistral", model=f"{backend.name}:{model}",
                    retrieval="dense+rewrite", chunk_strategy="fixed_512", top_k=3,
                    temperature=temperature,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    notes="RQ3 generator swap: Mistral-7B-Instruct answering over the "
                          "exact contexts from the GPT-3.5 enhanced_rewrite run.")
    path = write_predictions(rows, cfg, out_path)
    print(f"wrote {len(rows)} predictions -> {path.relative_to(REPO_ROOT)}")
    for r in rows[:2]:
        print(f"  Q: {r['question'][:70]}")
        print(f"  A: {str(r['generated_answer'])[:70]}")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Mistral-7B generator, reused contexts (Task T17).")
    p.add_argument("--model", default="mistral:7b-instruct")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = p.parse_args()
    run(a.model, a.temperature, a.limit, a.source, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
