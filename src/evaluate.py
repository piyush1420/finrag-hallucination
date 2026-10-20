"""RAGAS evaluation harness (Task T08).

Takes any predictions file written in the shared schema (schema.py) and computes the
three RAGAS metrics we care about for hallucination:

  * faithfulness: is the answer supported by the retrieved context? (low means made up)
  * answer_relevancy: does the answer actually address the question?
  * context_precision: did retrieval put the useful chunks near the top?

The judge is OpenAI GPT-3.5-turbo by default (needs a key in .env); embeddings are
always local sentence-transformers (no OpenAI embedding cost). Pass ``--backend ollama``
to judge with a local model instead (see src/llm.py).

The harness is pipeline-agnostic: point it at the baseline output now, and at the
dense / enhanced / Mistral outputs later (T18) with no code change. It reports mean and
standard deviation across the examples and updates results/eval_ragas.csv.

Run it:
    python src/evaluate.py results/raw_outputs/baseline_bm25.jsonl
    python src/evaluate.py <preds.jsonl> --limit 20              # cheaper smoke test
    python src/evaluate.py <preds.jsonl> --judge-model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from llm import get_backend, make_local_embeddings, make_ragas_llm
from schema import RAGAS_FIELD_MAP, load_run_config, read_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAGAS_CSV = RESULTS_DIR / "eval_ragas.csv"

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


def _to_ragas_dataset(rows: list[dict]):
    """Reshape our prediction rows into the columns RAGAS expects."""
    from datasets import Dataset

    data = {ragas_col: [] for ragas_col in RAGAS_FIELD_MAP.values()}
    for r in rows:
        for our_field, ragas_col in RAGAS_FIELD_MAP.items():
            data[ragas_col].append(r[our_field])
    return Dataset.from_dict(data)


def evaluate_file(preds_path: Path, backend_name: str | None = None,
                  judge_model: str | None = None, limit: int | None = None,
                  max_workers: int = 2):
    import pandas as pd
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.run_config import RunConfig as RagasRunConfig

    load_dotenv(REPO_ROOT / ".env")
    backend = get_backend(backend_name)
    judge_model = judge_model or backend.default_model
    llm = make_ragas_llm(backend, judge_model)
    embeddings = make_local_embeddings()
    print(f"judge: {backend.name} / {judge_model}; embeddings: local MiniLM")

    rows = read_predictions(preds_path)
    if limit:
        rows = rows[:limit]
    dataset = _to_ragas_dataset(rows)

    print(f"scoring {len(rows)} examples with RAGAS ...")
    # Local models are slower and flakier than OpenAI, so give RAGAS a generous
    # timeout and low concurrency instead of hammering the Ollama server.
    ragas_cfg = RagasRunConfig(timeout=180, max_workers=max_workers, max_retries=3)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=ragas_cfg,
        raise_exceptions=False,
    )
    per_row = result.to_pandas()

    # Pull run metadata so the results row says which pipeline/model these numbers are.
    try:
        cfg = load_run_config(preds_path)
    except FileNotFoundError:
        cfg = {"pipeline": preds_path.stem, "model": "unknown"}

    summary = {"pipeline": cfg.get("pipeline"), "model": cfg.get("model"),
               "judge": f"{backend.name}:{judge_model}", "n_examples": len(rows)}
    for metric in METRIC_NAMES:
        if metric in per_row.columns:
            col = per_row[metric].dropna()
            summary[f"{metric}_mean"] = round(float(col.mean()), 4) if len(col) else None
            summary[f"{metric}_std"] = round(float(col.std()), 4) if len(col) else None
            summary[f"{metric}_n"] = int(len(col))

    # Save per-row scores next to the predictions for later error analysis.
    per_row_path = RESULTS_DIR / "raw_outputs" / f"{preds_path.stem}_ragas_perrow.csv"
    per_row_path.parent.mkdir(parents=True, exist_ok=True)
    per_row.to_csv(per_row_path, index=False)

    # Update the shared results table: replace any existing row for this pipeline+model.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary])
    if RAGAS_CSV.exists():
        prev = pd.read_csv(RAGAS_CSV)
        mask = ~((prev["pipeline"] == summary["pipeline"]) & (prev["model"] == summary["model"]))
        summary_df = pd.concat([prev[mask], summary_df], ignore_index=True)
    summary_df.to_csv(RAGAS_CSV, index=False)

    print("\n--- RAGAS summary ---")
    for metric in METRIC_NAMES:
        m, s = summary.get(f"{metric}_mean"), summary.get(f"{metric}_std")
        n = summary.get(f"{metric}_n")
        if m is not None:
            print(f"  {metric:18s}: {m:.4f} +/- {s:.4f}  (scored {n}/{len(rows)})")
        else:
            print(f"  {metric:18s}: no valid scores")
    print(f"\nsaved summary -> {RAGAS_CSV.relative_to(REPO_ROOT)}")
    print(f"saved per-row -> {per_row_path.relative_to(REPO_ROOT)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation harness (Task T08).")
    parser.add_argument("predictions", type=Path,
                        help="a predictions .jsonl written in the shared schema")
    parser.add_argument("--backend", choices=("ollama", "openai"), default=None,
                        help="judge backend (default: $LLM_BACKEND or openai)")
    parser.add_argument("--judge-model", default=None,
                        help="judge model (default: backend's default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N examples (cheaper smoke test)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="RAGAS concurrency (raise to speed up on a local server)")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}")
    evaluate_file(args.predictions, args.backend, args.judge_model, args.limit,
                  max_workers=args.max_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
