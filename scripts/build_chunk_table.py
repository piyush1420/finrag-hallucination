"""Build the chunk-size ablation table: 256 vs 512-token chunks (Task T16, RQ1).

The proposal named chunk size (256 vs 512 tokens) as a tuned hyperparameter. This
compares the Dense pipeline at both sizes, holding everything else fixed (top-3,
GPT-3.5 generator, temp 0, GPT-4o-mini judge), so the only variable is chunk size.

Configs (both dense FAISS + MiniLM):
  * 512-tok -> results/raw_outputs/dense_faiss.jsonl   (our main dense run)
  * 256-tok -> results/raw_outputs/dense_256tok.jsonl  (python src/pipeline_dense.py
               --chunk-strategy fixed_256 --top-k 3 --out ...)

Run it (after evaluate.py + qa_metrics.py have scored both):
    python scripts/build_chunk_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from build_results_table import QA_CSV, RAGAS_CSV, _faith_answered, _retrieval_hit

MD_OUT = REPO_ROOT / "results" / "chunk_size_results.md"

# (pipeline stem, chunk-size label)
CONFIGS = [("dense_256tok", 256), ("dense_faiss", 512)]

COLUMNS = [
    ("chunk_tokens", "Chunk (tok)"),
    ("retrieval_hit", "Retr@3"),
    ("faithfulness_mean", "Faithful."),
    ("faith_answered", "Faith(ans)"),
    ("answer_relevancy_mean", "Ans. Rel."),
    ("context_precision_mean", "Ctx. Prec."),
    ("n_answered", "Answered"),
    ("f1", "F1"),
    ("em_numeric", "EM (num)"),
]


def _cell(col: str, x) -> str:
    if pd.isna(x):
        return "n/a"
    if col == "chunk_tokens":
        return str(int(x))
    if col == "retrieval_hit":
        return f"{x*100:.0f}%"
    if col == "n_answered":
        return f"{int(x)}/150"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    ragas = pd.read_csv(RAGAS_CSV)
    qa = pd.read_csv(QA_CSV)
    merged = ragas.merge(qa, on=["pipeline", "model"], how="outer", suffixes=("_ragas", "_qa"))

    rows = []
    for stem, tok in CONFIGS:
        r = merged[merged["pipeline"] == stem]
        if r.empty:
            raise SystemExit(f"no metrics for {stem}; run pipeline_dense (fixed_{tok}) + "
                             f"evaluate.py + qa_metrics.py first.")
        r = r.iloc[0].to_dict()
        r["chunk_tokens"] = tok
        r["retrieval_hit"] = _retrieval_hit(stem)
        r["faith_answered"] = _faith_answered(stem)
        rows.append(r)

    df = pd.DataFrame(rows)
    headers = [h for _c, h in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(c, row.get(c)) for c, _h in COLUMNS) + " |")
    table = "\n".join(lines)

    md = ["# Chunk-size ablation: 256 vs 512-token chunks (T16, RQ1)\n",
          "Dense (FAISS + MiniLM) pipeline, GPT-3.5-turbo (temp 0), top-3 retrieval, full "
          "150-question set. Only chunk size changes. RAGAS judge = GPT-4o-mini.\n",
          table, ""]
    MD_OUT.write_text("\n".join(md), encoding="utf-8")

    print(table)
    print(f"\nsaved -> {MD_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
