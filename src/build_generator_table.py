"""Assemble the RQ3 generator-comparison table (Task T17).

RQ3: does an open-source instruction-tuned model hallucinate less than GPT-3.5 given
the *same* retrieved context? We compare the two generators on the identical Enhanced
retrieval (query-rewrite + dense, 512-tok, top-3) -- retrieval is held fixed, so any
difference is the generator alone. Reads the same metric CSVs as the retrieval table
(eval_ragas.csv + eval_qa_metrics.csv) and writes a small Markdown table.

Run it (after evaluate.py + qa_metrics.py have scored both pipelines):
    python src/build_generator_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from build_results_table import QA_CSV, RAGAS_CSV, _faith_answered, _retrieval_hit

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_OUT = REPO_ROOT / "results" / "generator_comparison.md"

# pipeline stem -> generator display name (same Enhanced retrieval for both)
GENERATORS = [
    ("enhanced_rewrite", "GPT-3.5-turbo"),
    ("enhanced_mistral", "Mistral-7B-Instruct"),
]

COLUMNS = [
    ("generator", "Generator"),
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
    for stem, name in GENERATORS:
        r = merged[merged["pipeline"] == stem]
        if r.empty:
            print(f"warning: no metrics row for {stem}; skipping")
            continue
        r = r.iloc[0].to_dict()
        r["generator"] = name
        r["faith_answered"] = _faith_answered(stem)
        rows.append(r)

    df = pd.DataFrame(rows)
    headers = [h for _c, h in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(c, row.get(c)) for c, _h in COLUMNS) + " |")
    table = "\n".join(lines)

    hit = _retrieval_hit("enhanced_mistral")
    md = ["# Generator comparison on identical retrieval (RQ3, T17)\n",
          "Both generators answer over the **same** retrieved context (Enhanced pipeline: "
          "query-rewrite + dense, 512-token chunks, top-3; Retr@3 = "
          f"{hit*100:.0f}% for both). Only the answer generator differs. RAGAS judge = "
          "GPT-4o-mini; embeddings = MiniLM. Mistral-7B-Instruct runs locally via Ollama.\n",
          table, ""]
    MD_OUT.write_text("\n".join(md), encoding="utf-8")

    print(table)
    print(f"\nsaved -> {MD_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
