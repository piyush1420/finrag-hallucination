"""Build the T16 retrieval top-k ablation table + figure.

Original author: Piyush. Reworked in review (Bhalchandra) to be reproducible and
consistent with the rest of the paper:
  * scoped to the one hyperparameter we can vary reproducibly with the committed
    pipeline + cached 512-token embeddings -- retrieval depth top-k (3 vs 5). The
    earlier 2x2 that also varied chunk size (256 vs 512) needed 256-token chunk/
    embedding artifacts that are not in the repo, so it is not reproducible here.
  * uses the SAME judge as the main results (GPT-4o-mini), not GPT-3.5, so these
    numbers sit in the same frame as the pipeline comparison.
  * reads the shared aggregate CSVs (eval_ragas.csv + eval_qa_metrics.csv) like the
    other build_*_table scripts; Markdown output only (the project is docx-only).

Configs (dense FAISS + MiniLM, GPT-3.5-turbo generator, temp 0, 512-token chunks):
  * top-3  -> results/raw_outputs/dense_faiss.jsonl   (our main dense run)
  * top-5  -> results/raw_outputs/dense_top5.jsonl    (python src/pipeline_dense.py --top-k 5 --out ...)

Run it (after evaluate.py + qa_metrics.py have scored both):
    python scripts/build_sweep_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_results_table import QA_CSV, RAGAS_CSV, _faith_answered, _retrieval_hit

RESULTS = REPO_ROOT / "results"
MD_OUT = RESULTS / "t16_sweep_results.md"
FIG_OUT = RESULTS / "t16_sweep_chart.png"
CSV_OUT = RESULTS / "t16_sweep_results.csv"

# (pipeline stem, top-k label)
CONFIGS = [("dense_faiss", 3), ("dense_top5", 5)]

COLUMNS = [
    ("top_k", "Top-k"),
    ("retrieval_hit", "Retr@k"),
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
    if col == "top_k":
        return str(int(x))
    if col == "retrieval_hit":
        return f"{x*100:.0f}%"
    if col == "n_answered":
        return f"{int(x)}/150"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def load_rows() -> pd.DataFrame:
    ragas = pd.read_csv(RAGAS_CSV)
    qa = pd.read_csv(QA_CSV)
    merged = ragas.merge(qa, on=["pipeline", "model"], how="outer", suffixes=("_ragas", "_qa"))
    rows = []
    for stem, k in CONFIGS:
        r = merged[merged["pipeline"] == stem]
        if r.empty:
            raise SystemExit(f"no metrics for {stem}; run pipeline_dense (--top-k {k}) + "
                             f"evaluate.py + qa_metrics.py first.")
        r = r.iloc[0].to_dict()
        r["top_k"] = k
        r["retrieval_hit"] = _retrieval_hit(stem)
        r["faith_answered"] = _faith_answered(stem)
        rows.append(r)
    return pd.DataFrame(rows)


def make_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["faithfulness_mean"], width,
           label="Faithfulness", color="#3b6ea5")
    ax.bar([i + width / 2 for i in x], df["retrieval_hit"], width,
           label="Retr@k", color="#c26b3e")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"top-{int(k)}" for k in df["top_k"]])
    ax.set_ylabel("Score (0-1)")
    ax.set_title("T16 top-k ablation (dense, GPT-3.5, GPT-4o-mini judge)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    for i, v in enumerate(df["faithfulness_mean"]):
        ax.text(i - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(df["retrieval_hit"]):
        ax.text(i + width / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)


def main() -> int:
    df = load_rows()
    df.to_csv(CSV_OUT, index=False)

    headers = [h for _c, h in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(c, row.get(c)) for c, _h in COLUMNS) + " |")
    table = "\n".join(lines)

    md = ["# T16 retrieval top-k ablation\n",
          "Dense (FAISS + MiniLM) pipeline, GPT-3.5-turbo generator (temp 0), 512-token "
          "chunks, full 150-question FinanceBench set. Only retrieval depth (top-k) changes. "
          "RAGAS judge = GPT-4o-mini (same as the main results). Retr@k = share of questions "
          "whose top-k retrieval surfaced the correct filing.\n",
          table, "",
          "![top-k ablation](t16_sweep_chart.png)\n"]
    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    make_chart(df)

    print(table)
    print(f"\nsaved -> {MD_OUT.relative_to(REPO_ROOT)}, {CSV_OUT.name}, {FIG_OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
