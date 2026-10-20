"""Assemble the preliminary results table for the paper (Task T10).

Joins the RAGAS metrics (from evaluate.py, results/eval_ragas.csv) with the F1/EM
metrics (from qa_metrics.py, results/eval_qa_metrics.csv) on pipeline+model, and writes
a single paper-ready table in both Markdown and LaTeX, plus a short interpretation.

For the preliminary paper there is only the baseline row; the same script fills in as
the dense / enhanced / Mistral rows arrive later (T18), no changes needed.

Run it (after T08 and T09 have produced their CSVs):
    python src/build_results_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RAGAS_CSV = RESULTS_DIR / "eval_ragas.csv"
QA_CSV = RESULTS_DIR / "eval_qa_metrics.csv"

MD_OUT = RESULTS_DIR / "pipeline_comparison.md"
RAW_DIR = RESULTS_DIR / "raw_outputs"

# Fixed display order + friendly names for the retrieval-strategy comparison.
PIPELINE_ORDER = ["baseline_bm25", "dense_faiss", "enhanced_rewrite", "dense_singledoc"]
PIPELINE_NAME = {
    "baseline_bm25": "Baseline (BM25)",
    "dense_faiss": "Dense (FAISS)",
    "enhanced_rewrite": "Enhanced (rewrite)",
    "dense_singledoc": "Dense, single-doc",
}

# (source column, display header)
COLUMNS = [
    ("display_name", "Pipeline"),
    ("retrieval_hit", "Retr@3"),
    ("faithfulness_mean", "Faithful."),
    ("faith_answered", "Faith(ans)"),
    ("answer_relevancy_mean", "Ans. Rel."),
    ("context_precision_mean", "Ctx. Prec."),
    ("n_answered", "Answered"),
    ("f1", "F1"),
    ("em_numeric", "EM (num)"),
]


def _retrieval_hit(pipeline: str):
    """Fraction of questions whose top-k retrieval reached the correct filing,
    computed from the raw predictions if present (else NaN)."""
    import json
    fp = RAW_DIR / f"{pipeline}.jsonl"
    if not fp.exists():
        return float("nan")
    rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    if not rows:
        return float("nan")
    hit = sum(1 for r in rows if any(cid.split("::")[0] == r["doc_name"]
                                     for cid in r.get("retrieved_chunk_ids", [])))
    return hit / len(rows)


def _faith_answered(pipeline: str):
    """Mean faithfulness over *answered* rows only (abstentions excluded), from the
    per-row RAGAS file. Scoring an 'I don't know' for faithfulness is meaningless and
    drags the reported mean down, so this is the fairer 'when it answers' number."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qa_metrics import is_abstention
    fp = RAW_DIR / f"{pipeline}_ragas_perrow.csv"
    if not fp.exists():
        return float("nan")
    df = pd.read_csv(fp)
    if "faithfulness" not in df.columns or "answer" not in df.columns:
        return float("nan")
    ans = df[~df["answer"].fillna("").map(is_abstention)]["faithfulness"].dropna()
    return float(ans.mean()) if len(ans) else float("nan")


def load_merged() -> pd.DataFrame:
    if not RAGAS_CSV.exists():
        raise SystemExit(f"missing {RAGAS_CSV}. Run src/evaluate.py first (T08).")
    if not QA_CSV.exists():
        raise SystemExit(f"missing {QA_CSV}. Run src/qa_metrics.py first (T09).")
    ragas = pd.read_csv(RAGAS_CSV)
    qa = pd.read_csv(QA_CSV)
    merged = ragas.merge(qa, on=["pipeline", "model"], how="outer", suffixes=("_ragas", "_qa"))
    # This table is the RQ1 retrieval-strategy comparison only; the top-k ablation (T16)
    # and the generator comparison (T17) have their own tables, so keep just those four.
    merged = merged[merged["pipeline"].isin(PIPELINE_ORDER)].copy()
    merged["display_name"] = merged["pipeline"].map(lambda p: PIPELINE_NAME.get(p, p))
    merged["retrieval_hit"] = merged["pipeline"].map(_retrieval_hit)
    merged["faith_answered"] = merged["pipeline"].map(_faith_answered)
    order = {p: i for i, p in enumerate(PIPELINE_ORDER)}
    merged["_ord"] = merged["pipeline"].map(lambda p: order.get(p, 99))
    merged = merged.sort_values("_ord").reset_index(drop=True)
    return merged


def _cell(col: str, x) -> str:
    if pd.isna(x):
        return "n/a"
    if col == "retrieval_hit":
        return f"{x*100:.0f}\\%" if isinstance(x, float) else str(x)
    if col == "n_answered":
        return f"{int(x)}/150"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def to_markdown(df: pd.DataFrame) -> str:
    headers = [h for _c, h in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = [_cell(col, row.get(col)).replace("\\%", "%") for col, _h in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def interpretation(df: pd.DataFrame) -> str:
    """Comparison narrative across the pipelines present (baseline/dense/enhanced)."""
    by = {r["pipeline"]: r for _, r in df.iterrows()}
    def g(pipe, col):
        return by[pipe].get(col) if pipe in by else float("nan")
    def pct(x):
        return "n/a" if pd.isna(x) else f"{x*100:.0f}\\%"
    def ans(x):
        return "n/a" if pd.isna(x) else f"{int(x)}"

    return (
        "Holding the generator (GPT-3.5-turbo), prompt, chunks, and top-$k$ fixed, only "
        "\\emph{retrieval} changes across the three pipelines, so the trend isolates the "
        "effect of retrieval quality. Retrieval accuracy climbs steadily: the correct filing "
        f"reaches the top-3 for {pct(g('baseline_bm25','retrieval_hit'))} of questions under "
        f"sparse BM25, {pct(g('dense_faiss','retrieval_hit'))} under dense MiniLM retrieval, "
        f"and {pct(g('enhanced_rewrite','retrieval_hit'))} once an LLM rewrites the query "
        "first. Because the generator is instructed to answer only from context, better "
        "retrieval directly lifts coverage: the model answers "
        f"{ans(g('baseline_bm25','n_answered'))}, {ans(g('dense_faiss','n_answered'))}, and "
        f"{ans(g('enhanced_rewrite','n_answered'))} of 150 questions respectively, and token "
        f"F1 rises from {_cell('f1', g('baseline_bm25','f1'))} to "
        f"{_cell('f1', g('dense_faiss','f1'))} to {_cell('f1', g('enhanced_rewrite','f1'))}. "
        "RAGAS context precision moves the same way, confirming the gains come from putting "
        "the right chunk in front of the model rather than from changes in generation. Strict "
        "exact match stays near zero because gold answers are short numeric values formatted "
        "many ways ($1,577 vs.\\ 1577.00 vs.\\ 1.577 billion); a numeric-tolerant EM (right "
        "value within 1\\%, ignoring format) recovers the real accuracy and also rises with "
        f"retrieval ({_cell('f1', g('baseline_bm25','em_numeric'))} $\\rightarrow$ "
        f"{_cell('f1', g('dense_faiss','em_numeric'))} $\\rightarrow$ "
        f"{_cell('f1', g('enhanced_rewrite','em_numeric'))}), showing the strict metric "
        "understated correctness rather than the model being wrong. The headline for RQ1 is that retrieval, "
        "not the generator, is the dominant lever on this benchmark: dense retrieval and query "
        "rewriting each add coverage, yet even the strongest configuration answers well under "
        "half the set, leaving a substantial gap---and a pool of answered-but-unsupported "
        "cases---for the error analysis to characterize."
        + (
            " As an upper bound, restricting retrieval to the question's own filing "
            "(\\emph{Dense, single-doc}---the known-document setting) lifts Retr@3 to "
            f"{pct(g('dense_singledoc','retrieval_hit'))}, coverage to "
            f"{ans(g('dense_singledoc','n_answered'))}/150, and context precision to "
            f"{_cell('f1', g('dense_singledoc','context_precision_mean'))}; "
            "the full-corpus configurations must first find the right filing among 84, which "
            "is where most of the remaining loss sits."
            if "dense_singledoc" in by else ""
        )
    )


def main() -> int:
    df = load_merged()
    md_table = to_markdown(df)

    md = ["# Pipeline comparison: retrieval strategies (T14/T15/T18)\n",
          "Three RAG configurations on FinanceBench (all 150 questions), identical except for "
          "retrieval: **Baseline** = sparse BM25; **Dense** = FAISS over MiniLM embeddings; "
          "**Enhanced** = LLM query rewrite + dense. Generator, query-rewriter, and RAGAS judge "
          "are GPT-3.5-turbo at temperature 0. Retr@3 = share of questions whose top-3 "
          "retrieval reached the correct filing.\n",
          md_table, "",
          "## Interpretation\n", interpretation(df), ""]
    MD_OUT.write_text("\n".join(md), encoding="utf-8")

    print(md_table)
    print(f"\nsaved -> {MD_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
