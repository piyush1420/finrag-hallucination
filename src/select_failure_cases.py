"""Select a stratified 50-case sample of hallucination candidates for annotation (T19).

A "hallucination candidate" here = the model **answered** (did not abstain) but the
answer is **incorrect** (numeric-tolerant EM = 0). Those are the answered-but-wrong
cases the taxonomy (T20/T21) is about; abstentions ("I don't know") are excluded
because a refusal is not a hallucination.

We draw from the **Enhanced (query-rewrite + dense)** pipeline by default: it is our
strongest deployable config and still produces 56 answered-wrong cases, so its residual
hallucinations are the most informative. (The proposal named the BM25 baseline, but the
baseline answers too few questions to yield 50 answered failures.)

The sample is stratified across question_type. Empty columns are added for labeling.

Output: annotations/failure_cases_50.csv  (one row per case, ready to label).

Run:
    python src/select_failure_cases.py                 # enhanced, 50 cases
    python src/select_failure_cases.py --pipeline dense_faiss --n 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "raw_outputs"
QA_JSONL = REPO_ROOT / "data" / "raw" / "financebench_open_source.jsonl"
OUT = REPO_ROOT / "annotations" / "failure_cases_50.csv"

SEED = 42


def _company_map() -> dict[str, str]:
    m = {}
    with open(QA_JSONL, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            m[r["financebench_id"]] = r.get("company", "")
    return m


def _load_pipeline(pipeline: str) -> pd.DataFrame:
    """Join predictions + QA per-row + RAGAS per-row (aligned by question order)."""
    preds = [json.loads(l) for l in open(RAW / f"{pipeline}.jsonl", encoding="utf-8") if l.strip()]
    qa = pd.read_csv(RAW / f"{pipeline}_qa_perrow.csv")
    ragas_fp = RAW / f"{pipeline}_ragas_perrow.csv"
    ragas = pd.read_csv(ragas_fp) if ragas_fp.exists() else None
    company = _company_map()

    rows = []
    for i, p in enumerate(preds):
        cids = p.get("retrieved_chunk_ids", [])
        hit = any(cid.split("::")[0] == p["doc_name"] for cid in cids)
        ctx = "\n\n---\n\n".join(p.get("contexts", []))
        rows.append({
            "financebench_id": p["financebench_id"],
            "pipeline": pipeline,
            "company": company.get(p["financebench_id"], ""),
            "doc_name": p["doc_name"],
            "question_type": p["question_type"],
            "question": p["question"],
            "retrieved_context": (ctx[:1400] + " …[truncated]") if len(ctx) > 1400 else ctx,
            "generated_answer": p["generated_answer"],
            "gold_answer": p["gold_answer"],
            "retrieval_hit": hit,
            "abstained": bool(qa.iloc[i]["abstained"]),
            "f1": round(float(qa.iloc[i]["f1"]), 3),
            "em_numeric": int(qa.iloc[i]["em_numeric"]),
            "faithfulness": (round(float(ragas.iloc[i]["faithfulness"]), 3)
                             if ragas is not None and not pd.isna(ragas.iloc[i]["faithfulness"])
                             else ""),
        })
    return pd.DataFrame(rows)


def stratified_sample(cand: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample n rows spread evenly across question_type.

    Take an equal share from each type; if rounding leaves us short of n, top up
    from whatever is left. Uses a fixed seed so the sample is reproducible.
    """
    types = sorted(cand["question_type"].unique())
    share = n // len(types)
    picked = [g.sample(min(share, len(g)), random_state=SEED)
              for _, g in cand.groupby("question_type")]
    picked = pd.concat(picked)
    if len(picked) < n:  # top up to reach n
        leftover = cand.drop(picked.index)
        picked = pd.concat([picked, leftover.sample(min(n - len(picked), len(leftover)),
                                                     random_state=SEED)])
    return picked.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Select 50 failure cases for annotation (T19).")
    ap.add_argument("--pipeline", default="enhanced_rewrite")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = _load_pipeline(args.pipeline)
    cand = df[(~df["abstained"]) & (df["em_numeric"] == 0)].copy()
    print(f"{args.pipeline}: {len(cand)} answered-but-incorrect candidates "
          f"(of {int((~df['abstained']).sum())} answered)")
    if len(cand) < args.n:
        print(f"  note: only {len(cand)} candidates < {args.n}; taking all.")

    sample = stratified_sample(cand, args.n).reset_index(drop=True)

    # empty columns to fill in while labeling
    sample["hallucination_type"] = ""      # numerical | entity | reasoning | unsupported | other
    sample["secondary_type"] = ""
    sample["notes"] = ""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)

    print(f"\nwrote {len(sample)} cases -> {args.out.relative_to(REPO_ROOT)}")
    print("stratification (question_type):", sample["question_type"].value_counts().to_dict())
    print("companies covered:", sample["company"].nunique())
    print("retrieval missed the filing (answered anyway):",
          int((~sample["retrieval_hit"]).sum()), "of", len(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
