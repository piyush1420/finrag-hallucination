"""SQuAD-style token F1 and Exact Match (Task T09).

RAGAS tells us whether an answer is grounded in the retrieved context; it does not
tell us whether the answer is actually *right*. F1 and EM against the human gold answer
fill that gap, and the gap between them is exactly the interesting case for this project:
an answer can be faithful to its context but still wrong (high faithfulness, low F1),
which is a retrieval failure dressed up as a confident answer.

Normalization follows the SQuAD script: lowercase, strip punctuation and articles,
collapse whitespace. EM is 1 only if the normalized strings match exactly; token F1 is
the harmonic mean of precision/recall over the shared normalized tokens.

Financial answers are short and formatted in many ways ("$1,577", "1577.00", "1.577
billion"), so EM is expected to be low. We also report F1 on just the answered subset
(dropping the "I don't know" abstentions) so the number is not swamped by the baseline's
81% abstention rate.

Run it:
    python src/qa_metrics.py results/raw_outputs/baseline_bm25.jsonl
"""
from __future__ import annotations

import argparse
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import load_run_config, read_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
QA_CSV = RESULTS_DIR / "eval_qa_metrics.csv"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
# Answers that are really abstentions, not attempts.
_ABSTAIN = ("i don't know", "i do not know", "cannot be determined", "not enough information")


def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation + articles, squeeze spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    if not pred_toks or not gold_toks:
        # if either side is empty, F1 is 1 only when both are empty
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def is_abstention(answer: str) -> bool:
    low = answer.lower()
    return any(phrase in low for phrase in _ABSTAIN)


# --- Numeric-tolerant matching -------------------------------------------------
# Strict SQuAD EM treats "$1,577", "1577.00", and "1.577 billion" as different
# strings (it even splits on the decimal point). Financial gold answers are numeric
# and formatted many ways, so strict EM understates correctness. This measures
# whether the model produced the right *value*, ignoring format. Reported alongside
# (never replacing) strict EM/F1.

_UNIT_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6,
               "thousand": 1e3, "k": 1e3}
_NUM_RE = re.compile(
    r"\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(%|percent|billion|million|thousand|bn|mn|k)?",
    re.I,
)


def _to_floats(text: str) -> set[float]:
    """Extract numeric values from text as floats, including unit-scaled variants
    (so '1.577 billion' can match '1577' when the millions unit is implied)."""
    vals: set[float] = set()
    for m in _NUM_RE.finditer(str(text)):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        vals.add(v)
        scale = _UNIT_SCALE.get((m.group(2) or "").lower())
        if scale:
            vals.add(v * scale)
    return vals


def numeric_match(pred: str, gold: str, rel_tol: float = 0.01):
    """True/False if the gold's numeric value appears in pred within rel_tol;
    None if the gold answer has no number (caller falls back to strict EM)."""
    g = _to_floats(gold)
    if not g:
        return None
    p = _to_floats(pred)
    if not p:
        return False
    for gv in g:
        for pv in p:
            if gv == 0:
                if pv == 0:
                    return True
            elif abs(pv - gv) / abs(gv) <= rel_tol:
                return True
    return False


def exact_match_numeric(pred: str, gold: str) -> int:
    """Lenient EM: strict string match OR numeric value match (falls back to strict
    EM when the gold answer is non-numeric)."""
    if exact_match(pred, gold):
        return 1
    nm = numeric_match(pred, gold)
    return 1 if nm is True else 0


def score_file(preds_path: Path):
    import pandas as pd

    rows = read_predictions(preds_path)
    per_row = []
    for r in rows:
        pred, gold = r["generated_answer"], r["gold_answer"]
        per_row.append({
            "financebench_id": r["financebench_id"],
            "question_type": r["question_type"],
            "abstained": is_abstention(pred),
            "exact_match": exact_match(pred, gold),
            "em_numeric": exact_match_numeric(pred, gold),
            "f1": token_f1(pred, gold),
        })
    df = pd.DataFrame(per_row)

    answered = df[~df["abstained"]]
    try:
        cfg = load_run_config(preds_path)
    except FileNotFoundError:
        cfg = {"pipeline": preds_path.stem, "model": "unknown"}

    summary = {
        "pipeline": cfg.get("pipeline"),
        "model": cfg.get("model"),
        "n_examples": len(df),
        "exact_match": round(df["exact_match"].mean(), 4),
        "f1": round(df["f1"].mean(), 4),
        "em_numeric": round(df["em_numeric"].mean(), 4),
        "n_answered": int(len(answered)),
        "f1_answered_only": round(answered["f1"].mean(), 4) if len(answered) else 0.0,
        "em_answered_only": round(answered["exact_match"].mean(), 4) if len(answered) else 0.0,
        "em_numeric_answered": round(answered["em_numeric"].mean(), 4) if len(answered) else 0.0,
    }

    # per-row scores for error analysis
    per_row_path = RESULTS_DIR / "raw_outputs" / f"{preds_path.stem}_qa_perrow.csv"
    per_row_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_row_path, index=False)

    # update the shared QA-metrics table (replace this pipeline+model's row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([summary])
    if QA_CSV.exists():
        prev = pd.read_csv(QA_CSV)
        mask = ~((prev["pipeline"] == summary["pipeline"]) & (prev["model"] == summary["model"]))
        summary_df = pd.concat([prev[mask], summary_df], ignore_index=True)
    summary_df.to_csv(QA_CSV, index=False)

    print("--- QA metrics ---")
    print(f"  exact match (all {summary['n_examples']}): strict {summary['exact_match']:.4f}"
          f"  |  numeric-tolerant {summary['em_numeric']:.4f}")
    print(f"  token F1    (all {summary['n_examples']}): {summary['f1']:.4f}")
    print(f"  answered {summary['n_answered']}/{summary['n_examples']}: "
          f"F1 {summary['f1_answered_only']:.4f}, strict-EM {summary['em_answered_only']:.4f}, "
          f"numeric-EM {summary['em_numeric_answered']:.4f}")
    print(f"saved summary -> {QA_CSV.relative_to(REPO_ROOT)}")
    print(f"saved per-row -> {per_row_path.relative_to(REPO_ROOT)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Token F1 + Exact Match (Task T09).")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    if not args.predictions.exists():
        raise SystemExit(f"predictions file not found: {args.predictions}")
    score_file(args.predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
