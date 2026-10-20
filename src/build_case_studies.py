"""Curate the qualitative case-study table for the paper (Task T23 -> Table 2).

Picks two clear, illustrative examples per hallucination type from the labeled cases
in annotations/failure_cases_50.csv and formats them as a Markdown table. The chosen
cases are listed explicitly (by financebench_id) so the curation is transparent and
reproducible; everything else (question, answers, rationale) is read from the sheet.

Run:  python src/build_case_studies.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "annotations" / "failure_cases_50.csv"
OUT = ROOT / "results" / "case_studies.md"

# Curated examples per type (2 each), chosen for clarity. Order = paper order.
CURATED = [
    ("numerical",   "financebench_id_04854"),  # FCF off by an order of magnitude
    ("numerical",   "financebench_id_06272"),  # payout ratio: wrong dividends figure
    ("entity",      "financebench_id_01198"),  # AMD answered with Lockheed F-16/F-22 text
    ("entity",      "financebench_id_01028"),  # listed office cities, not operating regions
    ("unsupported", "financebench_id_00476"),  # fabricated a security; gold = "none"
    ("unsupported", "financebench_id_02419"),  # "not spinning off"; gold = spinning off Upjohn
    ("reasoning",   "financebench_id_00566"),  # used wrong fiscal period -> opposite conclusion
    ("reasoning",   "financebench_id_00499"),  # capital-intensive? opposite interpretation
]

TYPE_LABEL = {"numerical": "Numerical", "entity": "Entity",
              "unsupported": "Unsupported", "reasoning": "Reasoning"}


def _clip(x, n) -> str:
    s = " ".join(str(x).split())          # collapse whitespace/newlines
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _why(note: str) -> str:
    return _clip(str(note).replace("[DRAFT-LLM]", "").strip(" -"), 90)


def main() -> int:
    df = pd.read_csv(CSV).set_index("financebench_id")

    headers = ["Type", "Company", "Question", "Model answer", "Gold", "Why it is wrong"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for htype, fid in CURATED:
        r = df.loc[fid]
        lines.append("| " + " | ".join([
            TYPE_LABEL[htype], _clip(r["company"], 18), _clip(r["question"], 85),
            _clip(r["generated_answer"], 55), _clip(r["gold_answer"], 45), _why(r["notes"]),
        ]) + " |")
    table = "\n".join(lines)

    md = ["# Qualitative case studies (T23, paper Table 2)\n",
          "Two representative cases per hallucination type, drawn from the 50 labeled "
          "answered-but-incorrect cases (`annotations/failure_cases_50.csv`). Each row shows "
          "what the model answered, the gold answer, and why the error falls in that type.\n",
          table, ""]
    OUT.write_text("\n".join(md), encoding="utf-8")

    print(table)
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
