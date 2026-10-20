"""Plot the hallucination-type frequency figure (Task T22).

Reads the labeled cases in annotations/failure_cases_50.csv and makes a bar chart of
the four hallucination types (excluding 'other', which are auto-flagged cases that
turned out to be correct). Saves the figure + a small frequency table under results/.

Run:  python src/plot_hallucination_types.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "annotations" / "failure_cases_50.csv"
FIG = ROOT / "results" / "fig_hallucination_types.png"
TABLE = ROOT / "results" / "hallucination_type_freq.md"

LABELS = {
    "numerical": "Numerical\n(wrong figure/calc)",
    "entity": "Entity\n(wrong source/segment)",
    "unsupported": "Unsupported\n(fabricated)",
    "reasoning": "Reasoning\n(flawed logic)",
}


def main() -> int:
    df = pd.read_csv(CSV)
    true = df[df["hallucination_type"] != "other"]
    counts = true["hallucination_type"].value_counts()
    counts = counts.reindex([t for t in LABELS if t in counts.index])  # fixed order
    n = int(counts.sum())

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=[LABELS[t] for t in counts.index],
                palette="viridis", orient="h", ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.2, i, f"{v}  ({v/n:.0%})", va="center", fontsize=13, fontweight="bold")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("# cases"); ax.set_ylabel("")
    ax.set_title(f"Hallucination types on FinanceBench (n={n} answered-but-wrong cases)",
                 fontweight="bold", fontsize=15)
    ax.set_xticks([]);
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, bbox_inches="tight")

    # small markdown table for the paper/notes
    lines = [f"# Hallucination-type frequency (T22)\n",
             f"From {len(df)} auto-flagged cases, {len(df)-n} were actually correct "
             f"(metric artifacts); the remaining **{n}** are true hallucinations:\n",
             "| Type | Count | % |", "|---|---|---|"]
    for t in counts.index:
        lines.append(f"| {t} | {counts[t]} | {counts[t]/n:.0%} |")
    TABLE.write_text("\n".join(lines) + "\n")

    print("saved", FIG.name, "and", TABLE.name)
    print(counts.to_dict(), "| n =", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
