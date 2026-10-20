# Hallucination Taxonomy & Annotation Guidelines (T20)

For labeling the 50 answered-but-incorrect cases in `failure_cases_50.csv`. Goal:
assign each case **one primary hallucination type** so we can report a type-frequency
distribution (Figure 1) and case studies (Table 2) for the final paper (RQ2).

> **Scope.** Every case here is one where the model **committed to an answer** (did not
> abstain) but the answer is **wrong or unsupported**. Abstentions ("I don't know") are
> *not* in this set — a refusal is not a hallucination.

---

## The four categories

Label the **dominant** error. If two apply, pick the one that best explains *why the
answer is wrong*, and record the other in `secondary_type`.

### 1. `numerical`
Wrong number, wrong unit, wrong period, or a miscalculation — the entities and intent
are right but the **value** is wrong.
- *Examples:* reports \$1.2B when the filing says \$2.1B; gives FY2021 when asked FY2022;
  right line item but adds/divides incorrectly; wrong unit (thousands vs millions);
  transcribes a digit wrong from a table.
- *Tell:* the answer is "about the right thing" but the figure doesn't match gold.

### 2. `entity`
Wrong **entity** — company, subsidiary, executive, product, segment, or statement.
- *Examples:* answers about the wrong company (retrieval pulled another filing); cites the
  income statement figure when asked about the balance sheet; attributes a number to the
  wrong segment or year-label.
- *Tell:* the value may even be copied correctly, but it's the *wrong source entity*.
  (Common when `retrieval_hit = False` — the model answered from the wrong filing.)

### 3. `reasoning`
The premises/numbers in context are right, but the **logic** connecting them to the
answer is flawed.
- *Examples:* correct revenue and cost retrieved, but margin computed with the wrong
  formula; a valid comparison drawn in the wrong direction ("increased" vs "decreased");
  multi-step chain where one step is invalid.
- *Tell:* inputs are grounded, the *inference* is not.

### 4. `unsupported` (unsupported extrapolation)
A claim **not traceable to any retrieved passage** — the model invented it or pulled it
from parametric memory.
- *Examples:* states a figure that appears nowhere in the context; adds a confident
  qualitative claim ("driven by strong demand in Asia") with no support; fabricates a
  line item that isn't in the retrieved chunks.
- *Tell:* you cannot point to the number/claim anywhere in `retrieved_context`.

**`other`** — use sparingly for cases none of the above fit (e.g., irrelevant answer,
format-only mismatch that slipped the numeric-tolerant filter); explain in `notes`.

---

## How to label each row (workflow)
For each assigned row in `failure_cases_50.csv`:
1. Read `question`, then `gold_answer`.
2. Read `generated_answer` and scan `retrieved_context` (the 3 chunks the model saw;
   truncated — full text in `results/raw_outputs/enhanced_rewrite.jsonl` if needed).
3. Decide **why it's wrong** and put the category slug in `hallucination_type`
   (`numerical` / `entity` / `reasoning` / `unsupported` / `other`).
4. Optionally set `secondary_type`; add a one-line `notes` (what the right answer was /
   where the model went wrong). Helpful auto-signals already in the sheet: `retrieval_hit`
   (was the correct filing even retrieved?), `f1`, `faithfulness`.

### Decision hints
- `retrieval_hit = False` + confident number → usually **entity** (wrong filing) or
  **unsupported** (number not in the wrong context either).
- Right filing retrieved, number wrong → **numerical**.
- Right numbers present, conclusion wrong → **reasoning**.
- Claim/number absent from context → **unsupported**.

---

## Annotation sheet columns (`failure_cases_50.csv`)
| Column | Meaning |
|---|---|
| `financebench_id`, `pipeline`, `company`, `doc_name`, `question_type` | provenance |
| `question`, `retrieved_context`, `generated_answer`, `gold_answer` | what to judge |
| `retrieval_hit`, `f1`, `em_numeric`, `faithfulness` | auto signals (do not edit) |
| **`hallucination_type`** | **fill in**: primary category slug |
| `secondary_type`, `notes` | optional secondary label + rationale |

---

## Reliability note (single annotator)
All 50 cases are labeled by one annotator, so we do **not** report Cohen's kappa
(inter-annotator agreement needs ≥2 independent annotators). We note single-annotator
labeling as a limitation in the paper. *(Optional: a second independent pass on ~10 cases
— by another person or an LLM, disclosed as such — would let us report an agreement
number; skip unless we decide to add it.)*
