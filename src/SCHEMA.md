# Output schema and run config (Task T07)

This is the frozen data contract for the project. Every pipeline writes its answers
in the same row shape, and the evaluation code reads that shape without caring which
pipeline produced it. The Python definitions live in [`schema.py`](schema.py); this
file is the human-readable version.

Don't change a field name without a PR, because the pipelines (T06, T14, T15, T17) and
the eval harness (T08, T09) all depend on it.

## Prediction row

One answered question is one JSON object (one line in a `.jsonl` file):

| Field | Type | Meaning |
|-------|------|---------|
| `financebench_id` | str | Unique FinanceBench id. Links the row back to the gold record. |
| `question` | str | The question text, verbatim. |
| `contexts` | list[str] | Retrieved chunk texts, best-first. RAGAS reads this directly. |
| `retrieved_chunk_ids` | list[str] | Chunk ids lined up 1:1 with `contexts`, for traceability. |
| `generated_answer` | str | The model's answer. |
| `gold_answer` | str | Human-annotated FinanceBench answer. Used for F1 / Exact Match. |
| `doc_name` | str | The source filing the question is about. |
| `question_type` | str | FinanceBench `question_type` label (e.g. `metrics-generated`). |

`contexts` and `retrieved_chunk_ids` must be the same length. `validate_row()` in
`schema.py` checks this, so a malformed run fails while it's being written instead of
later during eval.

### Why these fields

RAGAS needs four columns: `question`, `answer`, `contexts`, `ground_truth`. We store
them under our own names and map them at eval time. The mapping is kept in one place
(`RAGAS_FIELD_MAP` in `schema.py`):

| Our field | RAGAS column |
|-----------|--------------|
| `question` | `question` |
| `generated_answer` | `answer` |
| `contexts` | `contexts` |
| `gold_answer` | `ground_truth` |

`retrieved_chunk_ids`, `doc_name`, and `question_type` aren't needed by RAGAS. We keep
them so we can slice results by filing or question type in the error analysis (T19-T23)
and trace any answer back to the exact chunk it came from.

## Run config (sidecar file)

Every predictions file gets a sidecar `<name>.jsonl.config.json` describing the run, so
results are reproducible and the results table can report exactly what produced each row.

| Field | Type | Example |
|-------|------|---------|
| `pipeline` | str | `baseline_bm25` |
| `model` | str | `gpt-3.5-turbo` |
| `retrieval` | str | `bm25` |
| `chunk_strategy` | str | `fixed_512` |
| `top_k` | int | `3` |
| `temperature` | float | `0.0` |
| `embedding_model` | str or null | `null` for BM25; the MiniLM name for dense retrieval |
| `n_examples` | int | `150` |
| `notes` | str | free-text |

We keep `temperature = 0.0` for every evaluated run so results are deterministic
(see the Conventions note in the README).

## Files on disk

```
results/
├── raw_outputs/                          # bulky, gitignored
│   ├── baseline_bm25_gpt35.jsonl         # one prediction row per question
│   └── baseline_bm25_gpt35.jsonl.config.json
└── <curated tables/figures>              # committed
```

Raw prediction dumps go in `results/raw_outputs/` (gitignored, per the repo rules).
Small curated tables and figures, like the RAGAS and F1 numbers, are committed under
`results/`.

## Using it in code

```python
from schema import RunConfig, write_predictions, read_predictions

cfg = RunConfig(
    pipeline="baseline_bm25", model="gpt-3.5-turbo", retrieval="bm25",
    chunk_strategy="fixed_512", top_k=3, temperature=0.0,
)
write_predictions(rows, cfg, "results/raw_outputs/baseline_bm25_gpt35.jsonl")

rows = read_predictions("results/raw_outputs/baseline_bm25_gpt35.jsonl")
```
