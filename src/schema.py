"""Shared output schema + run configuration for every RAG pipeline (Task T07).

The whole project depends on one data contract: each pipeline (baseline BM25,
dense FAISS, enhanced query-rewrite, and later the Mistral generator) writes its
predictions in the *same* row shape, so the evaluation code in ``evaluate.py`` can
score any pipeline's output without special-casing it.

Keep this file stable. If a field genuinely has to change, change it in a PR so the
pipelines and the eval harness move together (see the Conventions note in the README).

Two things live here:
  * the prediction row schema (what one answered question looks like on disk), and
  * ``RunConfig`` (the knobs used for a run: model, chunk size, top-k, temperature...),
    which we save next to every output file so the results table is reproducible.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# --- Prediction row schema -------------------------------------------------

# Every row written by a pipeline must contain exactly these keys. The order is
# just for readability in the JSONL; loaders should not rely on it.
REQUIRED_FIELDS: tuple[str, ...] = (
    "financebench_id",       # unique FinanceBench id; links a row back to the gold record
    "question",              # the question text, verbatim
    "contexts",              # list[str]: retrieved chunk texts, ranked best-first (RAGAS reads this)
    "retrieved_chunk_ids",   # list[str]: chunk ids lined up with `contexts`, for traceability
    "generated_answer",      # the model's answer text
    "gold_answer",           # human-annotated FinanceBench answer (used for F1 / Exact Match)
    "doc_name",              # the source filing the question is about
    "question_type",         # FinanceBench question_type label
)

# How our field names map onto the column names RAGAS expects. evaluate.py uses this
# so we only have to remember the mapping in one place.
RAGAS_FIELD_MAP: dict[str, str] = {
    "question": "question",
    "generated_answer": "answer",
    "contexts": "contexts",
    "gold_answer": "ground_truth",
}


@dataclass
class RunConfig:
    """Everything needed to reproduce a run, saved alongside its predictions."""

    pipeline: str            # "baseline_bm25" | "dense_faiss" | "enhanced_rewrite"
    model: str               # generator, e.g. "gpt-3.5-turbo"
    retrieval: str           # "bm25" | "dense"
    chunk_strategy: str      # "fixed_512" | "sentence"
    top_k: int               # number of chunks retrieved per question
    temperature: float       # 0.0 for deterministic eval
    embedding_model: str | None = None   # only for dense retrieval
    n_examples: int | None = None        # how many questions the run covered
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Validation ------------------------------------------------------------

def validate_row(row: dict[str, Any]) -> None:
    """Raise ValueError if `row` is missing fields or has the wrong types.

    Cheap to call, so pipelines can validate each row as they build it and fail
    loudly instead of writing a malformed file that only breaks later in eval.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        raise ValueError(f"row missing required field(s): {missing}")
    if not isinstance(row["contexts"], list) or not all(isinstance(c, str) for c in row["contexts"]):
        raise ValueError("`contexts` must be a list of strings")
    if not isinstance(row["retrieved_chunk_ids"], list):
        raise ValueError("`retrieved_chunk_ids` must be a list")
    if len(row["retrieved_chunk_ids"]) != len(row["contexts"]):
        raise ValueError("`retrieved_chunk_ids` and `contexts` must be the same length")


# --- Read / write ----------------------------------------------------------

def write_predictions(rows: Iterable[dict[str, Any]], run_config: RunConfig, path: str | Path) -> Path:
    """Write prediction rows to `path` (JSONL) and the run config to a sidecar.

    The sidecar is ``<path>.config.json`` so every output file is self-describing:
    you can always recover which model/chunking/top-k produced it.
    Returns the JSONL path actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    for r in rows:
        validate_row(r)

    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = dict(run_config.to_dict())
    cfg["n_examples"] = len(rows)
    sidecar = path.with_suffix(path.suffix + ".config.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    return path


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load prediction rows from a JSONL file written by `write_predictions`."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load the sidecar run config for a predictions file."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".config.json")
    with open(sidecar, encoding="utf-8") as fh:
        return json.load(fh)
