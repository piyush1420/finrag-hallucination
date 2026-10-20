# Data

This project uses the **open-source subset of FinanceBench** (150 annotated financial QA
examples) plus the source SEC filing PDFs they reference.

> **Nothing in this folder is committed to git** (see root `.gitignore`). Each member
> downloads the data locally by running the download script.

## How to download

From the repo root, with your environment set up (`pip install -r requirements.txt`):

```bash
python src/download_data.py
```

Flags:
- `--skip-pdfs` — fetch only the QA records / metadata (fast; no ~360 MB PDF download).
- `--skip-dataset` — fetch only the PDFs (requires the dataset to have been downloaded once).

PDF downloads are **resumable** — re-running skips files already present and retries failures.

## Sources

| Item | Source | Lands in |
|------|--------|----------|
| QA records (150 examples) | HuggingFace `PatronusAI/financebench` (config `default`, split `train`) | `data/raw/financebench_open_source.jsonl` |
| Per-document metadata | GitHub `patronus-ai/financebench` → `data/financebench_document_information.jsonl` | `data/raw/financebench_document_information.jsonl` |
| Source filing PDFs | GitHub `patronus-ai/financebench` → `pdfs/<doc_name>.pdf` | `data/pdfs/<doc_name>.pdf` |

## Expected layout after download

```
data/
├── raw/
│   ├── financebench_open_source.jsonl          # 150 QA records
│   └── financebench_document_information.jsonl  # document metadata
├── pdfs/
│   └── <doc_name>.pdf                           # one per unique doc_name
└── processed/                                   # produced later by chunking (Task T05)
```

## Record fields (QA JSONL)

Key fields used downstream: `financebench_id`, `company`, `doc_name`, `doc_type`,
`doc_period`, `question_type`, `question`, `answer`, `justification`, `evidence`,
`gics_sector`, `doc_link`.

- `doc_name` links a QA record to its PDF (filename without the `.pdf` extension).
- `evidence` holds the gold supporting passage(s) — the basis for judging groundedness.
- `answer` is the human-annotated gold answer used for F1 / Exact Match.

## Verifying your download

After running the script, confirm:
- `financebench_open_source.jsonl` has **150** lines.
- `data/pdfs/` contains one PDF per unique `doc_name` (the script prints the count and any failures).

If some PDFs fail (transient network/GitHub issues), just re-run the script — it only
re-fetches the missing ones.
