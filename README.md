# Analyzing Hallucination Patterns in RAG-Based Financial Question Answering

Course project for **CS6120 — Natural Language Processing**, Northeastern University.
Group of 4. We study how and when Retrieval-Augmented Generation (RAG) systems hallucinate
on financial question answering, using the **FinanceBench** dataset.

**Research questions**
1. How frequently do different RAG pipeline configurations hallucinate on financial QA?
2. What hallucination *types* dominate in finance — numerical, entity, reasoning, or unsupported extrapolation?
3. Do larger / instruction-tuned LLMs hallucinate less given the same retrieved context?

> 📋 The full task board, milestones, and deadlines live in **[PROJECT_PLAN.md](PROJECT_PLAN.md)** — read it before picking up work.

---

## Repository structure

```
finrag-hallucination/
├── README.md               # this file
├── PROJECT_PLAN.md         # task board + timeline — single source of truth
├── requirements.txt        # pinned Python dependencies
├── .env.example            # template for secrets (copy to .env)
├── .gitignore
├── data/
│   ├── raw/                # raw FinanceBench JSONL (gitignored)
│   ├── pdfs/               # source 10-K/10-Q/8-K PDFs (gitignored)
│   └── processed/          # chunked + metadata-tagged data (gitignored)
├── src/                    # pipeline + evaluation code (importable modules)
├── notebooks/              # EDA and experiment notebooks
├── results/                # curated tables & figures for the paper
│   └── raw_outputs/        # bulky raw model outputs (gitignored)
├── annotations/            # 50-case error-analysis sheets + taxonomy
└── paper/                  # preliminary & final paper sources
```

> Large/regenerable artifacts (PDFs, FAISS indices, model checkpoints, raw outputs) are
> **gitignored**. Commit only code, curated results, and small data descriptors.

---

## Setup

> 📌 **New here? Follow [SETUP.md](SETUP.md)** for the full walkthrough — fresh venv,
> your own OpenAI key (each member uses their own; never committed), and a one-command
> `python src/check_setup.py` verifier. The steps below are the short version.

### 1. Clone and create an environment
```bash
git clone <repo-url>
cd finrag-hallucination

# create + activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **Note (macOS):** `bitsandbytes` is skipped automatically on macOS — it's only needed for
> Mistral-7B 4-bit quantization, which we run on **Google Colab GPU**, not local machines.

### 3. Configure secrets
```bash
cp .env.example .env
# then edit .env and paste your real OPENAI_API_KEY
```
The `.env` file is gitignored — **never commit API keys.**

### 4. Verify the install
```bash
python -c "import langchain, faiss, sentence_transformers, openai, ragas, rank_bm25, fitz; print('OK')"
```
Optionally confirm your OpenAI key works with a tiny test call before running pipelines.

### 5. Register a Jupyter kernel (for notebooks)
```bash
python -m ipykernel install --user --name rag-finqa
```

---

## Contributing workflow

We work off feature branches and merge via reviewed pull requests.

1. Pick the next unblocked task in [PROJECT_PLAN.md](PROJECT_PLAN.md); set its **Status → IN PROGRESS** and add your name as **Owner**.
2. Branch: `git checkout -b feature/<task-id>-short-name` (e.g. `feature/T06-baseline-rag`).
3. Commit small, descriptive changes.
4. Open a PR; request review from **at least one** other member.
5. After merge, set the task **Status → DONE** and tick its acceptance criteria.

**Conventions**
- Keep the shared output schema (see Task T07) stable — pipelines and evaluation depend on it.
- Don't commit secrets, raw PDFs, indices, or large model files (already gitignored).
- Run experiments at `temperature = 0.0` for reproducible evaluation.

---

## Data

We use the open-source subset of **FinanceBench** (150 annotated examples) from HuggingFace
(`PatronusAI/financebench`), plus the source filing PDFs. **Download steps and a loader are
covered in tasks T03–T05** of the plan — see [PROJECT_PLAN.md](PROJECT_PLAN.md). Data is not
checked into git; each member downloads it locally into `data/`.

---

## Status

Project kicked off; setup phase in progress. Track live status in [PROJECT_PLAN.md](PROJECT_PLAN.md).
