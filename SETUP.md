# Setup Guide

How each member gets a working environment + their own OpenAI key. Do this once after
cloning. Covers Task **T02** in [PROJECT_PLAN.md](PROJECT_PLAN.md).

---

## 1. Python environment (use a fresh, dedicated venv)

**Use a new virtual environment for this project — do not reuse an existing one.** A fresh
env is the only way to confirm `requirements.txt` is complete, and it keeps all four of us on
an identical, conflict-free stack.

```bash
cd finrag-hallucination

python -m venv .venv             # Python 3.10+ recommended
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS note:** `bitsandbytes` is skipped automatically (it's `platform_system != "Darwin"`
> in `requirements.txt`). That's expected — it's only needed for Mistral-7B 4-bit quantization,
> which we run on **Colab GPU**, not local machines. Nothing else should fail.

Register a Jupyter kernel so notebooks use this env:

```bash
python -m ipykernel install --user --name rag-finqa
```

---

## 2. OpenAI API key — each member uses their OWN key

We do **not** share one key. Each member supplies their own, and the file holding it is
**never committed** (`.env` is gitignored). This keeps billing per-person and avoids leaking a
key in git history.

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste **your own** key:
   ```
   OPENAI_API_KEY=sk-...your-real-key...
   ```
3. Get a key (if you don't have one) at <https://platform.openai.com/api-keys>, and make sure
   your account has **billing/credits** enabled — a valid-looking but uncredited key will fail
   silently inside the pipeline.

**Rules:**
- ❌ Never commit `.env`, never paste a real key into code, a notebook, or chat.
- ✅ Only `.env.example` (placeholders, no secrets) is committed.
- If a key is ever exposed, **revoke it immediately** in the OpenAI dashboard and issue a new one.

> 💡 To keep cost predictable: run experiments at `temperature = 0.0`, and avoid re-running the
> full 150-example pipelines unnecessarily (see the API-budget note in the plan).

---

## 3. Verify your setup

Run the checker — it confirms the key libraries import, your `.env` loads, and (if a key is
present) makes one tiny OpenAI call to prove the key + billing work:

```bash
python src/check_setup.py
```

Expected: a line-by-line ✅ for imports, `.env`, and the API ping. If the API check fails,
fix your key/billing before picking up any pipeline task (T06+).

Quick import-only check (no API call):

```bash
python src/check_setup.py --no-api
```

---

## 4. Colab (for later GPU tasks T17 / T24)

Mistral-7B and the optional RoBERTa classifier need a GPU. Use the committed template at
[`notebooks/colab_gpu_template.ipynb`](notebooks/colab_gpu_template.ipynb): open it in Google
Colab, set **Runtime → Change runtime type → GPU**, and run the first cells to clone the repo
and confirm the GPU is visible.
