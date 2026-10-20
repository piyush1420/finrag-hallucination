"""Verify a member's local setup (Task T02).

Checks, in order:
  1. The key Python libraries import cleanly (so requirements.txt is satisfied).
  2. A .env file exists and OPENAI_API_KEY is set.
  3. (unless --no-api) one tiny OpenAI call succeeds — proving the key + billing work.

Usage:
    python src/check_setup.py            # full check, incl. a ~1-token API call
    python src/check_setup.py --no-api   # imports + .env only, no API call (no cost)

Nothing here reads or prints your key. The API call costs a fraction of a cent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

# Libraries that must import for the pipelines/eval to run. (Import name, pip name.)
REQUIRED_IMPORTS = [
    ("langchain", "langchain"),
    ("openai", "openai"),
    ("faiss", "faiss-cpu"),
    ("rank_bm25", "rank_bm25"),
    ("sentence_transformers", "sentence-transformers"),
    ("ragas", "ragas"),
    ("datasets", "datasets"),
    ("pandas", "pandas"),
    ("fitz", "PyMuPDF"),
    ("dotenv", "python-dotenv"),
]

OK = "✅"
FAIL = "❌"


def check_imports() -> bool:
    print("1. Checking required libraries ...")
    missing: list[str] = []
    for module, pip_name in REQUIRED_IMPORTS:
        try:
            __import__(module)
            print(f"   {OK} {module}")
        except Exception as exc:  # noqa: BLE001 - report any import error
            print(f"   {FAIL} {module}  (pip install {pip_name})  -> {exc}")
            missing.append(pip_name)
    if missing:
        print(f"\n   Missing/broken: {', '.join(missing)}")
        print("   Fix: activate your venv and run  pip install -r requirements.txt")
        return False
    return True


def check_env() -> str | None:
    """Return the API key if present, else None. Never prints the key."""
    print("\n2. Checking .env / OPENAI_API_KEY ...")
    from dotenv import load_dotenv  # imported here so import-check runs first
    import os

    if not ENV_FILE.exists():
        print(f"   {FAIL} no .env found at {ENV_FILE}")
        print("   Fix: cp .env.example .env   then paste your own OPENAI_API_KEY")
        return None

    load_dotenv(ENV_FILE)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-...") or key == "sk-":
        print(f"   {FAIL} OPENAI_API_KEY is empty or still the placeholder")
        print("   Fix: edit .env and paste your real key")
        return None

    print(f"   {OK} .env loaded; OPENAI_API_KEY is set (length {len(key)}, not shown)")
    return key


def check_api(key: str) -> bool:
    print("\n3. Testing OpenAI API (one tiny call) ...")
    import os
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
            temperature=0.0,
        )
        reply = (resp.choices[0].message.content or "").strip()
        print(f"   {OK} API responded with model '{model}': {reply!r}")
        return True
    except Exception as exc:  # noqa: BLE001 - surface the real reason
        print(f"   {FAIL} API call failed: {exc}")
        print("   Common causes: invalid key, no billing/credits, or no network.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local setup for T02.")
    parser.add_argument("--no-api", action="store_true",
                        help="skip the OpenAI API call (imports + .env only)")
    args = parser.parse_args()

    imports_ok = check_imports()
    key = check_env()

    api_ok = True
    if args.no_api:
        print("\n3. Skipping API call (--no-api).")
    elif key:
        api_ok = check_api(key)
    else:
        api_ok = False
        print("\n3. Skipping API call (no usable key found above).")

    print("\n--- Summary ---")
    print(f"Imports: {OK if imports_ok else FAIL}   "
          f".env/key: {OK if key else FAIL}   "
          f"API: {OK if (args.no_api or api_ok) else FAIL}")

    all_ok = imports_ok and (key is not None) and (args.no_api or api_ok)
    if all_ok:
        print(f"\n{OK} Setup looks good. You can pick up pipeline tasks (T03+).")
        return 0
    print(f"\n{FAIL} Setup incomplete — fix the items marked above, then re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
