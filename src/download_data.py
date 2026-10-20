"""Download the FinanceBench dataset and source filing PDFs.

Covers Task T03 in PROJECT_PLAN.md. Run once after cloning the repo:

    python src/download_data.py

What it fetches:
  1. The 150-example FinanceBench QA records from HuggingFace
     (`PatronusAI/financebench`, config "default", split "train")
     -> data/raw/financebench_open_source.jsonl
  2. Per-document metadata from the FinanceBench GitHub repo
     -> data/raw/financebench_document_information.jsonl
  3. One source filing PDF per unique `doc_name` from the GitHub repo
     -> data/pdfs/<doc_name>.pdf

Nothing here is committed to git (data/ is gitignored); each member runs this locally.
PDF downloads are resumable: files already present are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency. Run: pip install -r requirements.txt")

try:
    from tqdm import tqdm
except ImportError:  # tqdm is in requirements, but degrade gracefully
    def tqdm(iterable, **_kwargs):
        return iterable

# --- Constants -------------------------------------------------------------

HF_DATASET = "PatronusAI/financebench"
GH_RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
DOC_INFO_URL = f"{GH_RAW_BASE}/data/financebench_document_information.jsonl"
PDF_URL_TEMPLATE = GH_RAW_BASE + "/pdfs/{doc_name}.pdf"

# Resolve paths relative to the repo root (parent of this file's src/ dir).
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PDF_DIR = REPO_ROOT / "data" / "pdfs"

OPEN_SOURCE_JSONL = RAW_DIR / "financebench_open_source.jsonl"
DOC_INFO_JSONL = RAW_DIR / "financebench_document_information.jsonl"


# --- Helpers ---------------------------------------------------------------

def _http_get(url: str, dest: Path) -> None:
    """Download `url` to `dest`, writing atomically via a .part temp file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "financebench-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(64 * 1024):
            fh.write(chunk)
    tmp.replace(dest)


def download_dataset() -> list[dict]:
    """Pull the QA records from HuggingFace and save them as JSONL."""
    print(f"Loading {HF_DATASET} from HuggingFace ...")
    ds = load_dataset(HF_DATASET, "default", split="train")
    records = [dict(row) for row in ds]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OPEN_SOURCE_JSONL, "w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  saved {len(records)} records -> {OPEN_SOURCE_JSONL.relative_to(REPO_ROOT)}")
    return records


def download_doc_info() -> None:
    """Fetch the document-information JSONL from GitHub (best effort)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _http_get(DOC_INFO_URL, DOC_INFO_JSONL)
        print(f"  saved document info -> {DOC_INFO_JSONL.relative_to(REPO_ROOT)}")
    except urllib.error.HTTPError as exc:
        print(f"  WARNING: could not fetch document info ({exc}); continuing.")


def download_pdfs(records: list[dict]) -> tuple[int, int, list[str]]:
    """Download one PDF per unique doc_name. Returns (downloaded, skipped, failed)."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc_names = sorted({r["doc_name"] for r in records if r.get("doc_name")})
    print(f"Downloading {len(doc_names)} unique source PDFs ...")

    downloaded = skipped = 0
    failed: list[str] = []
    for doc_name in tqdm(doc_names, desc="PDFs"):
        dest = PDF_DIR / f"{doc_name}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            _http_get(PDF_URL_TEMPLATE.format(doc_name=doc_name), dest)
            downloaded += 1
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            failed.append(f"{doc_name} ({exc})")
    return downloaded, skipped, failed


# --- Main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Download FinanceBench data + PDFs.")
    parser.add_argument("--skip-dataset", action="store_true",
                        help="skip the HuggingFace QA records / doc-info download")
    parser.add_argument("--skip-pdfs", action="store_true",
                        help="skip the source PDF download")
    args = parser.parse_args()

    records: list[dict] = []
    if not args.skip_dataset:
        records = download_dataset()
        download_doc_info()
    elif not args.skip_pdfs:
        # Need records to know which PDFs to fetch.
        if not OPEN_SOURCE_JSONL.exists():
            sys.exit("Cannot --skip-dataset before it has been downloaded once.")
        with open(OPEN_SOURCE_JSONL, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh]

    print("\n--- Summary ---")
    print(f"QA records: {len(records)}")
    if not args.skip_pdfs:
        downloaded, skipped, failed = download_pdfs(records)
        print(f"PDFs downloaded: {downloaded} | already present: {skipped} | failed: {len(failed)}")
        if failed:
            print("Failed PDFs (re-run to retry):")
            for f in failed:
                print(f"  - {f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
