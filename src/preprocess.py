"""Turn the source filing PDFs into metadata-tagged chunks for retrieval (Task T05).

Two chunking strategies, because we want to compare how chunk shape affects
retrieval (and later, hallucination):

  * ``fixed_512``: a sliding window of 512 tokens with a small overlap. Simple,
    ignores sentence boundaries, can cut a number off from its label.
  * ``sentence``: pack whole sentences together until we hit the token budget,
    never splitting mid-sentence. Keeps context intact but chunks vary in length.

Token counts use tiktoken's ``cl100k_base`` (the GPT-3.5 / GPT-4 encoding), so "512
tokens" here means the same thing the generator sees, not a rough word count.

Every chunk carries its source metadata (company, doc type, page range, chunk id)
so retrieval results are traceable back to a specific filing and page. Chunks are
serialized to parquet under ``data/processed/`` and loaded with ``load_chunks`` so
no run ever has to re-parse the PDFs.

Run it:
    python src/preprocess.py                 # both strategies, 512 tokens
    python src/preprocess.py --strategy sentence
    python src/preprocess.py --max-tokens 256 --overlap 32

Note: 10-K/10-Q PDFs are table-heavy, and PDF text extraction flattens tables into
messy linear text. We keep the raw extracted text as-is; cleaning it up is a known
limitation worth flagging in the error analysis.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import tiktoken
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PDF_DIR = REPO_ROOT / "data" / "pdfs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
QA_JSONL = RAW_DIR / "financebench_open_source.jsonl"

STRATEGIES = ("fixed_512", "sentence")

# One shared encoder for the whole run.
_ENC = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str        # "<doc_name>::<strategy>::<index>"
    doc_name: str
    company: str
    doc_type: str
    doc_period: str
    gics_sector: str
    strategy: str
    page_start: int      # 1-indexed page where the chunk begins
    page_end: int        # 1-indexed page where the chunk ends
    n_tokens: int
    n_chars: int
    text: str


# --- Metadata + PDF text ---------------------------------------------------

def build_doc_metadata(qa_jsonl: Path = QA_JSONL) -> dict[str, dict]:
    """Map each doc_name to its filing metadata, pulled from the QA records.

    Several questions share one filing, so we just take the first record we see
    for each doc_name (the doc-level fields are identical across its questions).
    """
    meta: dict[str, dict] = {}
    with open(qa_jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            name = r.get("doc_name")
            if name and name not in meta:
                meta[name] = {
                    "company": r.get("company", ""),
                    "doc_type": r.get("doc_type", ""),
                    "doc_period": str(r.get("doc_period", "")),
                    "gics_sector": r.get("gics_sector", ""),
                }
    return meta


def extract_pages(pdf_path: Path) -> list[str]:
    """Return the text of each page, in order. Index i is page i+1."""
    pages: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return pages


# --- Chunkers --------------------------------------------------------------

def _fixed_size_chunks(pages: list[str], max_tokens: int, overlap: int) -> list[tuple[str, int, int]]:
    """Sliding window over the whole document's tokens.

    We first flatten the doc into a list of (token_id, page_number) pairs so a
    window that straddles a page boundary still knows which pages it touches.
    Returns (text, page_start, page_end) tuples.
    """
    tokens: list[int] = []
    token_pages: list[int] = []
    for page_idx, text in enumerate(pages):
        page_no = page_idx + 1
        ids = _ENC.encode(text)
        tokens.extend(ids)
        token_pages.extend([page_no] * len(ids))

    if not tokens:
        return []

    step = max(1, max_tokens - overlap)
    out: list[tuple[str, int, int]] = []
    for start in range(0, len(tokens), step):
        window = tokens[start:start + max_tokens]
        if not window:
            break
        text = _ENC.decode(window).strip()
        if text:
            page_start = token_pages[start]
            page_end = token_pages[min(start + max_tokens, len(tokens)) - 1]
            out.append((text, page_start, page_end))
        if start + max_tokens >= len(tokens):
            break
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentence_chunks(pages: list[str], max_tokens: int) -> list[tuple[str, int, int]]:
    """Greedily pack whole sentences until adding the next would blow the budget.

    Sentence splitting is a plain regex on ``. ! ?`` followed by whitespace. It is
    not perfect on abbreviations or table text, but it never cuts a sentence in half,
    which is the point of this strategy.
    """
    # (sentence_text, token_count, page_no)
    sentences: list[tuple[str, int, int]] = []
    for page_idx, text in enumerate(pages):
        page_no = page_idx + 1
        for raw in _SENT_SPLIT.split(text):
            sent = raw.strip()
            if not sent:
                continue
            n = len(_ENC.encode(sent))
            sentences.append((sent, n, page_no))

    out: list[tuple[str, int, int]] = []
    cur: list[str] = []
    cur_tokens = 0
    cur_page_start = None
    cur_page_end = None

    def flush() -> None:
        nonlocal cur, cur_tokens, cur_page_start, cur_page_end
        if cur:
            out.append((" ".join(cur).strip(), cur_page_start, cur_page_end))
        cur, cur_tokens, cur_page_start, cur_page_end = [], 0, None, None

    for sent, n, page_no in sentences:
        # A single sentence longer than the budget becomes its own chunk.
        if n > max_tokens:
            flush()
            out.append((sent, page_no, page_no))
            continue
        if cur_tokens + n > max_tokens and cur:
            flush()
        if not cur:
            cur_page_start = page_no
        cur.append(sent)
        cur_tokens += n
        cur_page_end = page_no
    flush()
    return out


# --- Build + serialize -----------------------------------------------------

def chunk_document(doc_name: str, meta: dict, strategy: str,
                   max_tokens: int, overlap: int) -> list[Chunk]:
    pdf_path = PDF_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        return []
    pages = extract_pages(pdf_path)

    if strategy.startswith("fixed_"):
        pieces = _fixed_size_chunks(pages, max_tokens, overlap)
    elif strategy == "sentence":
        pieces = _sentence_chunks(pages, max_tokens)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    chunks: list[Chunk] = []
    for i, (text, p_start, p_end) in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{doc_name}::{strategy}::{i:04d}",
            doc_name=doc_name,
            company=meta.get("company", ""),
            doc_type=meta.get("doc_type", ""),
            doc_period=meta.get("doc_period", ""),
            gics_sector=meta.get("gics_sector", ""),
            strategy=strategy,
            page_start=p_start,
            page_end=p_end,
            n_tokens=len(_ENC.encode(text)),
            n_chars=len(text),
            text=text,
        ))
    return chunks


def build_chunks(strategy: str, max_tokens: int, overlap: int) -> pd.DataFrame:
    """Chunk every downloaded PDF with one strategy and return a DataFrame."""
    doc_meta = build_doc_metadata()
    doc_names = sorted(doc_meta.keys())

    all_chunks: list[Chunk] = []
    missing = 0
    for name in tqdm(doc_names, desc=f"chunking ({strategy})"):
        chunks = chunk_document(name, doc_meta[name], strategy, max_tokens, overlap)
        if not chunks:
            missing += 1
        all_chunks.extend(chunks)

    if missing:
        print(f"  note: {missing} doc(s) had no PDF or no extractable text")
    return pd.DataFrame(asdict(c) for c in all_chunks)


def processed_path(strategy: str) -> Path:
    return PROCESSED_DIR / f"chunks_{strategy}.parquet"


def load_chunks(strategy: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load pre-built chunks from disk. No PDF parsing happens here.

    This is the function every pipeline calls; it fails with a clear message if the
    chunks were never built.
    """
    path = processed_dir / f"chunks_{strategy}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no chunk file at {path}. Run: python src/preprocess.py --strategy {strategy}"
        )
    return pd.read_parquet(path)


# --- Main ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk FinanceBench PDFs (Task T05).")
    parser.add_argument("--strategy", choices=("fixed_512", "fixed_256", "sentence", "both"),
                        default="both", help="which chunking strategy to build")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="token budget per chunk")
    parser.add_argument("--overlap", type=int, default=64,
                        help="token overlap between fixed-size windows")
    args = parser.parse_args()

    if not QA_JSONL.exists():
        raise SystemExit("QA records not found. Run: python src/download_data.py")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    todo = STRATEGIES if args.strategy == "both" else (args.strategy,)

    for strategy in todo:
        df = build_chunks(strategy, args.max_tokens, args.overlap)
        out = processed_path(strategy)
        df.to_parquet(out, index=False)
        print(f"[{strategy}] {len(df)} chunks from {df['doc_name'].nunique()} docs "
              f"-> {out.relative_to(REPO_ROOT)}")
        print(f"          tokens/chunk: mean {df['n_tokens'].mean():.0f}, "
              f"median {df['n_tokens'].median():.0f}, max {df['n_tokens'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
