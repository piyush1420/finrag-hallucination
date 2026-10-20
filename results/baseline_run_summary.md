# Baseline run summary (T06)

Pipeline 1: full-corpus BM25 retrieval + GPT-3.5-turbo, answering from context only. This is the curated summary; the full per-question predictions live in `results/raw_outputs/baseline_bm25.jsonl` (gitignored, regenerate with `python src/pipeline_baseline.py`).

## Run config

| setting | value |
|---|---|
| pipeline | baseline_bm25 |
| model | openai:gpt-3.5-turbo |
| retrieval | bm25 (full corpus, all 84 filings) |
| chunk_strategy | fixed_512 |
| top_k | 3 |
| temperature | 0.0 |
| n_examples | 150 |

## What came out

- Questions answered: **150** (all of FinanceBench open-source).
- Retrieval reached the correct filing (top-3): **64/150 = 42.7%**.
- Model abstained ("I don't know"): **103/150 = 69%**.
- Model gave an answer: **47/150 = 31%**.
- Answered *and* the right filing was never retrieved: **20/150**. These are the prime hallucination suspects for the error analysis (T19-T23): the model committed to an answer with no supporting document in context.

## Abstention by question type

| question_type | total | abstained | abstain rate |
|---|---|---|---|
| metrics-generated | 50 | 44 | 88% |
| domain-relevant | 50 | 36 | 72% |
| novel-generated | 50 | 23 | 46% |

## Reading it

The naive full-corpus BM25 retriever surfaces the correct filing only about two times in five, and with the wrong context in front of it GPT-3.5 mostly declines to answer rather than fabricate. So the baseline trades coverage for safety: few outright fabrications, but it leaves most questions on the table (especially the metrics-generated ones, which demand a specific figure from a specific filing). That retrieval gap is what the dense and query-rewrite pipelines (T14, T15) are meant to close, and the 20 answered-but-unsupported cases are where we expect the clearest hallucinations for the error analysis.
