# Preliminary results (T10)

Baseline pipeline (BM25 + GPT-3.5-turbo) on FinanceBench. Both the token-level metrics (F1/EM, `src/qa_metrics.py`, T09; n=150) and the RAGAS metrics (`src/evaluate.py`, T08; n=150) cover all 150 questions. Generator and RAGAS judge are GPT-3.5-turbo at temperature 0; RAGAS embeddings are local sentence-transformers.

| Pipeline | Model | Faithfulness | Answer Rel. | Context Prec. | EM | F1 |
|---|---|---|---|---|---|---|
| baseline_bm25 | openai:gpt-3.5-turbo | 0.102 | 0.211 | 0.534 | 0.000 | 0.083 |

*RAGAS n = 150, F1/EM n = 150.*

## Interpretation

The baseline pairs a plain BM25 retriever with GPT-3.5-turbo. BM25 surfaces the correct filing for only 43\% of questions, so the generator---instructed to answer only from context---abstains on roughly two-thirds, answering 47 of 150. This caution is deliberate and safer than fabrication, but it caps coverage: token F1 is 0.083 over all questions and 0.261 on the answered subset, with exact match at zero because short numeric answers are written many ways ($1,577 vs.\ 1577.00) and rarely match after normalization. RAGAS reflects the same bottleneck: faithfulness (0.102) and answer relevancy (0.211) are low---an abstention has nothing to ground and does not address the question---while context precision (0.534) shows retrieval places a useful chunk near the top only about half the time. Read together, the picture is a retrieval-bound baseline: the generator rarely fabricates, but weak sparse retrieval leaves most questions unanswered. Closing that gap with dense and query-rewrite retrieval---and surfacing the answered-but-unsupported cases---is the focus of the next phase.
