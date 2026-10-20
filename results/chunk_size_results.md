# Chunk-size ablation: 256 vs 512-token chunks (T16, RQ1)

Dense (FAISS + MiniLM) pipeline, GPT-3.5-turbo (temp 0), top-3 retrieval, full 150-question set. Only chunk size changes. RAGAS judge = GPT-4o-mini.

| Chunk (tok) | Retr@3 | Faithful. | Faith(ans) | Ans. Rel. | Ctx. Prec. | Answered | F1 | EM (num) |
|---|---|---|---|---|---|---|---|---|
| 256 | 59% | 0.234 | 0.462 | 0.324 | 0.288 | 70/150 | 0.095 | 0.160 |
| 512 | 64% | 0.197 | 0.352 | 0.414 | 0.417 | 82/150 | 0.116 | 0.180 |
