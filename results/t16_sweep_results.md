# T16 retrieval top-k ablation

Dense (FAISS + MiniLM) pipeline, GPT-3.5-turbo generator (temp 0), 512-token chunks, full 150-question FinanceBench set. Only retrieval depth (top-k) changes. RAGAS judge = GPT-4o-mini (same as the main results). Retr@k = share of questions whose top-k retrieval surfaced the correct filing.

| Top-k | Retr@k | Faithful. | Faith(ans) | Ans. Rel. | Ctx. Prec. | Answered | F1 | EM (num) |
|---|---|---|---|---|---|---|---|---|
| 3 | 64% | 0.197 | 0.352 | 0.414 | 0.417 | 82/150 | 0.116 | 0.180 |
| 5 | 73% | 0.248 | 0.397 | 0.468 | 0.388 | 97/150 | 0.126 | 0.220 |

![top-k ablation](t16_sweep_chart.png)
