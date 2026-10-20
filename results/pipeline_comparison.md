# Pipeline comparison: retrieval strategies (T14/T15/T18)

Three RAG configurations on FinanceBench (all 150 questions), identical except for retrieval: **Baseline** = sparse BM25; **Dense** = FAISS over MiniLM embeddings; **Enhanced** = LLM query rewrite + dense. Generator, query-rewriter, and RAGAS judge are GPT-3.5-turbo at temperature 0. Retr@3 = share of questions whose top-3 retrieval reached the correct filing.

| Pipeline | Retr@3 | Faithful. | Faith(ans) | Ans. Rel. | Ctx. Prec. | Answered | F1 | EM (num) |
|---|---|---|---|---|---|---|---|---|
| Baseline (BM25) | 43% | 0.104 | 0.408 | 0.234 | 0.240 | 47/150 | 0.083 | 0.120 |
| Dense (FAISS) | 64% | 0.197 | 0.352 | 0.414 | 0.417 | 82/150 | 0.116 | 0.180 |
| Enhanced (rewrite) | 69% | 0.209 | 0.353 | 0.432 | 0.462 | 87/150 | 0.121 | 0.207 |
| Dense, single-doc | 100% | 0.300 | 0.572 | 0.424 | 0.435 | 96/150 | 0.139 | 0.267 |

## Interpretation

Holding the generator (GPT-3.5-turbo), prompt, chunks, and top-$k$ fixed, only \emph{retrieval} changes across the three pipelines, so the trend isolates the effect of retrieval quality. Retrieval accuracy climbs steadily: the correct filing reaches the top-3 for 43\% of questions under sparse BM25, 64\% under dense MiniLM retrieval, and 69\% once an LLM rewrites the query first. Because the generator is instructed to answer only from context, better retrieval directly lifts coverage: the model answers 47, 82, and 87 of 150 questions respectively, and token F1 rises from 0.083 to 0.116 to 0.121. RAGAS context precision moves the same way, confirming the gains come from putting the right chunk in front of the model rather than from changes in generation. Strict exact match stays near zero because gold answers are short numeric values formatted many ways ($1,577 vs.\ 1577.00 vs.\ 1.577 billion); a numeric-tolerant EM (right value within 1\%, ignoring format) recovers the real accuracy and also rises with retrieval (0.120 $\rightarrow$ 0.180 $\rightarrow$ 0.207), showing the strict metric understated correctness rather than the model being wrong. The headline for RQ1 is that retrieval, not the generator, is the dominant lever on this benchmark: dense retrieval and query rewriting each add coverage, yet even the strongest configuration answers well under half the set, leaving a substantial gap---and a pool of answered-but-unsupported cases---for the error analysis to characterize. As an upper bound, restricting retrieval to the question's own filing (\emph{Dense, single-doc}---the known-document setting) lifts Retr@3 to 100\%, coverage to 96/150, and context precision to 0.435; the full-corpus configurations must first find the right filing among 84, which is where most of the remaining loss sits.
