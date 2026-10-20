# Generator comparison on identical retrieval (RQ3, T17)

Both generators answer over the **same** retrieved context (Enhanced pipeline: query-rewrite + dense, 512-token chunks, top-3; Retr@3 = 69% for both). Only the answer generator differs. RAGAS judge = GPT-4o-mini; embeddings = MiniLM. Mistral-7B-Instruct runs locally via Ollama.

| Generator | Faithful. | Faith(ans) | Ans. Rel. | Ctx. Prec. | Answered | F1 | EM (num) |
|---|---|---|---|---|---|---|---|
| GPT-3.5-turbo | 0.209 | 0.353 | 0.432 | 0.462 | 87/150 | 0.121 | 0.207 |
| Mistral-7B-Instruct | 0.592 | 0.631 | 0.254 | 0.411 | 69/150 | 0.128 | 0.373 |
