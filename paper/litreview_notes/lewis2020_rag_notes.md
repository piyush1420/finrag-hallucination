# RAG — Paper Summary & Study Notes

**Prepared by:** Bhalchandra Shinde · **Task:** T11 (literature review) · **Theme:** RAG methods & retrieval

**Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. (Facebook AI Research / UCL / NYU)
**Published:** 2020 (arXiv:2005.11401), updated April 2021 · BibTeX key: `lewis2020rag`

---

## 1. The core problem RAG solves
Pre-trained LMs store knowledge in their weights, which creates three problems:
1. **Stale knowledge** — can't easily update as the world changes.
2. **No provenance** — can't show *where* an answer came from.
3. **Hallucinations** — confidently generate plausible but incorrect facts.

RAG's solution: give the LM a **live, searchable document index** at inference time, so it looks things up rather than relying solely on memorized knowledge.

## 2. Architecture — two components
```
Query (x) → [Retriever: DPR] → top-K docs from Wikipedia → [Generator: BART] → answer (y)
```
- **Retriever (DPR):** two BERT encoders (query + document); Maximum Inner Product Search (MIPS) over dense vectors. Wikipedia split into 21M 100-word chunks, each pre-encoded. Index is **fixed** during training (only the query encoder updates) → saves compute.
- **Generator (BART-large, 400M):** takes query + retrieved docs (concatenated), generates free-form text.
- **Parametric memory** = BART weights (what it memorized); **non-parametric memory** = the document index (external, editable).
- Trained **jointly end-to-end**; retrieved doc is a **latent variable** (no supervision on which doc to retrieve).

## 3. Two variants
- **RAG-Sequence:** same document for the whole answer; best for a single coherent source (most QA).
- **RAG-Token:** a different document per output token; best for synthesis across sources (e.g. Jeopardy generation).

## 4. Key insight — parametric + non-parametric working together
Generating a Jeopardy clue about Hemingway, RAG-Token attends to Doc 1 ("A Farewell to Arms") then Doc 2 ("The Sun Also Rises"); after the first word of each title the document posterior flattens and **BART's parametric memory completes the title**. Division of labor: retriever surfaces *topics*, generator fills in *details*.

## 5. Experiments & results
- **Open-domain QA (NQ Exact Match):** T5-11B closed-book 34.5 · REALM 40.4 · DPR 41.5 · **RAG-Sequence 44.5** — SOTA with ~626M params vs T5-11B's 11B (~17× fewer).
- RAG answers correctly even when the answer is **not verbatim** in any retrieved doc (**11.8% of NQ**) — an extractive model scores 0% there.
- **Abstractive QA (MS-MARCO):** +2.6 BLEU over BART, more factual.
- **Jeopardy generation:** humans judged RAG more factual (42.7% vs 7.1%) and more specific (37.4% vs 16.8%); higher n-gram diversity.
- **FEVER fact verification:** within 4.3% of complex SOTA pipelines without their extra supervision.

## 6. Key ablations
- **Learned retrieval beats frozen retrieval** on all tasks.
- **DPR (dense) beats BM25** on most tasks; exception FEVER (entity-heavy → keyword matching suits BM25).
- **More docs help** up to a point (RAG-Token plateaus ~10 docs).
- **Index hot-swapping:** replacing the 2016 with the 2018 Wikipedia index → matched ~69% vs mismatched ~8% accuracy on world-leader questions. **Update knowledge by swapping the index — no retraining.**

## 7. Broader significance
| Memory | Where | Characteristics |
|---|---|---|
| Parametric | BART weights | implicit, fast, hard to update, can hallucinate |
| Non-parametric | document index | explicit, inspectable, easily updated, grounded |

Non-parametric memory is raw text → human-readable and human-writable.
**Limitations noted:** retrieval can "collapse" (always fetch same docs, generator ignores them); some tasks unanswerable from Wikipedia.

## 8. Connection to our work (FinanceBench RAG)
FinanceBench's "vector store" configs are essentially RAG applied to SEC filings (Wikipedia → filings). RAG's **non-verbatim** capability is central to our setting: our EDA shows **98% of FinanceBench gold answers never appear verbatim** in the evidence, so the generator must *synthesize* not copy. RAG's ablations also justify our design axis (BM25 baseline → dense/DPR-style retrieval) and its index-swap result reinforces our core finding that **retrieval quality, not generator memory, is the bottleneck.**

## Quick reference
Retriever DPR (bi-encoder BERT) · Generator BART-large (400M) · Knowledge = Wikipedia (21M×100-word chunks) · Index = FAISS + MIPS · ~626M trainable params · signal = (question, answer) only · variants RAG-Sequence / RAG-Token · key advantage = swap index, no retraining · code in HuggingFace Transformers.
