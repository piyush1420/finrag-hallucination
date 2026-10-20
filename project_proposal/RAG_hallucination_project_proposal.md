# Project Proposal: Analyzing Hallucination Patterns in RAG-Based Financial Question Answering

**Course:** NLP Final Project
**Group Size:** 4
**Target Workshop:** Towards Knowledgeable Foundation Models (or NLP for Positive Impact)

---

## Proposal Narrative

### Task, Goal, and Research Questions

This project investigates how and when Retrieval-Augmented Generation (RAG) systems produce
hallucinated outputs in the context of financial question answering (QA). While RAG has emerged
as a leading approach for grounding large language model (LLM) responses in verified external
knowledge, recent work has shown that even with accurate retrieved context, LLMs still generate
responses that contradict or go beyond the retrieved documents. This failure mode — known as RAG
hallucination — is particularly consequential in high-stakes domains like finance, where a single
incorrect numerical figure or misattributed claim can have serious real-world consequences.

Our central research questions are: (1) How frequently do different RAG pipeline configurations
produce hallucinated outputs on financial QA tasks? (2) What types of hallucinations are most
common in the financial domain — numerical errors, entity misattribution, or unsupported
reasoning? (3) Do larger or instruction-tuned LLMs hallucinate less than smaller models when
given the same retrieved context? We argue that answering these questions produces concrete,
actionable insights for practitioners building financial AI systems, and contributes original
empirical findings to a rapidly growing research area that still lacks thorough domain-specific
analysis.

---

### Data

Our primary dataset is **FinanceBench**, a publicly available benchmark specifically designed for
open-book financial QA. FinanceBench comprises questions about publicly traded companies drawn
from real financial filings including 10-K, 10-Q, and 8-K documents, with human-annotated gold
answers and evidence strings. The open-source subset contains 150 annotated examples freely
available on HuggingFace and GitHub, making it immediately accessible without any data collection
effort on our part.

Each example in FinanceBench contains a natural language question, a ground-truth answer, and
a pointer to the relevant evidence passage in the source document. This structure is ideal for
our purposes because it allows us to evaluate not just answer correctness but also whether a
model's response is grounded in the retrieved evidence — the key criterion for distinguishing
hallucination from accurate generation. As a secondary dataset for broader comparison, we will
use **RAGBench**, a large-scale multi-domain RAG evaluation dataset that includes a finance
subset alongside customer support and technical documentation domains, allowing us to examine
whether hallucination patterns in finance differ from other domains.

We will not create any new datasets. Both FinanceBench and RAGBench are existing, peer-reviewed
resources with established evaluation protocols, and their use is well-justified given the
research questions above.

---

### Tools and Infrastructure

For retrieval, we will use **LangChain** to build and compare RAG pipeline configurations,
combined with **FAISS** for dense vector indexing and **BM25** (via the `rank_bm25` library)
for sparse retrieval. Document chunking and preprocessing will be handled using **LlamaIndex**
utilities. All pipeline code will be written in Python.

For evaluation, we will use the **RAGAS framework**, which provides reference-free metrics
specifically designed for RAG systems, including faithfulness (whether the answer is grounded
in the retrieved context), answer relevancy, and context precision. We will also compute
standard exact match (EM) and F1 scores against the gold answers in FinanceBench for
compatibility with prior work. For hallucination classification, we will use a combination of
automated scoring via RAGAS and manual annotation of a random sample of failure cases (n=50)
to categorize error types.

Compute will be managed carefully within the $50 Google Cloud credit budget. Small-scale
experiments during development will run locally or on free-tier Colab. GPU-intensive runs
(e.g., inference with larger open models) will be reserved for final experiments.

---

### Models

We will evaluate three RAG configurations across two model families, giving us a 3×2 experimental
grid:

**RAG Configurations:**
- **Baseline RAG:** BM25 sparse retrieval + GPT-3.5-turbo generator (no reranking, fixed chunk size of 512 tokens)
- **Dense RAG:** FAISS with `sentence-transformers/all-MiniLM-L6-v2` embeddings + GPT-3.5-turbo generator
- **Enhanced RAG:** Dense retrieval + query rewriting (using a prompted LLM to expand the query before retrieval) + GPT-3.5-turbo generator

**Generator Models (applied to the best-performing retrieval config):**
- GPT-3.5-turbo via OpenAI API (strong commercial baseline)
- Mistral-7B-Instruct via HuggingFace (open-source, runnable on Colab with quantization)

The BERT-based models in our group's toolkit will be used for the hallucination classification
step — specifically, we will fine-tune a `roberta-base` classifier on a small labeled sample
of RAG outputs to automatically categorize hallucination types, building on the RAGTruth
dataset as a supervision source.

All three RAG configurations will be implemented by the group using LangChain; no off-the-shelf
end-to-end RAG systems will be used as black boxes. The key hyperparameters we will tune are
chunk size (256 vs. 512 tokens), number of retrieved passages (top-3 vs. top-5), and
temperature (0.0 for deterministic outputs during evaluation).

---

### Evaluation

Our evaluation has two components. First, **pipeline-level evaluation**: for each RAG
configuration and model combination, we report faithfulness, answer relevancy, and F1 against
gold answers using RAGAS and standard metrics. This lets us rank configurations and identify
which retrieval strategy best reduces hallucination. Second, **error analysis**: for the
baseline configuration (most likely to hallucinate), we manually inspect 50 failure cases and
categorize them into a taxonomy of hallucination types we expect to find in financial text:
(a) numerical hallucinations (wrong figures, wrong units, wrong year), (b) entity hallucinations
(wrong company name, wrong executive, wrong product), (c) reasoning hallucinations (logically
flawed conclusions from correct premises), and (d) unsupported extrapolations (claims not
traceable to any retrieved passage). This qualitative analysis is the most novel contribution
of the paper and directly addresses the gap identified in prior work.

---

### Related Work

We plan to include the following 8 papers in our proposal's related work, expanding to 20–24
by the preliminary report:

1. **Lewis et al. (2020), "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS)** — the foundational RAG paper; provides the architectural baseline our work builds on.
2. **Islam et al. (2023), "FinanceBench" (arXiv)** — introduces our primary dataset; directly relevant as the benchmark we evaluate on.
3. **Niu et al. (2024), "RAGTruth: A Hallucination Corpus for RAG" (ACL 2024)** — provides a hallucination taxonomy and labeled corpus we use to train our classifier.
4. **Song et al. (2024), "RAG-HAT: A Hallucination-Aware Tuning Pipeline" (EMNLP 2024)** — closest prior work on detecting and correcting RAG hallucinations; we compare our findings against their error analysis.
5. **Friel et al. (2024), "RAGBench: Explainable Benchmark for RAG Systems" (arXiv)** — our secondary dataset; relevant for cross-domain comparison.
6. **Es et al. (2024), "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (EACL 2024)** — the evaluation framework we use; important to cite and understand deeply.
7. **Chen et al. (2021), "FinQA: A Dataset of Numerical Reasoning over Financial Data" (EMNLP 2021)** — foundational financial NLP dataset; contextualizes the difficulty of numerical reasoning in our domain.
8. **Ji et al. (2025), "PHANTOM: A Benchmark for Hallucination Detection in Financial Long-Context QA" (NeurIPS 2025)** — the most recent directly related work; establishes that hallucination detection in finance remains an open problem with significant benchmark gaps.

---

### Other Resources and Components to Implement

Beyond the libraries listed above, we will need access to the OpenAI API (GPT-3.5-turbo) and
HuggingFace model hub (Mistral-7B). We will implement the query rewriting component from
scratch using a prompted GPT-3.5-turbo call, following the approach described in prior work
on query expansion for RAG. The hallucination taxonomy and manual annotation schema will also
be developed by the group, informed by the RAGTruth and PHANTOM taxonomies from prior work.

---

### Visualizations and Results

We expect to produce: (1) a main results table comparing all RAG configurations across RAGAS
faithfulness, answer relevancy, and F1 metrics; (2) a bar chart breaking down hallucination
type frequencies (numerical vs. entity vs. reasoning vs. unsupported) for the baseline
configuration; (3) a confusion matrix or classification report for the RoBERTa hallucination
classifier; and (4) 2–3 illustrative examples of each hallucination type drawn from the error
analysis, presented as a qualitative case study table in the paper.

---

### Timeline and Working Plan

| Week | Dates | Milestone |
|------|-------|-----------|
| Week 1 | 6/5 – 6/12 | Group registration; divide related work readings (2 papers/person); submit proposal |
| Week 2 | 6/13 – 6/19 | Set up codebase; implement baseline RAG pipeline on FinanceBench; verify evaluation scripts |
| Week 3 | 6/20 – 6/26 | Implement dense RAG and enhanced RAG; run initial experiments with GPT-3.5-turbo |
| Week 4 | 6/27 – 7/3 | Complete preliminary results; write preliminary paper (4 pages); read remaining papers for lit review (20–24 total) |
| Week 5 | 7/4 – 7/10 | Run Mistral-7B experiments; begin manual error annotation (50 cases) |
| Week 6 | 7/11 – 7/17 | Train RoBERTa hallucination classifier; complete error taxonomy |
| Week 7 | 7/18 – 7/24 | Full results analysis; write results and discussion sections |
| Week 8 | 7/25 – 7/31 | Final paper writing, revision, and submission |
| Week 9 | 8/1 – 8/4 | Poster creation and video recording |

---

### Group Roles

Since the group prefers a roughly equal split, each member will contribute to all phases but
will take primary ownership of one component:

- **Person 1 (Literature Lead):** Owns the related work section; reads and summarizes the most papers; also contributes to writing the introduction and background sections.
- **Person 2 (Pipeline Engineer):** Owns the RAG pipeline implementation (LangChain, FAISS, BM25, query rewriting); also writes the methodology section.
- **Person 3 (Experiments Lead):** Owns running evaluations, collecting results, and producing all tables and figures using RAGAS and standard metrics.
- **Person 4 (Analysis Lead):** Owns the manual error annotation, hallucination taxonomy development, and the RoBERTa classifier; writes the error analysis and discussion sections.

All four members will contribute to the final paper draft, poster, and video presentation,
with each person presenting the component they were most responsible for.
