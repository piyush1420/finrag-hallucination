# FinanceBench — Paper Summary & Study Notes

**Prepared by:** Bhalchandra Shinde · **Task:** T11 (literature review) · **Theme:** Financial NLP & QA

**Paper:** "FinanceBench: A New Benchmark for Financial Question Answering"
**Authors:** Islam, Kannappan, Kiela, Qian, Scherrer, Vidgen (Patronus AI / Contextual AI / Stanford)
**Published:** November 2023 (arXiv:2311.11944) · BibTeX key: `islam2023financebench`

---

## 1. What is FinanceBench?
An **open-book** benchmark for evaluating LLMs on **financial questions** from real public-company filings.
- 10,231 questions · 40 US public companies · 361 filings (10-K/10-Q/8-K/Earnings), 2015–2023.
- Each entry: question, gold answer, evidence string, page number.
- Fills a gap: prior benchmarks lacked realistic retrieval + reasoning tasks analysts actually perform.

## 2. Three question types
| Type | Description | Count |
|---|---|---|
| Domain-Relevant | generic to any public company ("Did it pay a dividend?") | 925 |
| Novel-Generated | analyst-crafted, company-specific, reasoning | 1,323 |
| Metrics-Generated | programmatic from 18 metrics (income/balance/cash flow) | 7,983 |

Metrics-generated dominate (~78%).

## 3. Reasoning taxonomy
- Information Extraction — 28% (just find the number)
- **Numerical Reasoning — 66%** (calculate/compare)
- Logical Reasoning — 6% (judge/assess)

→ Most financial QA is **not** simple lookup; it requires math.

## 4. Models & configs tested
- **Models:** GPT-4 / GPT-4-Turbo, Claude 2 (100k), Llama 2 70B.
- **5 retrieval settings:** Closed Book · Oracle (evidence pages given) · Shared Vector Store (all 360 docs) · Single Vector Store (per-doc) · Long Context (full filing).
- 16 configs evaluated on a **150-question human-reviewed sample** (2,400 responses).

## 5. Key results (accuracy)
| Model + config | Correct | Incorrect | Refused |
|---|---|---|---|
| GPT-4-Turbo — Closed Book | 9% | 3% | 88% |
| Llama2 — Shared Store | 19% | 70% | 11% |
| GPT-4-Turbo — Shared Store | 19% | 13% | 68% |
| Llama2 — Single Store | 41% | 54% | 5% |
| GPT-4-Turbo — Single Store | 50% | 11% | 39% |
| Claude 2 — Long Context | 76% | 21% | 3% |
| **GPT-4-Turbo — Long Context** | **79%** | 17% | 4% |
| GPT-4-Turbo — Oracle (ceiling) | 85% | 15% | 0% |

Best realistic setup ~79%; oracle ceiling 85% (models fail 15% even with perfect context).

## 6. Behavioral differences
- **Llama 2** → more **wrong** answers; **GPT-4-Turbo** → more **refusals** when uncertain. Refusals are safer than confident wrong answers.
- Claude 2 vs GPT-4-Turbo (long context): similar accuracy; both had more incorrect than refusals — an enterprise concern.

## 7. Prompt order matters
- **Context-First** (doc → question): GPT-4-Turbo 78%, Claude 2 76%.
- **Context-Last** (question → doc): GPT-4-Turbo 25%, Claude 2 37%.
~50-point swing. **Practical tip: put context BEFORE the question in long-context prompts.**

## 8. By question type
Best on novel/domain-relevant (extraction/general knowledge); **worst on metrics-generated** (multi-step numerical reasoning).

## 9. Qualitative error themes
1. High-quality correct (sometimes better than gold). 2. Different-but-valid (qualitative Qs). 3. **Hallucinations** — confident, well-structured, wrong; hardest to catch; more common in Llama 2. 4. Helpful refusals. 5. Irrelevant responses.

## 10. Limitations
Single-turn only · public filings only · no cross-company comparison · US public companies only · a few too-easy questions · some ambiguous answers.

## 11. Conclusions
All tested LLMs have meaningful weaknesses; best non-oracle fails ~21%, often via confident hallucination. Recommends: rigorous pre-deployment eval, retrieval augmentation, double-checking, context-before-question. Future: fine-tuning, few-shot, CoT, tool augmentation, multi-turn.

## Connection to our work
FinanceBench is our dataset — we use the **open-source 150-question subset**. Its numeric skew (66% numerical reasoning), the hallucination failure theme, the oracle ceiling below 100%, and the refuse-vs-wrong split all directly motivate our RQs. Our GPT-3.5 baseline mirrors the "stronger model refuses rather than fabricates" pattern (it abstains on ~69% of questions under weak BM25 retrieval), and our prompt template adopts their context-before-question finding. FinanceBench measures *answer accuracy*; our contribution is to isolate and taxonomize *RAG hallucination* specifically.

## Quick reference
Total Q 10,231 · companies 40 · filings 361 · eval 150×16=2,400 · best realistic ~79% (GPT-4-Turbo long context) · oracle ~85% · closed-book ~9% · numerical-reasoning 66%.
