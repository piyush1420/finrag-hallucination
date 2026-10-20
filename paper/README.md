# Paper

**Analyzing Hallucination Patterns in RAG-Based Financial Question Answering** — CS6120 NLP.

Format: **Word (.docx)** — agreed with the instructor. LaTeX is no longer used.
The `.docx` is generated from a Markdown source with `pandoc` (ACL 2023 Word template
styles + resolved citations), then opened in Word for final formatting.

## Files
### Final paper (M3, due 2026-07-31)
- `CS6120NLP_Group7_RAG_Hallucination_final_paper.docx` — **the final paper** (Word, hand-formatted). This is the submission artifact.
- `main.md` — Markdown source the docx was first generated from; final formatting was then done by hand in Word.

### Preliminary paper (M2, submitted 2026-07-03) — frozen record
- `CS6120NLP_RAG_Hallucination_preliminary_paper.pdf` — the PDF that was submitted.
- `preliminary_paper.docx` — Word version of the preliminary paper (frozen snapshot).
- `preliminary_paper.md` — Markdown source of the preliminary paper (frozen snapshot).

### Shared assets
- `references.bib` — references, grouped by theme (RAG methods / hallucination detection /
  financial NLP). Verify author lists + pages before final submit.
- `fig_eda_overview.png` — EDA overview figure.
- `litreview_notes/` — per-member study notes feeding the related-work section.
- `templates/acl2023.docx` — instructor's ACL 2023 **Word** template (pandoc reference doc).

## Build (Word)
Regenerate `main.docx` from `main.md` with the ACL Word template applied:
```bash
cd paper
pandoc main.md --citeproc --bibliography=references.bib \
  --reference-doc=templates/acl2023.docx -o CS6120NLP_Group7_RAG_Hallucination_final_paper.docx
```
Then open `main.docx` in Word for final formatting and submit.

## Before submitting — checklist
- [ ] Fill in all four author names + emails.
- [ ] Full-group proofread; confirm page limit.
- [ ] Submit per course instructions before 2026-07-31.
