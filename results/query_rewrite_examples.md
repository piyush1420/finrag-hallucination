# Query-rewrite examples (T15)

GPT-3.5 rewrites each question into a retrieval query before dense search (spelling out abbreviations, adding filing line-item terms, dropping conversational framing). Full log in `results/raw_outputs/enhanced_rewrite.jsonl`.

**1. Original:** What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.

**   Rewritten:** Find the capital expenditure amount for fiscal year 2018 in USD millions for 3M in the cash flow statement.

**2. Original:** Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.

**   Rewritten:** Retrieve the FY2018 balance sheet for 3M and find the net Property, Plant, and Equipment (PPNE) value in USD billions.

**3. Original:** Is 3M a capital-intensive business based on FY2022 data?

**   Rewritten:** Is 3M a capital-intensive business based on fiscal year 2022 data?

**4. Original:** What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.

**   Rewritten:** Operating margin change FY2022 3M. If operating margin not useful, explain why.

**5. Original:** If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?

**   Rewritten:** Exclude impact mergers acquisitions segment drag overall growth 2022 3M
