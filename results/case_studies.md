# Qualitative case studies (T23, paper Table 2)

Two representative cases per hallucination type, drawn from the 50 labeled answered-but-incorrect cases (`annotations/failure_cases_50.csv`). Each row shows what the model answered, the gold answer, and why the error falls in that type.

| Type | Company | Question | Model answer | Gold | Why it is wrong |
|---|---|---|---|---|---|
| Numerical | General Mills | According to the information provided in the statement of cash flows, what is the FY… | $25,825 million | $3215.00 | wrong FCF (25,825 vs 3,215), order-of-magnitude off |
| Numerical | Coca-Cola | What is Coca Cola's FY2022 dividend payout ratio (using total cash dividends paid an… | Coca Cola's FY2022 dividend payout ratio is 42.68% ($4… | 0.8 | wrong dividends figure -> payout 0.43 vs 0.8 |
| Entity | AMD | What drove revenue change as of the FY22 for AMD? | The increase in revenue for AMD was primarily driven b… | In 2022, AMD reported Higher sales of their… | answered with Lockheed F-16/F-22 content — wrong company (retrieval miss) |
| Entity | American Express | What are the geographies that American Express primarily operates in as of 2022? | Phoenix, Arizona, Sunrise, Florida, Gurgaon, India, Br… | United States, EMEA, APAC, and LACC | listed office cities, not operating regions (US/EMEA/APAC/LACC) |
| Unsupported | American Express | Which debt securities are registered to trade on a national securities exchange unde… | American Express Company Deferred Compensation Plan fo… | There are none | gold is 'none'; fabricated a specific security |
| Unsupported | Pfizer | As of Q2'2023, is Pfizer spinning off any large business segments? | As of Q2'2023, Pfizer is not spinning off any large bu… | Yes, it's spinning off Upjohn. | said 'not spinning off'; Pfizer is spinning off Upjohn |
| Reasoning | Verizon | Has Verizon increased its debt on balance sheet between 2022 and the 2021 fiscal per… | Yes, Verizon increased its total debt on the balance s… | No. Verizon's debt decreased by $229 million. | used FY2020->2021 not 2021->2022 -> opposite conclusion |
| Reasoning | 3M | Is 3M a capital-intensive business based on FY2022 data? | Yes, 3M is a capital-intensive business based on FY202… | No, the company is managing its CAPEX and Fi… | concluded 'Yes' capital-intensive; gold 'No' (opposite interpretation) |
