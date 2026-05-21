# Feedback: Final Report
**Course:** 2526 Information Retrieval (KW1 V) — Radboud University
**Student:** Robert Blaauwendraad
**Total Score:** 18 / 25

The students did a good job, managing to get SPLADE running and comparing it to a sparse BM25 baseline (and several other combinations). While the report mentioned experimenting with OWS results, this is not done; which was the core part of this project.

---

## Rubric

### Originality — Score: 4 / 5

| Level 5 | Level 4 | Level 3 | Level 2 | Level 1 |
|---|---|---|---|---|
| Surprising: Noteworthy new problem, technique, methodology, or insight. | Creative: Relatively few people in our community would have put these ideas together. | Somewhat conventional: A number of people could have come up with this if they thought about it for a while. | Rather straightforward: Obvious, or a minor improvement on familiar techniques. | Significant portions have actually been done before or done better. |

**Feedback:** Interesting combination of sparse retrieval (with query expansion) and learned sparse retrieval.

---

### Proposed Method / Solution / Analysis / Application — Score: 2 / 5

| Level 5 | Level 4 | Level 3 | Level 2 | Level 1 |
|---|---|---|---|---|
| Attractive and/or innovation, which is well motivated and justified. | Appropriate approach, which is solid, motivated and justified. | Reasonable approach, but the motivation and justification could be improved. | Potentially reasonable approach, but there are many concerns. | Approach is flawed/poor. |

**Feedback:** The approach itself seems solid, but it seems the students only evaluated the models on the Robust04 collection, not on the OWS test collection they were also meant to use. No significant test is performed.

---

### Methodology — Score: 4 / 5

| Level 5 | Level 4 | Level 3 | Level 2 | Level 1 |
|---|---|---|---|---|
| Very strong and appropriate methodology that fully supports claims. | Strong/appropriate methodology that supports claims. | Appropriate methodology but has some small issues, unlikely to affect claims. | Weak methodology that has some issues, which bring claims into doubt. | Questionable methodology lacking in numerous areas. |

**Feedback:** Methodology seems decent, with tuning wherever possible and different experiments to support different claims. Some details are missing — e.g., did the students use a separate train/val/test split or were the models tuned on the test set?

---

### Quality of Presentation — Score: 4 / 5

| Level 5 | Level 4 | Level 3 | Level 2 | Level 1 |
|---|---|---|---|---|
| Very well written in every aspect, a pleasure to read, easy to follow. | With all the essential content and understandable by most readers. | Missing a few important details but the major points were clear. | Important questions were hard to resolve even with effort. | Much of the report is confusing. |

**Feedback:** The report is well-written and easy to follow, but some important details are missing. The link to the Git repository also results in a 404.

---

### Impact — Score: 4 / 5

| Level 5 | Level 4 | Level 3 | Level 2 | Level 1 |
|---|---|---|---|---|
| Will affect the field by altering other people's choice of research topics or basic approach. | Some of the ideas or results will substantially help other people's ongoing research. | Interesting but not too influential. If published, the work would be cited, but mainly for comparison or as a source of minor contributions. | Marginally interesting. May or may not be cited or used. | Likely to have little impact on the field. |

**Feedback:** Results are interesting and useful, but impact could have been higher if the OWS test collection had also been considered.

---

## Key Issues to Address

1. **Missing OWS evaluation** — the system was never tested on the OWS collection, which was a core requirement of the project. This is the primary reason for the low method score.
2. **No train/val/test split clarification** — it is unclear whether hyperparameter tuning was done on the test set, which undermines reproducibility claims.
3. **Broken Git repository link** — the repository URL returns a 404.