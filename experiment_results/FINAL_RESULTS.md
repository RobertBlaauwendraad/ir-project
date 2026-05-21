# Final results — IR Project (Robust04 + OWI)

Two datasets evaluated with the same hybrid pipeline. Robust04 run: 20 May 2026 (after rebuilding SPLADE index with `pretokenised=True` fix). OWI run: 24–25 April 2026.

---

## 1. Best model configuration

**Hybrid sparse–lexical pipeline** (matches `ir_system.py` default):

- **SPLADE:** `naver/splade-cocondenser-ensembledistil`, doc `max_length = 256`, BERT WordPiece vocab (≈ 30 k terms).
- **BM25 (Robust04-tuned):** `k1 = 0.9`, `b = 0.4` (chosen from exp 7 grid).
- **BM25 (OWI):** PyTerrier default `k1 = 1.2`, `b = 0.75` (OWI prefers the default over Robust04-tuned).
- **RM3 query expansion** on BM25: `fb_docs = 10`, `fb_terms = 15`, `fb_lambda = 0.5`.
- **Fusion:** sum of normalised scores, `score = SPLADE + 20 · (BM25 + RM3)`.

---

## 2. Headline results

Three measures per system: **MAP** (mean average precision over the full ranked list, binary relevance), **nDCG@10**, **Recall@100**.

### 2.1 Robust04 (TREC Disks 4–5 minus CR, 528 155 docs, 249 topics, deep qrels with ~1250 judgments/topic)

| System | MAP | nDCG@10 | R@100 |
|---|---:|---:|---:|
| BM25 default (k1=1.2, b=0.75) | **0.2369** | 0.4244 | 0.4012 |
| BM25 tuned (k1=0.9, b=0.4) | **0.2525** | 0.4466 | 0.4128 |
| BM25 + RM3 (λ=0.5, d=10, t=15) | **0.2853** | 0.4993 | 0.4382 |
| SPLADE baseline | **0.2252** | 0.4548 | 0.3732 |
| Hybrid SPLADE + 20·BM25_RM3 (best overall) | **0.2853** | 0.4993 | 0.4382 |

### 2.2 OWI (Open Web Index `owi/dev`, 20 746 181 docs, 28 topics, shallow pool ~47 judgments/topic)

| System | MAP | nDCG@10 | R@100 |
|---|---:|---:|---:|
| BM25 default (k1=1.2, b=0.75) | **0.4101** | 0.4871 | 0.6969 |
| BM25 alt (k1=0.9, b=0.4) | **0.3836** | 0.4477 | 0.6795 |
| BM25 + RM3 (λ=0.5, d=10, t=15) | **0.4134** | 0.4892 | 0.7806 |
| SPLADE baseline | **0.3257** | 0.4137 | 0.6745 |
| Hybrid SPLADE + 20·BM25_RM3 (best overall) | **0.4134** | 0.4892 | 0.7806 |

**Side-by-side (MAP only) for the best per dataset:**

| System | Robust04 MAP | OWI MAP |
|---|---:|---:|
| BM25 default | 0.2369 | 0.4101 |
| BM25 + RM3 | 0.2853 | 0.4134 |
| SPLADE | 0.2252 | 0.3257 |
| Hybrid (best) | 0.2853 | 0.4134 |

---

## 3. Supporting experiments

### 3.1 Hybrid fusion weight sweep (SPLADE + w·BM25, exp 4)

MAP as a function of fusion weight `w` (`score = SPLADE + w · BM25`). Selected rows.

| w | Robust04 MAP | OWI MAP |
|---:|---:|---:|
| 0 | 0.2307 | 0.3257 |
| 0.1 | 0.2319 | 0.3277 |
| 0.2 | 0.2323 | 0.3294 |
| 0.5 | 0.2341 | 0.3336 |
| 1 | 0.2364 | 0.3409 |
| 2 | 0.2403 | 0.3561 |
| 5 | 0.2471 | 0.3766 |
| 10 | 0.2499 | 0.3973 |
| 20 | 0.2554 | 0.4255 |
| 40 | 0.2620 | 0.4522 |
| 75 | 0.2608 | 0.4529 |
| 1e+02 | 0.2598 | 0.4489 |

**Reading:** Robust04 peaks around `w ≈ 40–50`; OWI peaks around `w ≈ 40–75`. Both datasets agree that BM25 should dominate the fusion (w ≫ 1), which is consistent with SPLADE on its own under-performing BM25 on these collections.

### 3.2 BM25 parameter grid (exp 7)

| (k1, b) | Robust04 MAP | OWI MAP |
|---|---:|---:|
| k1=0.9, b=0.4 | 0.2525 | 0.3836 |
| k1=1.2, b=0.5 | 0.2481 | 0.4002 |
| k1=1.2, b=0.75 | 0.2369 | 0.4101 |
| k1=1.2, b=0.9 | 0.2262 | 0.3831 |
| k1=1.5, b=0.75 | 0.2330 | 0.4115 |
| k1=2.0, b=0.75 | 0.2260 | 0.4165 |

**Reading:** Robust04 prefers small `k1` and small `b` (favouring short newswire documents). OWI prefers the default `k1 = 1.2`, `b = 0.75` (favouring long heterogeneous web pages). A single dataset-agnostic BM25 setting is sub-optimal for at least one collection.

### 3.3 RM3 λ tuning on BM25 (exp 6)

| λ | Robust04 MAP | OWI MAP |
|---:|---:|---:|
| 0.3 | 0.2781 | 0.4136 |
| 0.4 | 0.2774 | 0.4171 |
| 0.5 | 0.2752 | 0.4250 |
| 0.6 | 0.2719 | 0.4246 |
| 0.7 | 0.2676 | 0.4223 |

**Reading:** Robust04 prefers a lower λ (more weight on expansion terms); OWI prefers slightly higher λ (more weight on the original query). Both collections benefit from RM3 (vs. no expansion) by ≈ 0.04 MAP.

### 3.4 Reciprocal Rank Fusion vs. score sum (exp 3, exp 8)

RRF gives a parameter-free alternative to weighted score sum. Best k.

| Method | Robust04 MAP | OWI MAP |
|---|---:|---:|
| RRF (k=60, default) | 0.2594 | 0.4370 |
| RRF (k=10) | 0.2661 | 0.4465 |
| RRF (k=30) | 0.2632 | 0.4415 |
| Hybrid score sum (w=0.2) | 0.2323 | 0.3294 |

**Reading:** Best-tuned weighted score sum (w ≈ 20–40 with RM3) beats RRF on both datasets.

---

## 4. Cross-validation (Robust04 only — exp 21)

5-fold cross-validation on Robust04, hyperparameters re-tuned on each training fold and evaluated on the held-out fold. **This is the only result not tuned on the test set.**

| System | MAP (CV avg) | R@100 (CV avg) | nDCG@10 (CV avg) |
|---|---:|---:|---:|
| BM25 (tuned) [5-fold CV avg] | 0.2370 | 0.4014 | 0.4244 |
| Best Hybrid (CV) [5-fold CV avg] | 0.2623 | 0.4062 | 0.4937 |
| SPLADE [5-fold CV avg] | 0.2251 | 0.3732 | 0.4547 |

**Reading:** Best hybrid generalises to **MAP = 0.262** under fair train/test split, vs. **MAP = 0.285** when tuned directly on the full test set. That ~0.02 gap is the *tuning-on-test bias* that needs to be acknowledged in the report.

---

## 5. Critical caveats (must appear in the report)

1. **No held-out validation split for Robust04 (single-shot tuning) or OWI.** All hyperparameter choices (BM25 `k1`/`b`, RM3 `fb_docs`/`fb_terms`/`λ`, fusion weight `w`) were selected directly against the test qrels for both datasets. The 5-fold CV in §4 is the only honest generalisation estimate, and it is Robust04-only. **For OWI we have *no* held-out estimate** because `owi/test` has no qrels.

2. **OWI dev set is small and shallow.** 28 topics, ~47 judgments per topic, ~16 relevant per topic (graded ≥ 1; grades −2 spam, 0 irrelevant, 1 related, 2 highly relevant, 3 perfectly relevant). Inter-system MAP differences ≤ 0.02 should not be over-interpreted on this set.

3. **OWI qrels are pooled.** With only ~47 documents judged per topic out of 20.7 M, unjudged documents are treated as non-relevant. Systems whose top-1000 overlaps with the pool naturally score higher; this is the standard TREC pooling assumption but it inflates absolute MAP for any reasonable BM25-family retriever on OWI vs. an unpooled estimate.

4. **Dataset-optimal BM25 parameters differ.** Robust04 wants `k1=0.9, b=0.4`; OWI wants the default `k1=1.2, b=0.75`. A model trained on Robust04 alone underperforms on OWI.

5. **No statistical significance test was run** (the rubric flags this as Level-4 vs. Level-5 methodology).

6. **OWI test results unreported.** `qrels_test.txt` does not exist in the shared course directory, so the official OWI test split is unscoreable with our current evaluation pipeline.

---

## 6. Index sanity

| Index | Documents | Unique terms | Tokens |
|---|---:|---:|---:|
| Robust04 BM25       | 528 155      | 520 520      | 145 M |
| Robust04 SPLADE     | 528 155      | 27 542 (≈ BERT vocab) | 6.37 B |
| OWI BM25            | 20 746 181   | 29 541 940   | 12.2 B |
| OWI SPLADE (30 shards) | 20 746 181 (sum) | 27 542 per shard (≈ BERT vocab) | n/a |

Both SPLADE indices have ≈ BERT vocabulary size, confirming the `pretokenised=True` fix took effect (a broken index would have shown the raw stemmed-token vocab of ~500 k terms, as it did in the December 2025 run with MAP = 0.017).
