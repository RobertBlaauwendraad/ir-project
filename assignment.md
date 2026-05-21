# Project 1: Build an IR System from Scratch

## Context

You are launching an IR system for a document collection for the first time, with no available test collection and no historical user queries. The project is split into two steps.

---

## Step 1: Build the Best IR System

Choose an existing document and test collection (e.g., Robust04) as your development ground:

- Index the document collection
- Choose a ranking model that goes **beyond lexical retrieval** — for example learning-to-rank, cross-encoders, dense retrieval, learned sparse retrieval, LLM-based models, etc.
- The method may be complex (e.g., involves expansion or classification)
- Evaluate the model on this existing test collection

---

## Step 2: Improve Your Model on the OWS Collection

Apply your model to the **Open Web Search (OWS)** document collection:

- A small set of queries will be provided to you
- Improve your model — this may involve manual steps (e.g., query expansion or setting weights for a regression model)
- The aim is to understand what makes retrieval better in a real web setting

---

## OWS Collection Features

The OWS collection includes the following document features:

- **Textual:** page title, main content (minimal HTML tags), meta description
- **Link-based:** hyperlinks / web graph, anchor texts, centrality measures (PageRank, inlink count, etc.)
- **Topical:** Curlie.org topic labels, Schema.org information (Microdata/JSON-LD)

Note: web content can be spammy, noisy, or otherwise imperfect.

---

## Summary

> Build your own IR system for the OWS dataset, evaluate it on an existing test collection, then test it using the outcomes of the evaluation project. The system must go beyond lexical retrieval — neural or learning-to-rank approaches are expected.

---

## References

- Dinzinger et al. (2023). *OWler: Preliminary results for building a collaborative open web crawler.*
- Granitzer et al. (2024). *Impact and development of an Open Web Index for open web search.*
- Hendriksen et al. (2024). *The Open Web Index: Crawling and Indexing the Web for Public Use.*
- Burges (2010). *From RankNet to LambdaRank to LambdaMART: An overview.*
- Nogueira et al. (2020). *Document Ranking with a Pretrained Sequence-to-Sequence Model.*
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- Khattab et al. (2020). *ColBERT: Efficient and effective passage search via contextualized late interaction over BERT.*
- Formal et al. (2021). *SPLADE: Sparse lexical and expansion model for first stage ranking.*