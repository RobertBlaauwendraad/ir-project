#!/usr/bin/env python3
import os
import sys
import logging
import pandas as pd
import pyterrier as pt
import pyt_splade
import torch
import ir_datasets_owi


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = detect_device()
logger.info(f"Using device: {device}")

data_dir = "./data"
splade_index_dir = os.path.join(data_dir, "owi_splade_index")

logger.info(f"Looking for SPLADE shards in {splade_index_dir}...")

# Generate paths for potential shards (0 to 49)
shard_candidates = [os.path.join(splade_index_dir, f"part_{i}") for i in range(50)]
valid_shards = [p for p in shard_candidates if os.path.exists(os.path.join(p, "data.properties"))]

if not valid_shards:
    logger.error("No valid shards found! Did the indexing job finish?")
    sys.exit(1)

logger.info(f"Found {len(valid_shards)} valid shards. Loading MultiIndex...")
index_ref = pt.MultiIndex([pt.IndexFactory.of(p) for p in valid_shards])

logger.info("Initializing SPLADE Model...")
# Use the HuggingFace ID directly
local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "splade-cocondenser-ensembledistil")
splade_model = local_model_path

splade = pyt_splade.Splade(
        model=splade_model,
        device=device,
        max_length=256
    )

logger.info("Loading OWI Dataset...")
ir_datasets_owi.register() 
dataset = pt.get_dataset("irds:owi/dev")


splade_retr = splade.query_encoder() >> pt.BatchRetrieve(index_ref, wmodel="Tf")


logger.info("Starting Experiment...")

topics = dataset.get_topics()
# Ensure query column exists (some datasets use 'title', some 'text')
if "text" in topics.columns:
    topics["query"] = topics["text"]
elif "title" in topics.columns:
    topics["query"] = topics["title"]

# Try to load Qrels (Ground Truth)
try:
    qrels = dataset.get_qrels()
    has_qrels = True
    logger.info("Qrels found! Running full evaluation.")
except:
    has_qrels = False
    logger.info("No Qrels found. Running in Search-Only mode.")

if has_qrels:
    from pyterrier.measures import MAP, nDCG, Recall
    results = pt.Experiment(
        [splade_retr],
        topics,
        qrels,
        eval_metrics=[MAP, nDCG@10, Recall@100],
        names=["SPLADE OWI Baseline"]
    )
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(results)
    results.to_csv("owi_splade_results.csv")
else:
    # Just retrieve top docs for the first few topics
    logger.info("Retrieving top 10 results for first 10 queries...")
    res = splade_retr.transform(topics.head(10))
    print(res[['qid', 'docno', 'score', 'rank']])
    res.to_csv("owi_splade_search_output.csv")

logger.info("Done!")