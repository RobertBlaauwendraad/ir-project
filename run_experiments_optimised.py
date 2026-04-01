"""
IR Experiments Runner

Usage:
    python run_experiments_optimised.py                    # Run all experiments
    python run_experiments_optimised.py --exp 1 2 3       # Run specific experiments
    python run_experiments_optimised.py --exp 1-5         # Run experiment range
    python run_experiments_optimised.py --list            # List all experiments
"""

import argparse
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import reduce
from typing import List, Optional

import ir_datasets
import numpy as np
import pandas as pd
import pyt_splade
import pyterrier as pt
import pyterrier_alpha as pta
import torch
from pyterrier.measures import MAP, nDCG, Recall, P

import ir_datasets_owi

warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

DATASET_CONFIGS = {
    "robust04": {
        "irds_id": "disks45/nocr/trec-robust-2004",
        "index_prefix": "robust04",
        "query_field": "title",
    },
    "owi": {
        "irds_id": "owi/dev",
        "index_prefix": "owi",
        "query_field": "text",
    },
    "owi/subsampled": {
        "irds_id": "owi/subsampled/dev",
        "index_prefix": "owi_subsampled",
        "query_field": "text",
    },
}

_dataset = None
_splade = None
_splade_retr = None
_bm25_retr = None
_bm25_index_ref = None
_splade_index_ref = None
_topics = None
_qrels = None

_cache: dict = {}
_cache_dir: str = ""

#region logging

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiments_{timestamp}.log")

    logger = logging.getLogger("ir_experiments")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to: {log_file}")
    return logger

#endregion

#region device detection

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

#endregion

#region cache

def _cache_path(name: str) -> str:
    return os.path.join(_cache_dir, f"{name}.pkl")


def get_cached(name: str, compute_fn, logger: logging.Logger) -> pd.DataFrame:
    """Return cached DataFrame (memory → disk → compute)."""
    if name in _cache:
        return _cache[name]

    path = _cache_path(name)
    if os.path.exists(path):
        logger.info(f"[cache] Loading '{name}' from disk ({path})")
        df = pd.read_pickle(path)
        _cache[name] = df
        return df

    logger.info(f"[cache] Computing '{name}'…")
    df = compute_fn()
    df.to_pickle(path)
    logger.info(f"[cache] Saved '{name}' to {path}")
    _cache[name] = df
    return df

#endregion

#region retrieval

def _retrieve_shards_parallel(shard_paths: list, topics: pd.DataFrame,
                               wmodel: str, logger: logging.Logger,
                               num_results: int = 1000) -> pd.DataFrame:
    """Retrieve from multiple shards in parallel using threads."""
    logger.info(f"Retrieving from {len(shard_paths)} shard(s) in parallel (wmodel={wmodel})…")

    def retrieve_one(path):
        retr = pt.terrier.Retriever(pt.IndexFactory.of(path),
                                    wmodel=wmodel,
                                    num_results=num_results)
        return retr.transform(topics)

    with ThreadPoolExecutor(max_workers=min(len(shard_paths), 8)) as ex:
        futures = {ex.submit(retrieve_one, p): p for p in shard_paths}
        parts = []
        for fut in as_completed(futures):
            parts.append(fut.result())

    combined = pd.concat(parts, ignore_index=True)
    # Keep only the top num_results per query across all shards
    combined = (combined
                .sort_values(["qid", "score"], ascending=[True, False])
                .groupby("qid")
                .head(num_results)
                .reset_index(drop=True))
    return combined

#endregion

def _fuse(res_a: pd.DataFrame, res_b: pd.DataFrame,
          w_a: float = 1.0, w_b: float = 1.0) -> pd.DataFrame:
    """
    Linear combination of two pre-retrieved result sets.
    Returns a DataFrame with columns [qid, docno, score] ranked per query.
    """
    a = res_a[["qid", "docno", "score"]].copy()
    b = res_b[["qid", "docno", "score"]].copy()
    a["score"] *= w_a
    b["score"] *= w_b

    merged = pd.merge(a, b, on=["qid", "docno"], how="outer", suffixes=("_a", "_b"))
    merged["score"] = merged["score_a"].fillna(0) + merged["score_b"].fillna(0)
    merged = merged[["qid", "docno", "score"]]
    merged["rank"] = (merged.groupby("qid")["score"]
                      .rank(ascending=False, method="first")
                      .astype(int))
    return merged.sort_values(["qid", "rank"]).reset_index(drop=True)


#region setup

def load_sharded_index(index_dir: str, logger: logging.Logger):
    shard_candidates = [os.path.join(index_dir, f"part_{i}") for i in range(50)]
    valid_shards = [p for p in shard_candidates
                    if os.path.exists(os.path.join(p, "data.properties"))]

    if valid_shards:
        logger.info(f"Found {len(valid_shards)} shards in {index_dir}")
        return valid_shards, True
    elif os.path.exists(os.path.join(index_dir, "data.properties")):
        logger.info(f"Found single index at {index_dir}")
        return [index_dir], False
    else:
        raise FileNotFoundError(f"No valid index found at {index_dir}")

def setup_environment(logger: logging.Logger, device: str,
                      data_dir: str = "./data",
                      dataset_name: str = "robust04"):
    global _dataset, _splade, _splade_retr, _bm25_retr
    global _bm25_index_ref, _splade_index_ref, _topics, _qrels, _cache_dir

    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Dataset: {dataset_name}")
    torch.manual_seed(26)

    dataset_config = DATASET_CONFIGS[dataset_name]
    index_prefix = dataset_config["index_prefix"]
    irds_id = dataset_config["irds_id"]
    query_field = dataset_config["query_field"]

    # Set up disk cache directory
    _cache_dir = os.path.join(data_dir, "retrieval_cache", index_prefix)
    os.makedirs(_cache_dir, exist_ok=True)
    logger.info(f"Retrieval cache directory: {_cache_dir}")

    if dataset_name.startswith("owi"):
        logger.info("Registering OWI dataset…")
        ir_datasets_owi.register()

    local_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "splade-cocondenser-ensembledistil")
    hf_model_name = "naver/splade-cocondenser-ensembledistil"
    splade_model = local_model_path if os.path.isdir(local_model_path) else hf_model_name
    logger.info(f"Initialising SPLADE from: {splade_model}")
    _splade = pyt_splade.Splade(
        model=splade_model,
        device=device,
        max_length=128
    )

    # Dataset
    logger.info(f"Loading dataset: {dataset_name} (irds:{irds_id})…")
    _dataset = pt.get_dataset(f"irds:{irds_id}")

    data_dir = os.path.abspath(data_dir)
    bm25_index_dir = os.path.join(data_dir, f"{index_prefix}_bm25_index")
    splade_index_dir = os.path.join(data_dir, f"{index_prefix}_splade_index")

    # BM25 index
    logger.info(f"Loading BM25 index from {bm25_index_dir}…")
    bm25_shard_paths, bm25_is_sharded = load_sharded_index(bm25_index_dir, logger)
    _bm25_index_ref = pt.IndexFactory.of(bm25_shard_paths[0])

    if bm25_is_sharded:
        bm25_shard_retrievers = [
            pt.terrier.Retriever(pt.IndexFactory.of(p), wmodel="BM25")
            for p in bm25_shard_paths
        ]

        def retrieve_bm25_shards(topics):
            results = [r.transform(topics) for r in bm25_shard_retrievers]
            combined = pd.concat(results)[["qid", "docno", "score", "query"]]
            combined = (combined
                .sort_values("score", ascending=False)
                .groupby(["qid", "docno"], as_index=False).first()
                .sort_values(["qid", "score"], ascending=[True, False])
                .reset_index(drop=True)
            )
            combined["rank"] = combined.groupby("qid").cumcount()
            return combined

        _bm25_retr = pt.apply.generic(retrieve_bm25_shards)
        logger.info("BM25: single index loaded")

    # SPLADE index
    logger.info(f"Loading SPLADE index from {splade_index_dir}…")
    splade_shard_paths, splade_is_sharded = load_sharded_index(splade_index_dir, logger)
    _splade_index_ref = pt.IndexFactory.of(splade_shard_paths[0])

    if splade_is_sharded:
        query_encoder = _splade.query_encoder()
        shard_retrievers = [
            pt.terrier.Retriever(pt.IndexFactory.of(p), wmodel="Tf")
            for p in splade_shard_paths
        ]

        def retrieve_splade_shards(topics):
            logger.info(f"retrieve_splade_shards called, {len(topics)} topics, {len(shard_retrievers)} shards")
            encoded = query_encoder.transform(topics)
            results = [r.transform(encoded) for r in shard_retrievers]
            combined = pd.concat(results)[["qid", "docno", "score", "query"]]
            combined = (combined
                .sort_values("score", ascending=False)
                .groupby(["qid", "docno"], as_index=False).first()
                .sort_values(["qid", "score"], ascending=[True, False])
                .reset_index(drop=True)
            )
            combined["rank"] = combined.groupby("qid").cumcount()
            return combined

        _splade_retr = pt.apply.generic(retrieve_splade_shards)
        logger.info("SPLADE: single index loaded")

    # Topics / qrels
    _topics = _dataset.get_topics()
    if "query" not in _topics.columns:
        for field in [query_field, "text", "title"]:
            if field in _topics.columns:
                _topics["query"] = _topics[field]
                break
        else:
            raise ValueError(
                f"Cannot find query column in topics. "
                f"Available: {list(_topics.columns)}")
    _qrels = _dataset.get_qrels()

    logger.info("Environment setup complete!")

def splade_results(logger) -> pd.DataFrame:
    return get_cached("splade_base", lambda: _splade_retr.transform(_topics), logger)


def bm25_results(logger) -> pd.DataFrame:
    return get_cached("bm25_base", lambda: _bm25_retr.transform(_topics), logger)


def _make_bm25_tuned(k1: float = 0.9, b: float = 0.4):
    return pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": k1, "bm25.b": b},
    )


def bm25_tuned_results(logger) -> pd.DataFrame:
    def compute():
        retr = _make_bm25_tuned()
        return retr.transform(_topics)
    return get_cached("bm25_tuned_k09_b04", compute, logger)

def _get_text_fields() -> list:
    """Return the correct document text fields for the current dataset."""
    available = set(_dataset.info.get("fields", {}).keys()) if hasattr(_dataset, "info") else set()
    try:
        sample = next(iter(_dataset.irds_ref().docs_iter()))
        if hasattr(sample, "body"):
            return ["title", "body"]
        elif hasattr(sample, "main_content"):
            return ["title", "main_content"]
    except Exception:
        pass
    return ["title", "body"]

def _concat_text(row, fields: list) -> str:
    return " ".join(str(row.get(f) or "") for f in fields)


def _get_text_pipeline(base_retr):
    """Add text fetching + concat to a retriever pipeline."""
    fields = _get_text_fields()
    return (
        base_retr
        >> pt.text.get_text(_dataset, fields)
        >> pt.apply.text(lambda row, f=fields: _concat_text(row, f))
    )


def bm25_rm3_results(logger) -> pd.DataFrame:
    """Tuned BM25 + RM3 (fb_docs=10, fb_terms=15, lambda=0.5)."""
    def compute():
        bm25t = _make_bm25_tuned()
        pipe = (
            _get_text_pipeline(bm25t)
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
            >> bm25t
        )
        return pipe.transform(_topics)
    return get_cached("bm25_tuned_rm3_d10_t15_l05", compute, logger)


def bm25_bo1_results(logger) -> pd.DataFrame:
    """Tuned BM25 + Bo1 (fb_docs=5, fb_terms=15)."""
    def compute():
        bm25t = _make_bm25_tuned()
        pipe = (
            _get_text_pipeline(bm25t)
            >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=5, fb_terms=15)
            >> bm25t
        )
        return pipe.transform(_topics)
    return get_cached("bm25_tuned_bo1_d5_t15", compute, logger)


def save_results(results: pd.DataFrame, name: str, output_dir: str,
                 logger: logging.Logger):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{name}_{timestamp}.csv")
    results.to_csv(filename, index=False)
    logger.info(f"Results saved to: {filename}")


EXPERIMENTS = {}


def register_experiment(exp_id: int, name: str):
    def decorator(func):
        EXPERIMENTS[exp_id] = {"name": name, "func": func}
        return func
    return decorator

#endregion

#region experiments

@register_experiment(1, "Baseline SPLADE")
def exp1_baseline_splade(logger, output_dir):
    logger.info("Running Experiment 1: Baseline SPLADE")
    sp = splade_results(logger)
    results = pt.Experiment(
        [sp], _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE Baseline"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp1_baseline_splade", output_dir, logger)
    return results


@register_experiment(2, "Hybrid Retrieval with Different BM25 Weights")
def exp2_hybrid_retrieval(logger, output_dir):
    logger.info("Running Experiment 2: Hybrid Retrieval with Different BM25 Weights")
    sp = splade_results(logger)
    bm = bm25_results(logger)

    weights = [0.0, 0.05, 0.10, 0.20]
    systems = [_fuse(sp, bm, 1.0, w) for w in weights]
    names = ["SPLADE Only", "Hybrid (w=0.05)", "Hybrid (w=0.10)", "Hybrid (w=0.20)"]
    systems.append(bm)
    names.append("BM25 Only")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp2_hybrid_retrieval", output_dir, logger)
    return results


@register_experiment(3, "Reciprocal Rank Fusion")
def exp3_rrf(logger, output_dir):
    logger.info("Experiment 3: Reciprocal Rank Fusion")
    sp = splade_results(logger)
    bm = bm25_results(logger)
    hybrid_20 = _fuse(sp, bm, 1.0, 0.20)
    rrf = pta.fusion.rr_fusion(sp, bm, k=60)

    results = pt.Experiment(
        [sp, bm, hybrid_20, rrf],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "BM25", "Hybrid (w=0.2)", "RRF"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp3_rrf", output_dir, logger)
    return results


@register_experiment(4, "Hybrid Weight Optimisation")
def exp4_weight_optimisation(logger, output_dir):
    """Experiment 4: Hybrid Retrieval Weight Optimization."""
    logger.info("Experiment 4: Hybrid Weight Optimisation")
    sp = splade_results(logger)
    bm = bm25_results(logger)

    all_weights = (
        list(np.arange(0.0, 1.1, 0.1))
        + [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100]
    )
    systems = [_fuse(sp, bm, 1.0, w) for w in all_weights]
    names = [f"Hybrid (w={w:.2g})" for w in all_weights]

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp4_weight_optimisation", output_dir, logger)
    return results


@register_experiment(5, "Query Expansion Models (Bo1 and RM3)")
def exp5_expansion_models(logger, output_dir):
    """Experiment 5: Expansion Models (Bo1 and RM3)."""
    logger.info("Experiment 5: Query Expansion Models")
    sp = splade_results(logger)
    bm = bm25_results(logger)
    hybrid_20 = _fuse(sp, bm, 1.0, 20.0)

    # Bo1 Query Expansion
    logger.info("Testing Bo1 expansion...")
    bm25_bo1_default = get_cached("bm25_default_bo1_d3_t10", lambda: (
        _get_text_pipeline(_bm25_retr)
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=3, fb_terms=10)
        >> _bm25_retr
    ).transform(_topics), logger)

    # RM3 Query Expansion
    logger.info("Computing RM3 expansion...")
    bm25_rm3_default = get_cached("bm25_default_rm3_d10_t10_l05", lambda: (
        _get_text_pipeline(_bm25_retr)
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10, fb_lambda=0.5)
        >> _bm25_retr
    ).transform(_topics), logger)

    upgraded_bo1 = _fuse(sp, bm25_bo1_default, 1.0, 20.0)
    upgraded_rm3 = _fuse(sp, bm25_rm3_default, 1.0, 20.0)

    results = pt.Experiment(
        [sp, hybrid_20, upgraded_bo1, upgraded_rm3],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "Hybrid (w=20)", "Hybrid + Bo1", "Hybrid + RM3"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp5_expansion_models", output_dir, logger)
    return results


@register_experiment(6, "RM3 Lambda Parameter Optimisation")
def exp6_rm3_lambda(logger, output_dir):
    logger.info("Experiment 6: RM3 Lambda Parameter Optimisation")
    sp = splade_results(logger)
    lambdas = [0.3, 0.4, 0.5, 0.6, 0.7]

    systems, names = [], []
    for lam in lambdas:
        key = f"bm25_default_rm3_d10_t10_l{lam:.1f}".replace(".", "p")
        rm3_res = get_cached(key, lambda l=lam: (
            _get_text_pipeline(_bm25_retr)
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10, fb_lambda=l)
            >> _bm25_retr
        ).transform(_topics), logger)
        systems.append(_fuse(sp, rm3_res, 1.0, 20.0))
        names.append(f"RM3 (lambda={lam})")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp6_rm3_lambda", output_dir, logger)
    return results


@register_experiment(7, "BM25 Parameter Tuning (k1 and b)")
def exp7_bm25_tuning(logger, output_dir):
    logger.info("Experiment 7: BM25 Parameter Tuning")
    bm25_configs = [
        {"k1": 1.2, "b": 0.75},
        {"k1": 0.9, "b": 0.4},
        {"k1": 1.5, "b": 0.75},
        {"k1": 1.2, "b": 0.5},
        {"k1": 2.0, "b": 0.75},
        {"k1": 1.2, "b": 0.9},
    ]

    systems, names = [], []
    for cfg in bm25_configs:
        k, b = cfg["k1"], cfg["b"]
        key = f"bm25_k{k}_b{b}".replace(".", "p")
        res = get_cached(key, lambda k1=k, bv=b: pt.terrier.Retriever(
            _bm25_index_ref, wmodel="BM25",
            controls={"bm25.k_1": k1, "bm25.b": bv},
        ).transform(_topics), logger)
        systems.append(res)
        names.append(f"BM25 (k1={k}, b={b})")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp7_bm25_tuning", output_dir, logger)
    return results


@register_experiment(8, "RRF with Different k Values")
def exp8_rrf_k_values(logger, output_dir):
    logger.info("Experiment 8: RRF with Different k Values")
    sp = splade_results(logger)
    bm = bm25_results(logger)

    systems = [pta.fusion.rr_fusion(sp, bm, k=k) for k in [10, 30, 60, 100, 200]]
    names = [f"RRF (k={k})" for k in [10, 30, 60, 100, 200]]

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp8_rrf_k_values", output_dir, logger)
    return results


@register_experiment(9, "MonoT5 Cross-Encoder Re-ranking")
def exp9_monot5_reranking(logger, output_dir):
    logger.info("Experiment 9: MonoT5 Cross-Encoder Re-ranking")
    try:
        from pyterrier_t5 import MonoT5ReRanker
    except ImportError:
        logger.warning("pyterrier_t5 not installed, skipping MonoT5 experiment")
        return pd.DataFrame()

    mono_t5 = MonoT5ReRanker(model="castorini/monot5-base-msmarco", batch_size=16)
    sp = splade_results(logger)
    bm = bm25_results(logger)
    hybrid_20 = _fuse(sp, bm, 1.0, 20.0)

    def rerank(res, cutoff=100):
        top = res.groupby("qid").head(cutoff)
        return (
            top
            >> pt.text.get_text(_dataset, ["title", "body"])
            >> pt.apply.text(lambda row: (row["title"] or "") + " " + (row["body"] or ""))
            >> mono_t5
        )

    splade_monot5 = rerank(sp)
    hybrid_monot5 = rerank(hybrid_20)

    results = pt.Experiment(
        [sp, splade_monot5, hybrid_monot5],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "SPLADE + MonoT5", "Hybrid + MonoT5"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp9_monot5_reranking", output_dir, logger)
    return results


@register_experiment(10, "Combined Query Expansion with Hybrid")
def exp10_combined_expansion(logger, output_dir):
    logger.info("Experiment 10: Combined Query Expansion with Hybrid")
    sp = splade_results(logger)
    bm = bm25_results(logger)
    hybrid_20 = _fuse(sp, bm, 1.0, 20.0)
    bo1 = bm25_bo1_results(logger)
    rm3 = bm25_rm3_results(logger)

    results = pt.Experiment(
        [hybrid_20, _fuse(sp, bo1, 1.0, 20.0), _fuse(sp, rm3, 1.0, 20.0)],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (w=20)", "Hybrid + Bo1", "Hybrid + RM3"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp10_combined_expansion", output_dir, logger)
    return results


@register_experiment(11, "Best Hybrid Weight with Query Expansion")
def exp11_best_weight_qe(logger, output_dir):
    logger.info("Experiment 11: Best Hybrid Weight with Query Expansion")
    sp = splade_results(logger)
    rm3 = bm25_rm3_results(logger)

    weights = [10, 15, 20, 25, 30, 40]
    systems = [_fuse(sp, rm3, 1.0, w) for w in weights]
    names = [f"Hybrid+RM3 (w={w})" for w in weights]

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp11_best_weight_qe", output_dir, logger)
    return results


@register_experiment(12, "Three-way Fusion")
def exp12_three_way_fusion(logger, output_dir):
    logger.info("Experiment 12: Three-way Fusion")
    sp = splade_results(logger)
    bm = bm25_results(logger)
    rm3 = bm25_rm3_results(logger)

    hybrid_20 = _fuse(sp, bm, 1.0, 20.0)
    three_way_rrf = pta.fusion.rr_fusion(sp, bm, rm3, k=60)

    results = pt.Experiment(
        [sp, hybrid_20, three_way_rrf],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "Hybrid (w=20)", "Three-way RRF"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp12_three_way_fusion", output_dir, logger)
    return results


@register_experiment(13, "Tuned BM25 in Hybrid")
def exp13_tuned_bm25_hybrid(logger, output_dir):
    logger.info("Experiment 13: Tuned BM25 in Hybrid")
    sp = splade_results(logger)
    bm = bm25_results(logger)
    bm_tuned = bm25_tuned_results(logger)
    rm3 = bm25_rm3_results(logger)

    results = pt.Experiment(
        [_fuse(sp, bm, 1.0, 20.0),
         _fuse(sp, bm_tuned, 1.0, 20.0),
         _fuse(sp, rm3, 1.0, 20.0)],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (default BM25)", "Hybrid (tuned BM25)", "Hybrid (tuned BM25 + RM3)"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp13_tuned_bm25_hybrid", output_dir, logger)
    return results


@register_experiment(14, "Fine-tune Weight for Tuned BM25 + RM3")
def exp14_fine_tune_weight(logger, output_dir):
    logger.info("Experiment 14: Fine-tune Weight for Tuned BM25 + RM3")
    sp = splade_results(logger)
    rm3 = bm25_rm3_results(logger)

    weights = [15, 18, 20, 22, 25, 28, 30, 35, 40]
    systems = [_fuse(sp, rm3, 1.0, w) for w in weights]
    names = [f"Hybrid (tuned BM25+RM3, w={w})" for w in weights]

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp14_fine_tune_weight", output_dir, logger)
    return results


@register_experiment(15, "Aggressive RM3 Parameter Tuning")
def exp15_aggressive_rm3(logger, output_dir):
    logger.info("Experiment 15: Aggressive RM3 Parameter Tuning")
    sp = splade_results(logger)

    rm3_configs = [
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.5},
        {"fb_docs": 15, "fb_terms": 20, "fb_lambda": 0.5},
        {"fb_docs": 10, "fb_terms": 25, "fb_lambda": 0.5},
        {"fb_docs": 20, "fb_terms": 15, "fb_lambda": 0.5},
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.4},
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.6},
        {"fb_docs": 15, "fb_terms": 20, "fb_lambda": 0.4},
        {"fb_docs": 5,  "fb_terms": 10, "fb_lambda": 0.5},
        {"fb_docs": 10, "fb_terms": 30, "fb_lambda": 0.5},
    ]

    bm25t = _make_bm25_tuned()
    systems, names = [], []
    for cfg in rm3_configs:
        d, t, l = cfg["fb_docs"], cfg["fb_terms"], cfg["fb_lambda"]
        key = f"bm25_tuned_rm3_d{d}_t{t}_l{l:.1f}".replace(".", "p")
        rm3_res = get_cached(key, lambda _d=d, _t=t, _l=l: (
            _get_text_pipeline(bm25t)
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=_d, fb_terms=_t, fb_lambda=_l)
            >> bm25t
        ).transform(_topics), logger)
        systems.append(_fuse(sp, rm3_res, 1.0, 20.0))
        names.append(f"RM3 (docs={d}, terms={t}, λ={l})")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp15_aggressive_rm3", output_dir, logger)
    return results


@register_experiment(16, "Grid Search BM25 Parameters with RM3")
def exp16_bm25_grid_rm3(logger, output_dir):
    logger.info("Experiment 16: Grid Search BM25 Parameters with RM3")
    sp = splade_results(logger)

    bm25_grid = [
        {"k1": 0.9, "b": 0.4},
        {"k1": 0.8, "b": 0.3},
        {"k1": 0.9, "b": 0.3},
        {"k1": 1.0, "b": 0.4},
        {"k1": 0.7, "b": 0.4},
        {"k1": 0.9, "b": 0.5},
        {"k1": 1.0, "b": 0.3},
        {"k1": 0.8, "b": 0.4},
    ]

    systems, names = [], []
    for cfg in bm25_grid:
        k, b = cfg["k1"], cfg["b"]
        key = f"bm25_k{k}_b{b}_rm3_d10_t15_l0p5".replace(".", "p")
        retr = pt.terrier.Retriever(
            _bm25_index_ref, wmodel="BM25",
            controls={"bm25.k_1": k, "bm25.b": b})
        rm3_res = get_cached(key, lambda r=retr: (
            _get_text_pipeline(r)
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
            >> r
        ).transform(_topics), logger)
        systems.append(_fuse(sp, rm3_res, 1.0, 20.0))
        names.append(f"BM25 (k1={k}, b={b}) + RM3")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp16_bm25_grid_rm3", output_dir, logger)
    return results


@register_experiment(17, "SPLADE Weight Multipliers")
def exp17_splade_weights(logger, output_dir):
    logger.info("Experiment 17: SPLADE Weight Multipliers")
    sp = splade_results(logger)
    rm3 = bm25_rm3_results(logger)

    splade_weights = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    systems = [_fuse(sp, rm3, sw, 20.0) for sw in splade_weights]
    names = [f"Hybrid (SPLADE*{sw} + 20*BM25+RM3)" for sw in splade_weights]

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp17_splade_weights", output_dir, logger)
    return results


@register_experiment(18, "Double Query Expansion (RM3 + Bo1)")
def exp18_double_expansion(logger, output_dir):
    logger.info("Experiment 18: Double Query Expansion (RM3 + Bo1)")
    sp = splade_results(logger)
    rm3 = bm25_rm3_results(logger)
    bm25t = _make_bm25_tuned()

    # RM3 then Bo1
    rm3_then_bo1 = get_cached("bm25_tuned_rm3_then_bo1", lambda: (
        _get_text_pipeline(bm25t)
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25t
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row["title"] or "") + " " + (row["body"] or ""))
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=3, fb_terms=5)
        >> bm25t
    ).transform(_topics), logger)

    # Bo1 then RM3
    bo1_then_rm3 = get_cached("bm25_tuned_bo1_then_rm3", lambda: (
        _get_text_pipeline(bm25t)
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=5, fb_terms=10)
        >> bm25t
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row["title"] or "") + " " + (row["body"] or ""))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10, fb_lambda=0.5)
        >> bm25t
    ).transform(_topics), logger)

    results = pt.Experiment(
        [_fuse(sp, rm3, 1.0, 20.0),
         _fuse(sp, rm3_then_bo1, 1.0, 20.0),
         _fuse(sp, bo1_then_rm3, 1.0, 20.0)],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (RM3 only)", "Hybrid (RM3 → Bo1)", "Hybrid (Bo1 → RM3)"],
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp18_double_expansion", output_dir, logger)
    return results


@register_experiment(19, "Weighted Three-way Fusion")
def exp19_weighted_fusion(logger, output_dir):
    logger.info("Experiment 19: Weighted Three-way Fusion")
    sp = splade_results(logger)
    bm_tuned = bm25_tuned_results(logger)
    rm3 = bm25_rm3_results(logger)

    def three_way(w1, w2, w3):
        ab = _fuse(sp, bm_tuned, w1, w2)
        return _fuse(ab, rm3, 1.0, w3)

    fusion_configs = [
        (1.0, 20.0, 0.0),
        (1.0, 10.0, 10.0),
        (1.0, 5.0,  15.0),
        (1.0, 15.0, 10.0),
        (1.5, 15.0, 10.0),
        (0.8, 20.0,  5.0),
    ]
    systems = [three_way(*cfg) for cfg in fusion_configs]
    names = [f"Fusion (SPLADE*{w1} + BM25*{w2} + BM25_RM3*{w3})"
             for w1, w2, w3 in fusion_configs]

    # Three-way RRF variants
    for k in [30, 60, 100]:
        systems.append(pta.fusion.rr_fusion(sp, bm_tuned, rm3, k=k))
        names.append(f"Three-way RRF (k={k})")

    results = pt.Experiment(
        systems, _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=names,
    )
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp19_weighted_fusion", output_dir, logger)
    return results


@register_experiment(20, "Best Overall Model Evaluation")
def exp20_best_overall(logger, output_dir):
    logger.info("Experiment 20: Best Overall Model Evaluation")
    logger.info("=" * 60)

    sp = splade_results(logger)
    bm = bm25_results(logger)
    rm3 = bm25_rm3_results(logger)
    best_model = _fuse(sp, rm3, 1.0, 20.0)

    results = pt.Experiment(
        [sp, bm, best_model],
        _topics, _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE Baseline", "BM25 Baseline", "Best Overall"],
    )
    logger.info(f"\n{results.to_string()}")

    map_col = next(c for c in results.columns if c.startswith("AP") or c == "map")
    splade_map = results.loc[results["name"] == "SPLADE Baseline", map_col].values[0]
    bm25_map   = results.loc[results["name"] == "BM25 Baseline",   map_col].values[0]
    best_map   = results.loc[results["name"] == "Best Overall",    map_col].values[0]
    logger.info(f"MAP improvement over SPLADE: +{(best_map - splade_map) / splade_map * 100:.2f}%")
    logger.info(f"MAP improvement over BM25:   +{(best_map - bm25_map)   / bm25_map   * 100:.2f}%")

    save_results(results, "exp20_best_overall", output_dir, logger)
    return results

#endregion

def parse_experiments(exp_args: List[str]) -> List[int]:
    experiments = []
    for arg in exp_args:
        if "-" in arg:
            start, end = map(int, arg.split("-"))
            experiments.extend(range(start, end + 1))
        else:
            experiments.append(int(arg))
    return sorted(set(experiments))


def list_experiments():
    print("\nAvailable experiments:")
    print("-" * 60)
    for exp_id in sorted(EXPERIMENTS.keys()):
        print(f"  {exp_id:2d}. {EXPERIMENTS[exp_id]['name']}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run IR experiments on SPLADE + BM25 hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp", nargs="*",
                        help="Experiment IDs (e.g. 1 2 3 or 1-5)")
    parser.add_argument("--list", action="store_true",
                        help="List all available experiments")
    parser.add_argument("--output-dir", default="./experiment_results",
                        help="Directory for output CSV files")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"],
                        help="Device (default: auto-detect)")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing indices and cache")
    parser.add_argument("--dataset", default="robust04",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to use (default: robust04)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached retrieval results before running")

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    output_dir = os.path.abspath(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("IR EXPERIMENTS RUNNER (optimised)")
    logger.info("=" * 60)

    device = args.device or detect_device()
    data_dir = os.path.abspath(args.data_dir)

    try:
        setup_environment(logger, device, data_dir, dataset_name=args.dataset)
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        raise

    # Optionally wipe cache
    if args.clear_cache:
        import shutil
        logger.warning(f"Clearing retrieval cache at {_cache_dir}")
        shutil.rmtree(_cache_dir, ignore_errors=True)
        os.makedirs(_cache_dir, exist_ok=True)

    exp_ids = parse_experiments(args.exp) if args.exp else sorted(EXPERIMENTS.keys())
    invalid = [e for e in exp_ids if e not in EXPERIMENTS]
    if invalid:
        logger.error(f"Invalid experiment IDs: {invalid}")
        return

    logger.info(f"Running experiments: {exp_ids}")

    all_results = {}
    for exp_id in exp_ids:
        info = EXPERIMENTS[exp_id]
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT {exp_id}: {info['name']}")
        logger.info("=" * 60)
        try:
            all_results[exp_id] = info["func"](logger, output_dir)
            logger.info(f"Experiment {exp_id} completed successfully")
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Experiments run: {list(all_results.keys())}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
