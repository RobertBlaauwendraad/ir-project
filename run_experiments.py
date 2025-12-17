#!/usr/bin/env python3
"""
IR Experiments Runner

Converts notebook experiments into a cluster-ready Python script with proper logging.
Runs experiments on SPLADE + BM25 hybrid retrieval with various configurations.

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --exp 1 2 3       # Run specific experiments
    python run_experiments.py --exp 1-5         # Run experiment range
    python run_experiments.py --list            # List all experiments
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import pyt_splade
import pyterrier as pt
import pyterrier_alpha as pta
import torch
from pyterrier.measures import MAP, nDCG, Recall, P

warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


# Global variables for shared resources
_dataset = None
_splade = None
_splade_retr = None
_bm25_retr = None
_bm25_index_ref = None
_splade_index_ref = None
_topics = None
_qrels = None


def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiments_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("ir_experiments")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def save_results(results: pd.DataFrame, name: str, output_dir: str, logger: logging.Logger):
    """Save experiment results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{name}_{timestamp}.csv")
    results.to_csv(filename, index=False)
    logger.info(f"Results saved to: {filename}")


def setup_environment(logger: logging.Logger, device: str):
    """Initialize PyTerrier, models, and indices."""
    global _dataset, _splade, _splade_retr, _bm25_retr
    global _bm25_index_ref, _splade_index_ref, _topics, _qrels
    
    logger.info(f"Using device: {device}")
    torch.manual_seed(26)
    
    # Initialize SPLADE
    # Check for local model path first (for cluster without internet access)
    # Falls back to Hugging Face model name if local path doesn't exist
    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "splade-cocondenser-ensembledistil")
    hf_model_name = "naver/splade-cocondenser-ensembledistil"
    
    if os.path.isdir(local_model_path):
        splade_model = local_model_path
        logger.info(f"Initializing SPLADE model from local path: {local_model_path}")
    else:
        splade_model = hf_model_name
        logger.info(f"Initializing SPLADE model from Hugging Face: {hf_model_name}")
    
    _splade = pyt_splade.Splade(
        model=splade_model,
        device=device,
        max_length=256
    )
    
    # Load dataset
    logger.info("Loading Robust04 dataset...")
    _dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
    
    # Load indices
    bm25_index_dir = os.path.abspath("./robust04_bm25_index")
    splade_index_dir = os.path.abspath("./robust04_splade_index")
    
    logger.info("Loading BM25 index...")
    _bm25_index_ref = pt.IndexFactory.of(bm25_index_dir)
    
    logger.info("Loading SPLADE index...")
    _splade_index_ref = pt.IndexFactory.of(splade_index_dir)
    
    # Create retrievers
    logger.info("Creating retrievers...")
    _splade_retr = _splade.query_encoder() >> pt.terrier.Retriever(_splade_index_ref, wmodel="Tf")
    _bm25_retr = pt.terrier.Retriever(_bm25_index_ref, wmodel="BM25")
    
    # Prepare topics and qrels
    _topics = _dataset.get_topics()
    _topics["query"] = _topics["title"]
    _qrels = _dataset.get_qrels()
    
    logger.info("Environment setup complete!")


# Experiment registry
EXPERIMENTS = {}


def register_experiment(exp_id: int, name: str):
    """Decorator to register an experiment function."""
    def decorator(func):
        EXPERIMENTS[exp_id] = {"name": name, "func": func}
        return func
    return decorator


# =============================================================================
# EXPERIMENTS 1-5
# =============================================================================

@register_experiment(1, "Baseline SPLADE")
def exp1_baseline_splade(logger: logging.Logger, output_dir: str):
    """Experiment 1: Baseline SPLADE evaluation."""
    logger.info("Running Experiment 1: Baseline SPLADE")
    
    results = pt.Experiment(
        [_splade_retr],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE Baseline"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp1_baseline_splade", output_dir, logger)
    return results


@register_experiment(2, "Hybrid Retrieval with Different BM25 Weights")
def exp2_hybrid_retrieval(logger: logging.Logger, output_dir: str):
    """Experiment 2: Improved SPLADE with Hybrid Retrieval."""
    logger.info("Running Experiment 2: Hybrid Retrieval with Different BM25 Weights")
    
    hybrid_05 = _splade_retr + (0.05 * _bm25_retr)
    hybrid_10 = _splade_retr + (0.10 * _bm25_retr)
    hybrid_20 = _splade_retr + (0.20 * _bm25_retr)
    
    results = pt.Experiment(
        [_splade_retr, _bm25_retr, hybrid_05, hybrid_10, hybrid_20],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE Only", "BM25 Only", "Hybrid (w=0.05)", "Hybrid (w=0.1)", "Hybrid (w=0.2)"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp2_hybrid_retrieval", output_dir, logger)
    return results


@register_experiment(3, "Reciprocal Rank Fusion")
def exp3_rrf(logger: logging.Logger, output_dir: str):
    """Experiment 3: Reciprocal Rank Fusion."""
    logger.info("Running Experiment 3: Reciprocal Rank Fusion")
    
    # Precompute retrieval results
    splade_res = _splade_retr.transform(_topics)
    bm25_res = _bm25_retr.transform(_topics)
    
    hybrid_20 = _splade_retr + (0.20 * _bm25_retr)
    rrf = pta.fusion.rr_fusion(splade_res, bm25_res, k=60)
    
    results = pt.Experiment(
        [splade_res, bm25_res, hybrid_20, rrf],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "BM25", "Hybrid (w=0.2)", "RRF"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp3_rrf", output_dir, logger)
    return results


@register_experiment(4, "Hybrid Weight Optimization")
def exp4_weight_optimization(logger: logging.Logger, output_dir: str):
    """Experiment 4: Hybrid Retrieval Weight Optimization."""
    logger.info("Running Experiment 4: Hybrid Weight Optimization")
    
    # Fine-grained weights 0.0 to 1.0
    logger.info("Testing weights 0.0 to 1.0...")
    weights = np.arange(0.0, 1.1, 0.1)
    results_list = []
    
    for w in weights:
        hybrid = _splade_retr + (w * _bm25_retr)
        exp = pt.Experiment(
            [hybrid],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f'Hybrid (w={w:.1f})']
        )
        results_list.append(exp)
    
    results_fine = pd.concat(results_list)
    logger.info(f"\nFine weights (0.0-1.0):\n{results_fine.to_string()}")
    
    # Medium weights
    logger.info("Testing weights 1 to 20...")
    weights = [1, 2, 3, 5, 8, 10, 15, 20]
    results_list = []
    
    for w in weights:
        hybrid = _splade_retr + (w * _bm25_retr)
        exp = pt.Experiment(
            [hybrid],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f'Hybrid (w={w:.1f})']
        )
        results_list.append(exp)
    
    results_medium = pd.concat(results_list)
    logger.info(f"\nMedium weights (1-20):\n{results_medium.to_string()}")
    
    # Large weights
    logger.info("Testing weights 20 to 100...")
    weights = [20, 30, 40, 50, 75, 100]
    results_list = []
    
    for w in weights:
        hybrid = _splade_retr + (w * _bm25_retr)
        exp = pt.Experiment(
            [hybrid],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f'Hybrid (w={w:.1f})']
        )
        results_list.append(exp)
    
    results_large = pd.concat(results_list)
    logger.info(f"\nLarge weights (20-100):\n{results_large.to_string()}")
    
    # Combine all results
    all_results = pd.concat([results_fine, results_medium, results_large])
    save_results(all_results, "exp4_weight_optimization", output_dir, logger)
    return all_results


@register_experiment(5, "Query Expansion Models (Bo1 and RM3)")
def exp5_expansion_models(logger: logging.Logger, output_dir: str):
    """Experiment 5: Expansion Models (Bo1 and RM3)."""
    logger.info("Running Experiment 5: Query Expansion Models")
    
    hybrid_20 = _splade_retr + (20 * _bm25_retr)
    
    # Bo1 Query Expansion
    logger.info("Testing Bo1 query expansion...")
    bm25_bo1 = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=3, fb_terms=10)
        >> _bm25_retr
    )
    
    upgraded_hybrid = _splade_retr + (20.0 * bm25_bo1)
    
    results_bo1 = pt.Experiment(
        [_splade_retr, hybrid_20, upgraded_hybrid],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "Hybrid (w=20)", "Upgraded Hybrid (w=20 + Bo1)"]
    )
    logger.info(f"\nBo1 results:\n{results_bo1.to_string()}")
    
    # RM3 Query Expansion
    logger.info("Testing RM3 query expansion...")
    bm25_rm3 = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10)
        >> _bm25_retr
    )
    
    rm3_hybrid = _splade_retr + (20.0 * bm25_rm3)
    
    results_rm3 = pt.Experiment(
        [rm3_hybrid],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (w=20 + RM3)"]
    )
    logger.info(f"\nRM3 results:\n{results_rm3.to_string()}")
    
    all_results = pd.concat([results_bo1, results_rm3])
    save_results(all_results, "exp5_expansion_models", output_dir, logger)
    return all_results


# =============================================================================
# EXPERIMENTS 6-10
# =============================================================================

@register_experiment(6, "RM3 Lambda Parameter Optimization")
def exp6_rm3_lambda(logger: logging.Logger, output_dir: str):
    """Experiment 6: Expansion Model Parameter Optimization."""
    logger.info("Running Experiment 6: RM3 Lambda Parameter Optimization")
    
    lambdas = [0.3, 0.4, 0.5, 0.6, 0.7]
    results_list = []
    
    for lam in lambdas:
        logger.info(f"Testing lambda={lam}...")
        rm3_lambda = (
            _bm25_retr
            >> pt.text.get_text(_dataset, ["title", "body"])
            >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10, fb_lambda=lam)
            >> _bm25_retr
        )
        
        rm3_final = _splade_retr + (20.0 * rm3_lambda)
        
        results_list.append(
            pt.Experiment(
                [rm3_final],
                _topics,
                _qrels,
                eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
                names=[f"RM3 (lambda={lam})"]
            )
        )
    
    results = pd.concat(results_list)
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp6_rm3_lambda", output_dir, logger)
    return results


@register_experiment(7, "BM25 Parameter Tuning (k1 and b)")
def exp7_bm25_tuning(logger: logging.Logger, output_dir: str):
    """Experiment 7: BM25 Parameter Tuning."""
    logger.info("Running Experiment 7: BM25 Parameter Tuning")
    
    bm25_configs = [
        {"k1": 1.2, "b": 0.75},  # Default
        {"k1": 0.9, "b": 0.4},   # Lower values
        {"k1": 1.5, "b": 0.75},  # Higher k1
        {"k1": 1.2, "b": 0.5},   # Lower b
        {"k1": 2.0, "b": 0.75},  # High k1
        {"k1": 1.2, "b": 0.9},   # High b
    ]
    
    results_list = []
    for config in bm25_configs:
        logger.info(f"Testing k1={config['k1']}, b={config['b']}...")
        bm25_tuned = pt.terrier.Retriever(
            _bm25_index_ref,
            wmodel="BM25",
            controls={"bm25.k_1": config["k1"], "bm25.b": config["b"]}
        )
        
        exp = pt.Experiment(
            [bm25_tuned],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"BM25 (k1={config['k1']}, b={config['b']})"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp7_bm25_tuning", output_dir, logger)
    return results


@register_experiment(8, "RRF with Different k Values")
def exp8_rrf_k_values(logger: logging.Logger, output_dir: str):
    """Experiment 8: RRF with Different k Values."""
    logger.info("Running Experiment 8: RRF with Different k Values")
    
    rrf_k_values = [10, 30, 60, 100, 200]
    
    # Precompute retrieval results
    splade_res = _splade_retr.transform(_topics)
    bm25_res = _bm25_retr.transform(_topics)
    
    results_list = []
    for k in rrf_k_values:
        logger.info(f"Testing k={k}...")
        rrf_k = pta.fusion.rr_fusion(splade_res, bm25_res, k=k)
        
        exp = pt.Experiment(
            [rrf_k],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"RRF (k={k})"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp8_rrf_k_values", output_dir, logger)
    return results


@register_experiment(9, "MonoT5 Cross-Encoder Re-ranking")
def exp9_monot5_reranking(logger: logging.Logger, output_dir: str):
    """Experiment 9: Cross-Encoder Re-ranking with MonoT5."""
    logger.info("Running Experiment 9: MonoT5 Cross-Encoder Re-ranking")
    
    try:
        from pyterrier_t5 import MonoT5ReRanker
    except ImportError:
        logger.warning("pyterrier_t5 not installed, skipping MonoT5 experiment")
        return pd.DataFrame()
    
    # Initialize MonoT5 re-ranker
    logger.info("Initializing MonoT5 re-ranker...")
    mono_t5 = MonoT5ReRanker(model="castorini/monot5-base-msmarco", batch_size=16)
    
    # SPLADE + MonoT5 re-ranking
    logger.info("Building SPLADE + MonoT5 pipeline...")
    splade_monot5 = (
        _splade_retr % 100  # Get top 100 from SPLADE
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> mono_t5
    )
    
    # Hybrid + MonoT5 re-ranking
    logger.info("Building Hybrid + MonoT5 pipeline...")
    hybrid_monot5 = (
        (_splade_retr + (20 * _bm25_retr)) % 100  # Get top 100 from hybrid
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> mono_t5
    )
    
    results = pt.Experiment(
        [_splade_retr, splade_monot5, hybrid_monot5],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "SPLADE + MonoT5", "Hybrid + MonoT5"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp9_monot5_reranking", output_dir, logger)
    return results


@register_experiment(10, "Combined Query Expansion with Hybrid")
def exp10_combined_expansion(logger: logging.Logger, output_dir: str):
    """Experiment 10: Combined Query Expansion with Hybrid."""
    logger.info("Running Experiment 10: Combined Query Expansion with Hybrid")
    
    hybrid_20 = _splade_retr + (20 * _bm25_retr)
    
    # Bo1 with optimized parameters
    logger.info("Building Bo1 optimized pipeline...")
    bm25_bo1_opt = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=5, fb_terms=15)
        >> _bm25_retr
    )
    
    # RM3 with optimized parameters
    logger.info("Building RM3 optimized pipeline...")
    bm25_rm3_opt = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> _bm25_retr
    )
    
    # Hybrid with Bo1
    hybrid_bo1 = _splade_retr + (20.0 * bm25_bo1_opt)
    
    # Hybrid with RM3
    hybrid_rm3 = _splade_retr + (20.0 * bm25_rm3_opt)
    
    results = pt.Experiment(
        [hybrid_20, hybrid_bo1, hybrid_rm3],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (w=20)", "Hybrid + Bo1", "Hybrid + RM3"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp10_combined_expansion", output_dir, logger)
    return results


# =============================================================================
# EXPERIMENTS 11-15
# =============================================================================

@register_experiment(11, "Best Hybrid Weight with Query Expansion")
def exp11_best_weight_qe(logger: logging.Logger, output_dir: str):
    """Experiment 11: Best Hybrid Weight with Query Expansion."""
    logger.info("Running Experiment 11: Best Hybrid Weight with Query Expansion")
    
    # RM3 with optimized parameters
    bm25_rm3_opt = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> _bm25_retr
    )
    
    weights_qe = [10, 15, 20, 25, 30, 40]
    results_list = []
    
    for w in weights_qe:
        logger.info(f"Testing weight={w}...")
        hybrid_rm3_w = _splade_retr + (w * bm25_rm3_opt)
        
        exp = pt.Experiment(
            [hybrid_rm3_w],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"Hybrid+RM3 (w={w})"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp11_best_weight_qe", output_dir, logger)
    return results


@register_experiment(12, "Three-way Fusion")
def exp12_three_way_fusion(logger: logging.Logger, output_dir: str):
    """Experiment 12: Three-way Fusion."""
    logger.info("Running Experiment 12: Three-way Fusion")
    
    # Precompute results
    splade_res = _splade_retr.transform(_topics)
    bm25_res = _bm25_retr.transform(_topics)
    
    # RM3 with optimized parameters
    bm25_rm3_opt = (
        _bm25_retr
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> _bm25_retr
    )
    bm25_rm3_res = bm25_rm3_opt.transform(_topics)
    
    hybrid_20 = _splade_retr + (20 * _bm25_retr)
    three_way_rrf = pta.fusion.rr_fusion(splade_res, bm25_res, bm25_rm3_res, k=60)
    
    results = pt.Experiment(
        [_splade_retr, hybrid_20, three_way_rrf],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE", "Hybrid (w=20)", "Three-way RRF"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp12_three_way_fusion", output_dir, logger)
    return results


@register_experiment(13, "Tuned BM25 in Hybrid")
def exp13_tuned_bm25_hybrid(logger: logging.Logger, output_dir: str):
    """Experiment 13: Tuned BM25 in Hybrid."""
    logger.info("Running Experiment 13: Tuned BM25 in Hybrid")
    
    hybrid_20 = _splade_retr + (20 * _bm25_retr)
    
    # Tuned BM25 (best params from experiment 7)
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    # Hybrid with tuned BM25
    hybrid_tuned = _splade_retr + (20 * bm25_tuned_best)
    
    # Hybrid with tuned BM25 + RM3
    bm25_tuned_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    
    hybrid_tuned_rm3 = _splade_retr + (20 * bm25_tuned_rm3)
    
    results = pt.Experiment(
        [hybrid_20, hybrid_tuned, hybrid_tuned_rm3],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (default BM25)", "Hybrid (tuned BM25)", "Hybrid (tuned BM25 + RM3)"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp13_tuned_bm25_hybrid", output_dir, logger)
    return results


@register_experiment(14, "Fine-tune Weight for Tuned BM25 + RM3")
def exp14_fine_tune_weight(logger: logging.Logger, output_dir: str):
    """Experiment 14: Fine-tune Hybrid Weight for Tuned BM25 + RM3."""
    logger.info("Running Experiment 14: Fine-tune Weight for Tuned BM25 + RM3")
    
    # Tuned BM25 with RM3
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    bm25_tuned_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    
    weights_fine = [15, 18, 20, 22, 25, 28, 30, 35, 40]
    results_list = []
    
    for w in weights_fine:
        logger.info(f"Testing weight={w}...")
        hybrid_tuned_rm3_w = _splade_retr + (w * bm25_tuned_rm3)
        
        exp = pt.Experiment(
            [hybrid_tuned_rm3_w],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"Hybrid (tuned BM25+RM3, w={w})"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\nFine-tuned weight results:\n{results.to_string()}")
    save_results(results, "exp14_fine_tune_weight", output_dir, logger)
    return results


@register_experiment(15, "Aggressive RM3 Parameter Tuning")
def exp15_aggressive_rm3(logger: logging.Logger, output_dir: str):
    """Experiment 15: Aggressive RM3 Parameter Tuning."""
    logger.info("Running Experiment 15: Aggressive RM3 Parameter Tuning")
    
    # Tuned BM25
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    rm3_configs = [
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.5},  # Current best
        {"fb_docs": 15, "fb_terms": 20, "fb_lambda": 0.5},  # More docs and terms
        {"fb_docs": 10, "fb_terms": 25, "fb_lambda": 0.5},  # More terms
        {"fb_docs": 20, "fb_terms": 15, "fb_lambda": 0.5},  # More docs
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.4},  # Lower lambda
        {"fb_docs": 10, "fb_terms": 15, "fb_lambda": 0.6},  # Higher lambda
        {"fb_docs": 15, "fb_terms": 20, "fb_lambda": 0.4},  # Combined
        {"fb_docs": 5, "fb_terms": 10, "fb_lambda": 0.5},   # Lighter expansion
        {"fb_docs": 10, "fb_terms": 30, "fb_lambda": 0.5},  # Many more terms
    ]
    
    results_list = []
    for config in rm3_configs:
        logger.info(f"Testing docs={config['fb_docs']}, terms={config['fb_terms']}, lambda={config['fb_lambda']}...")
        bm25_rm3_config = (
            bm25_tuned_best
            >> pt.text.get_text(_dataset, ["title", "body"])
            >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
            >> pt.rewrite.RM3(_bm25_index_ref,
                             fb_docs=config["fb_docs"],
                             fb_terms=config["fb_terms"],
                             fb_lambda=config["fb_lambda"])
            >> bm25_tuned_best
        )
        
        hybrid_rm3_config = _splade_retr + (20 * bm25_rm3_config)
        
        exp = pt.Experiment(
            [hybrid_rm3_config],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"RM3 (docs={config['fb_docs']}, terms={config['fb_terms']}, λ={config['fb_lambda']})"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\nRM3 parameter tuning results:\n{results.to_string()}")
    save_results(results, "exp15_aggressive_rm3", output_dir, logger)
    return results


# =============================================================================
# EXPERIMENTS 16-19
# =============================================================================

@register_experiment(16, "Grid Search BM25 Parameters with RM3")
def exp16_bm25_grid_rm3(logger: logging.Logger, output_dir: str):
    """Experiment 16: Grid Search BM25 Parameters with RM3."""
    logger.info("Running Experiment 16: Grid Search BM25 Parameters with RM3")
    
    bm25_grid = [
        {"k1": 0.9, "b": 0.4},   # Current best
        {"k1": 0.8, "b": 0.3},   # Lower values
        {"k1": 0.9, "b": 0.3},   # Lower b
        {"k1": 1.0, "b": 0.4},   # Slightly higher k1
        {"k1": 0.7, "b": 0.4},   # Lower k1
        {"k1": 0.9, "b": 0.5},   # Higher b
        {"k1": 1.0, "b": 0.3},   # Higher k1, lower b
        {"k1": 0.8, "b": 0.4},   # Slightly lower k1
    ]
    
    results_list = []
    for config in bm25_grid:
        logger.info(f"Testing k1={config['k1']}, b={config['b']}...")
        bm25_grid_retr = pt.terrier.Retriever(
            _bm25_index_ref,
            wmodel="BM25",
            controls={"bm25.k_1": config["k1"], "bm25.b": config["b"]}
        )
        
        bm25_grid_rm3 = (
            bm25_grid_retr
            >> pt.text.get_text(_dataset, ["title", "body"])
            >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
            >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
            >> bm25_grid_retr
        )
        
        hybrid_grid = _splade_retr + (20 * bm25_grid_rm3)
        
        exp = pt.Experiment(
            [hybrid_grid],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"BM25 (k1={config['k1']}, b={config['b']}) + RM3"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\nBM25 grid search with RM3 results:\n{results.to_string()}")
    save_results(results, "exp16_bm25_grid_rm3", output_dir, logger)
    return results


@register_experiment(17, "SPLADE Weight Multipliers")
def exp17_splade_weights(logger: logging.Logger, output_dir: str):
    """Experiment 17: SPLADE Weight Multipliers."""
    logger.info("Running Experiment 17: SPLADE Weight Multipliers")
    
    # Tuned BM25 with RM3
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    bm25_tuned_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    
    splade_weights = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results_list = []
    
    for sw in splade_weights:
        logger.info(f"Testing SPLADE weight={sw}...")
        hybrid_splade_w = (sw * _splade_retr) + (20 * bm25_tuned_rm3)
        
        exp = pt.Experiment(
            [hybrid_splade_w],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"Hybrid (SPLADE*{sw} + 20*BM25+RM3)"]
        )
        results_list.append(exp)
    
    results = pd.concat(results_list)
    logger.info(f"\nSPLADE weight multiplier results:\n{results.to_string()}")
    save_results(results, "exp17_splade_weights", output_dir, logger)
    return results


@register_experiment(18, "Double Query Expansion (RM3 + Bo1)")
def exp18_double_expansion(logger: logging.Logger, output_dir: str):
    """Experiment 18: Double Query Expansion (RM3 + Bo1)."""
    logger.info("Running Experiment 18: Double Query Expansion (RM3 + Bo1)")
    
    # Tuned BM25
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    # Single RM3 baseline
    bm25_tuned_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    hybrid_tuned_rm3 = _splade_retr + (20 * bm25_tuned_rm3)
    
    # RM3 then Bo1
    logger.info("Building RM3 → Bo1 pipeline...")
    bm25_double_expansion = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=3, fb_terms=5)
        >> bm25_tuned_best
    )
    hybrid_double_exp = _splade_retr + (20 * bm25_double_expansion)
    
    # Bo1 then RM3
    logger.info("Building Bo1 → RM3 pipeline...")
    bm25_bo1_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.Bo1QueryExpansion(_bm25_index_ref, fb_docs=5, fb_terms=10)
        >> bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=10, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    hybrid_bo1_rm3 = _splade_retr + (20 * bm25_bo1_rm3)
    
    results = pt.Experiment(
        [hybrid_tuned_rm3, hybrid_double_exp, hybrid_bo1_rm3],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["Hybrid (RM3 only)", "Hybrid (RM3 → Bo1)", "Hybrid (Bo1 → RM3)"]
    )
    
    logger.info(f"\n{results.to_string()}")
    save_results(results, "exp18_double_expansion", output_dir, logger)
    return results


@register_experiment(19, "Weighted Three-way Fusion")
def exp19_weighted_fusion(logger: logging.Logger, output_dir: str):
    """Experiment 19: Weighted Three-way Fusion."""
    logger.info("Running Experiment 19: Weighted Three-way Fusion")
    
    # Tuned BM25
    bm25_tuned_best = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    bm25_tuned_rm3 = (
        bm25_tuned_best
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> bm25_tuned_best
    )
    
    # Compute results
    splade_res_fresh = _splade_retr.transform(_topics)
    bm25_tuned_res = bm25_tuned_best.transform(_topics)
    bm25_tuned_rm3_res = bm25_tuned_rm3.transform(_topics)
    
    def weighted_fusion(res1, res2, res3, w1, w2, w3):
        """Combine three result sets with weights."""
        merged = res1.copy()
        merged['score'] = w1 * merged['score']
        
        for df, w in [(res2, w2), (res3, w3)]:
            temp = df[['qid', 'docno', 'score']].copy()
            temp['score'] = w * temp['score']
            merged = pd.merge(merged[['qid', 'docno', 'score']], temp,
                             on=['qid', 'docno'], how='outer', suffixes=('', '_r'))
            merged['score'] = merged['score'].fillna(0) + merged['score_r'].fillna(0)
            merged = merged[['qid', 'docno', 'score']]
        
        merged['rank'] = merged.groupby('qid')['score'].rank(ascending=False, method='first').astype(int)
        return merged.sort_values(['qid', 'rank'])
    
    # Test different weight combinations
    fusion_configs = [
        (1.0, 20.0, 0.0),    # Current: SPLADE + 20*BM25_RM3
        (1.0, 10.0, 10.0),   # Equal BM25 and BM25+RM3
        (1.0, 5.0, 15.0),    # More weight on RM3
        (1.0, 15.0, 10.0),   # Balanced
        (1.5, 15.0, 10.0),   # Higher SPLADE
        (0.8, 20.0, 5.0),    # Add some plain BM25
    ]
    
    results_list = []
    for w1, w2, w3 in fusion_configs:
        logger.info(f"Testing fusion weights: SPLADE*{w1} + BM25*{w2} + BM25_RM3*{w3}...")
        fused = weighted_fusion(splade_res_fresh, bm25_tuned_res, bm25_tuned_rm3_res, w1, w2, w3)
        
        exp = pt.Experiment(
            [fused],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"Fusion (SPLADE*{w1} + BM25*{w2} + BM25_RM3*{w3})"]
        )
        results_list.append(exp)
    
    fusion_results = pd.concat(results_list)
    logger.info(f"\nWeighted three-way fusion results:\n{fusion_results.to_string()}")
    
    # Also try RRF with different k values for three-way
    logger.info("Testing three-way RRF with different k values...")
    rrf_results_list = []
    for k in [30, 60, 100]:
        logger.info(f"Testing RRF k={k}...")
        rrf_3way = pta.fusion.rr_fusion(splade_res_fresh, bm25_tuned_res, bm25_tuned_rm3_res, k=k)
        
        exp = pt.Experiment(
            [rrf_3way],
            _topics,
            _qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=[f"Three-way RRF (k={k})"]
        )
        rrf_results_list.append(exp)
    
    rrf_results = pd.concat(rrf_results_list)
    logger.info(f"\nThree-way RRF results:\n{rrf_results.to_string()}")
    
    all_results = pd.concat([fusion_results, rrf_results])
    save_results(all_results, "exp19_weighted_fusion", output_dir, logger)
    return all_results


# =============================================================================
# BEST OVERALL MODEL
# =============================================================================

@register_experiment(20, "Best Overall Model Evaluation")
def exp20_best_overall(logger: logging.Logger, output_dir: str):
    """Experiment 20: Best Overall Model Evaluation."""
    logger.info("Running Experiment 20: Best Overall Model Evaluation")
    logger.info("=" * 60)
    logger.info("BEST OVERALL MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Best BM25 (tuned k1=0.9, b=0.4)
    best_bm25 = pt.terrier.Retriever(
        _bm25_index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 0.9, "bm25.b": 0.4}
    )
    
    # BM25 with RM3 query expansion (fb_docs=10, fb_terms=15, fb_lambda=0.5)
    best_bm25_rm3 = (
        best_bm25
        >> pt.text.get_text(_dataset, ["title", "body"])
        >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
        >> pt.rewrite.RM3(_bm25_index_ref, fb_docs=10, fb_terms=15, fb_lambda=0.5)
        >> best_bm25
    )
    
    # Best Overall: SPLADE + 20 * (Tuned BM25 + RM3)
    best_model = _splade_retr + (20 * best_bm25_rm3)
    
    logger.info("\nConfiguration:")
    logger.info("  - First stage: SPLADE (naver/splade-cocondenser-ensembledistil)")
    logger.info("  - Second stage: Tuned BM25 (k1=0.9, b=0.4) + RM3 expansion")
    logger.info("  - RM3 params: fb_docs=10, fb_terms=15, fb_lambda=0.5")
    logger.info("  - Fusion weight: SPLADE + 20 * BM25_RM3")
    
    results = pt.Experiment(
        [_splade_retr, _bm25_retr, best_model],
        _topics,
        _qrels,
        eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
        names=["SPLADE Baseline", "BM25 Baseline", "Best Overall"]
    )
    
    logger.info(f"\n{results.to_string()}")
    
    # Calculate improvements
    logger.info("\nImprovements over baselines:")
    map_col = [col for col in results.columns if col.startswith('AP') or col == 'map'][0]
    splade_map = results.loc[results['name'] == 'SPLADE Baseline', map_col].values[0]
    bm25_map = results.loc[results['name'] == 'BM25 Baseline', map_col].values[0]
    best_map = results.loc[results['name'] == 'Best Overall', map_col].values[0]
    logger.info(f"  MAP improvement over SPLADE: +{((best_map - splade_map) / splade_map * 100):.2f}%")
    logger.info(f"  MAP improvement over BM25: +{((best_map - bm25_map) / bm25_map * 100):.2f}%")
    
    save_results(results, "exp20_best_overall", output_dir, logger)
    return results


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parse_experiments(exp_args: List[str]) -> List[int]:
    """Parse experiment arguments into list of experiment IDs."""
    experiments = []
    for arg in exp_args:
        if '-' in arg:
            # Range: e.g., "1-5"
            start, end = map(int, arg.split('-'))
            experiments.extend(range(start, end + 1))
        else:
            experiments.append(int(arg))
    return sorted(set(experiments))


def list_experiments():
    """Print list of available experiments."""
    print("\nAvailable experiments:")
    print("-" * 60)
    for exp_id in sorted(EXPERIMENTS.keys()):
        print(f"  {exp_id:2d}. {EXPERIMENTS[exp_id]['name']}")
    print("-" * 60)
    print("\nUsage examples:")
    print("  python run_experiments.py                  # Run all experiments")
    print("  python run_experiments.py --exp 1 2 3     # Run experiments 1, 2, 3")
    print("  python run_experiments.py --exp 1-5       # Run experiments 1-5")
    print("  python run_experiments.py --exp 1-5 10    # Run experiments 1-5 and 10")


def main():
    parser = argparse.ArgumentParser(
        description="Run IR experiments on SPLADE + BM25 hybrid retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                    # Run all experiments
  python run_experiments.py --exp 1 2 3       # Run specific experiments
  python run_experiments.py --exp 1-5         # Run experiment range
  python run_experiments.py --exp 1-5 10 20   # Run experiments 1-5, 10, and 20
  python run_experiments.py --list            # List all experiments
        """
    )
    parser.add_argument(
        "--exp", 
        nargs="*", 
        help="Experiment IDs to run (e.g., 1 2 3 or 1-5)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available experiments"
    )
    parser.add_argument(
        "--output-dir", 
        default="./experiment_results",
        help="Directory for output files (default: ./experiment_results)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    # Setup logging
    output_dir = os.path.abspath(args.output_dir)
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("IR EXPERIMENTS RUNNER")
    logger.info("=" * 60)
    
    # Detect device
    device = args.device or detect_device()
    
    # Setup environment
    try:
        setup_environment(logger, device)
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}")
        raise
    
    # Determine which experiments to run
    if args.exp:
        exp_ids = parse_experiments(args.exp)
    else:
        exp_ids = sorted(EXPERIMENTS.keys())
    
    # Validate experiment IDs
    invalid_ids = [eid for eid in exp_ids if eid not in EXPERIMENTS]
    if invalid_ids:
        logger.error(f"Invalid experiment IDs: {invalid_ids}")
        logger.info(f"Valid IDs: {sorted(EXPERIMENTS.keys())}")
        return
    
    logger.info(f"Running experiments: {exp_ids}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run experiments
    all_results = {}
    for exp_id in exp_ids:
        exp_info = EXPERIMENTS[exp_id]
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT {exp_id}: {exp_info['name']}")
        logger.info("=" * 60)
        
        try:
            results = exp_info['func'](logger, output_dir)
            all_results[exp_id] = results
            logger.info(f"Experiment {exp_id} completed successfully")
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENTS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Experiments run: {list(all_results.keys())}")


if __name__ == "__main__":
    main()
