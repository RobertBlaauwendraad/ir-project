#!/usr/bin/env python3
"""
Build indices for IR experiments.

This script creates the BM25 and SPLADE indices needed for running experiments.
It should be run once before running experiments, especially on a cluster where
indices need to be built in a specific data directory.

Usage:
    python build_indices.py                           # Build all indices in ./data
    python build_indices.py --data-dir /path/to/data  # Custom data directory
    python build_indices.py --bm25-only               # Build only BM25 index
    python build_indices.py --splade-only             # Build only SPLADE index
    python build_indices.py --force                   # Rebuild existing indices
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pyt_splade
import pyterrier as pt
import torch


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def detect_device():
    """Detect the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def custom_corpus_iter(dataset):
    """Create custom corpus iterator combining title and body."""
    for doc in dataset.get_corpus_iter():
        yield {
            'docno': doc['docno'],
            'text': (doc.get('title', '') + ' ' + doc.get('body', '')).strip()
        }


def build_bm25_index(data_dir: str, dataset, logger: logging.Logger, force: bool = False):
    """Build BM25 index for the dataset."""
    index_dir = os.path.join(data_dir, "robust04_bm25_index")
    
    if os.path.exists(index_dir) and not force:
        logger.info(f"BM25 index already exists at {index_dir}")
        logger.info("Use --force to rebuild")
        return index_dir
    
    if os.path.exists(index_dir) and force:
        logger.info(f"Removing existing BM25 index at {index_dir}")
        import shutil
        shutil.rmtree(index_dir)
    
    logger.info(f"Building BM25 index at {index_dir}...")
    logger.info("This may take a while...")
    
    start_time = datetime.now()
    
    indexer = pt.IterDictIndexer(index_dir)
    index_ref = indexer.index(custom_corpus_iter(dataset))
    
    elapsed = datetime.now() - start_time
    logger.info(f"BM25 index built successfully in {elapsed}")
    logger.info(f"Index location: {index_dir}")
    
    return index_dir


def build_splade_index(data_dir: str, dataset, logger: logging.Logger, device: str, force: bool = False):
    """Build SPLADE index for the dataset."""
    index_dir = os.path.join(data_dir, "robust04_splade_index")
    
    if os.path.exists(index_dir) and not force:
        logger.info(f"SPLADE index already exists at {index_dir}")
        logger.info("Use --force to rebuild")
        return index_dir
    
    if os.path.exists(index_dir) and force:
        logger.info(f"Removing existing SPLADE index at {index_dir}")
        import shutil
        shutil.rmtree(index_dir)
    
    logger.info(f"Building SPLADE index at {index_dir}...")
    logger.info("This may take a while (longer than BM25)...")
    
    # Initialize SPLADE model
    # Check for local model path first (for cluster without internet access)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(script_dir, "models", "splade-cocondenser-ensembledistil")
    hf_model_name = "naver/splade-cocondenser-ensembledistil"
    
    if os.path.isdir(local_model_path):
        splade_model = local_model_path
        logger.info(f"Using local SPLADE model: {local_model_path}")
    else:
        splade_model = hf_model_name
        logger.info(f"Using Hugging Face SPLADE model: {hf_model_name}")
    
    logger.info(f"Initializing SPLADE on device: {device}")
    splade = pyt_splade.Splade(
        model=splade_model,
        device=device,
        max_length=256
    )
    
    start_time = datetime.now()
    
    # Build SPLADE index using doc encoder pipeline
    splade_indexer = splade.doc_encoder() >> pt.IterDictIndexer(index_dir)
    index_ref = splade_indexer.index(custom_corpus_iter(dataset))
    
    elapsed = datetime.now() - start_time
    logger.info(f"SPLADE index built successfully in {elapsed}")
    logger.info(f"Index location: {index_dir}")
    
    return index_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build indices for IR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for storing indices (default: ./data)"
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Build only BM25 index"
    )
    parser.add_argument(
        "--splade-only",
        action="store_true",
        help="Build only SPLADE index"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if indices exist"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SPLADE (cuda/mps/cpu, auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Determine what to build
    build_bm25 = not args.splade_only
    build_splade = not args.bm25_only
    
    logger.info("=" * 60)
    logger.info("IR Project - Index Builder")
    logger.info("=" * 60)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Data directory: {data_dir}")
    
    # Detect device
    device = args.device or detect_device()
    logger.info(f"Using device: {device}")
    
    # Initialize PyTerrier
    logger.info("Initializing PyTerrier...")
    if not pt.started():
        pt.init()
    
    # Load dataset
    logger.info("Loading Robust04 dataset...")
    dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
    
    # Build indices
    if build_bm25:
        logger.info("-" * 40)
        bm25_dir = build_bm25_index(data_dir, dataset, logger, force=args.force)
    
    if build_splade:
        logger.info("-" * 40)
        splade_dir = build_splade_index(data_dir, dataset, logger, device, force=args.force)
    
    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nSummary:")
    if build_bm25:
        logger.info(f"  BM25 index:   {os.path.join(data_dir, 'robust04_bm25_index')}")
    if build_splade:
        logger.info(f"  SPLADE index: {os.path.join(data_dir, 'robust04_splade_index')}")
    
    logger.info("\nTo use these indices in experiments, update the index paths or set:")
    logger.info(f"  export IR_DATA_DIR={data_dir}")


if __name__ == "__main__":
    main()
