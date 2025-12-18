#!/usr/bin/env python3
"""
Build indices for IR experiments with SHARDING support.

This script creates the BM25 and SPLADE indices needed for running experiments.
It should be run once before running experiments, especially on a cluster where
indices need to be built in a specific data directory.

Supported datasets:
    - robust04: TREC Robust 2004 (disks45/nocr/trec-robust-2004)
    - owi: OWI (owi/dev, owi/subsampled/dev)

Usage:
    python build_indices_sharding.py                           # Build all indices in ./data (robust04)
    python build_indices_sharding.py --dataset owi             # Build indices for OWI dataset
    python build_indices_sharding.py --dataset owi/subsampled  # Build indices for OWI subsampled
    python build_indices_sharding.py --data-dir /path/to/data  # Custom data directory
    python build_indices_sharding.py --bm25-only               # Build only BM25 index
    python build_indices_sharding.py --splade-only             # Build only SPLADE index
    python build_indices_sharding.py --force                   # Rebuild existing indices
    python build_indices_sharding.py --shard-id x     # Run the job with the SHARD id x
    python build_indices_sharding.py --num-shards x   # 
"""

import argparse
import logging
import os
import sys
from datetime import datetime
import shutil

import ir_datasets
import itertools
import pyt_splade
import pyterrier as pt
import torch

import ir_datasets_owi


# Dataset configurations
DATASET_CONFIGS = {
    "robust04": {
        "irds_id": "disks45/nocr/trec-robust-2004",
        "index_prefix": "robust04",
        "text_fields": ["title", "body"],  # Fields to concatenate for indexing
    },
    "owi": {
        "irds_id": "owi/dev",
        "index_prefix": "owi",
        "text_fields": ["title", "main_content"],  # OWI document fields
    },
    "owi/subsampled": {
        "irds_id": "owi/subsampled/dev",
        "index_prefix": "owi_subsampled",
        "text_fields": ["title", "main_content"],
    },
}


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


def custom_corpus_iter(dataset, text_fields=None, shard_id=0, num_shards=1):
    """
    Create custom corpus iterator combining specified text fields.
    
    Args:
        dataset: PyTerrier dataset object
        text_fields: List of field names to concatenate for text. 
                     Defaults to ["title", "body"] for backwards compatibility.
    """
    if text_fields is None:
        text_fields = ["title", "body"]
    
    iterator = dataset.get_corpus_iter()
    
    if num_shards > 1:
        logging.info(f"Sharding enabled: Processing shard {shard_id}/{num_shards} (Step={num_shards})")
        iterator = itertools.islice(iterator, shard_id, None, num_shards)
    
    for doc in iterator:
        # Combine all specified text fields
        text_parts = []
        for field in text_fields:
            value = doc.get(field, '')
            if value:
                text_parts.append(str(value))
        
        yield {
            'docno': doc['docno'],
            'text': ' '.join(text_parts).strip()
        }


def build_bm25_index(data_dir: str, dataset, logger: logging.Logger, index_prefix: str = "robust04", text_fields: list = None, force: bool = False):
    """Build BM25 index for the dataset."""
    index_dir = os.path.join(data_dir, f"{index_prefix}_bm25_index")
    
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
    
    indexer = pt.IterDictIndexer(index_dir, meta={'docno': 256, 'text': 4096})
    index_ref = indexer.index(custom_corpus_iter(dataset, text_fields))
    
    elapsed = datetime.now() - start_time
    logger.info(f"BM25 index built successfully in {elapsed}")
    logger.info(f"Index location: {index_dir}")
    
    return index_dir


def build_splade_index(data_dir: str, dataset, logger: logging.Logger, device: str, index_prefix: str = "robust04", text_fields: list = None, force: bool = False, batch_size: int = 256, shard_id=0, num_shards=1):
    """Build SPLADE index for the dataset."""
    base_index_dir = os.path.join(data_dir, f"{index_prefix}_splade_index")
    
    if num_shards > 1:
        index_dir = os.path.join(base_index_dir, f"part_{shard_id}")
    else:
        index_dir = base_index_dir
    
    if os.path.exists(index_dir):
        if force:
            logger.info(f"Removing existing index at {index_dir}")
            shutil.rmtree(index_dir)
        else:
            logger.info(f"Index already exists at {index_dir}. Skipping.")
            return index_dir
    
    logger.info(f"Building SPLADE index at {index_dir} (Shard {shard_id+1}/{num_shards})...")
    
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
    logger.info(f"Using batch size: {batch_size}")
    splade = pyt_splade.Splade(
        model=splade_model,
        device=device,
        max_length=256
    )
    
    start_time = datetime.now()
    
    # Build SPLADE index using doc encoder pipeline
    # batch_size controls GPU throughput - higher = faster but more VRAM
    splade_indexer = splade.doc_encoder(batch_size=batch_size, verbose=True) >> pt.IterDictIndexer(index_dir, meta={'docno': 256, 'text': 4096})
    index_ref = splade_indexer.index(custom_corpus_iter(dataset, text_fields, shard_id, num_shards))
    
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SPLADE encoding (default: 64, increase if GPU has free VRAM)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="robust04",
        choices=list(DATASET_CONFIGS.keys()),
        help=f"Dataset to index (default: robust04, choices: {', '.join(DATASET_CONFIGS.keys())})"
    )
    
    parser.add_argument(
        "--shard-id", 
        type=int, 
        default=0, 
        help="Current shard ID (0-based)"
    )
    parser.add_argument(
        "--num-shards", 
        type=int, 
        default=1, 
        help="Total number of shards"
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
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[args.dataset]
    index_prefix = dataset_config["index_prefix"]
    text_fields = dataset_config["text_fields"]
    irds_id = dataset_config["irds_id"]
    
    # Register OWI dataset if needed
    if args.dataset.startswith("owi"):
        logger.info("Registering OWI dataset...")
        ir_datasets_owi.register()
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (irds:{irds_id})...")
    dataset = pt.get_dataset(f"irds:{irds_id}")
    
    # Build indices
    if build_bm25:
        logger.info("-" * 40)
        bm25_dir = build_bm25_index(data_dir, dataset, logger, index_prefix=index_prefix, text_fields=text_fields, force=args.force)
    
    if build_splade:
        logger.info("-" * 40)
        splade_dir = build_splade_index(data_dir, dataset, logger, device, index_prefix=index_prefix, text_fields=text_fields, force=args.force, batch_size=args.batch_size, shard_id=args.shard_id, num_shards=args.num_shards)
    
    logger.info("=" * 60)
    logger.info("Index building complete!")
    logger.info("=" * 60)
    
    # Print summary
    logger.info("\nSummary:")
    if build_bm25:
        logger.info(f"  BM25 index:   {os.path.join(data_dir, f'{index_prefix}_bm25_index')}")
    if build_splade:
        logger.info(f"  SPLADE index: {os.path.join(data_dir, f'{index_prefix}_splade_index')}")
    
    logger.info("\nTo use these indices in experiments, update the index paths or set:")
    logger.info(f"  export IR_DATA_DIR={data_dir}")


if __name__ == "__main__":
    main()
