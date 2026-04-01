import logging
import os
import sys
import shutil
from datetime import datetime

import torch
import pyterrier as pt
import pyt_splade
import ir_datasets

# Import your custom configuration
import ir_datasets_owi

# --- CONFIGURATION ---
INDEX_DIR = "./data/owi_splade_index"
BATCH_SIZE = 64  # Base batch size (scaled by GPU count)
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

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

def owi_iter():
    """Custom iterator for OWI that uses .default_text() to clean HTML."""
    # We load the dataset directly here to ensure access to the custom class
    dataset = ir_datasets.load("owi/dev")
    for doc in dataset.docs_iter():
        yield {
            'docno': doc.doc_id,
            'text': doc.default_text()
        }

def main():
    logger = setup_logging()

    ir_datasets_owi.register()
    logger.info("Initializing PyTerrier...")
    if not pt.started():
        pt.init()

    # check if index already exists
    if os.path.exists(INDEX_DIR):
        logger.info(f"Removing existing SPLADE index at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)
    
    logger.info(f"Building SPLADE index at {INDEX_DIR}...")
    logger.info("This may take a while (longer than BM25)...")

    device = detect_device()
    gpu_count = torch.cuda.device_count()
    
    current_batch_size = BATCH_SIZE
    if gpu_count > 1:
        current_batch_size = BATCH_SIZE * gpu_count

    logger.info(f"Using Hugging Face SPLADE model: {MODEL_NAME}")
    logger.info(f"Initializing SPLADE on device: {device}")
    logger.info(f"Using batch size: {current_batch_size}")

    splade = pyt_splade.Splade(
        model=MODEL_NAME,
        device=device,
        max_length=256
    )

    if gpu_count > 1:
        splade.model = torch.nn.DataParallel(splade.model)

    start_time = datetime.now()

    # build index
    splade_indexer = splade.doc_encoder(batch_size=current_batch_size, verbose=True) >> \
                     pt.IterDictIndexer(INDEX_DIR, meta={'docno': 64, 'text': 4096})

    # Run the indexer
    index_ref = splade_indexer.index(owi_iter(), batch_size=current_batch_size)

    elapsed = datetime.now() - start_time
    logger.info(f"SPLADE index built successfully in {elapsed}")
    logger.info(f"Index location: {INDEX_DIR}")

    os._exit(0)

if __name__ == "__main__":
    main()
