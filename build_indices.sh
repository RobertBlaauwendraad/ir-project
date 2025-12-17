#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --mem=15G
#SBATCH --output=logs/build_indices_%j.out
#SBATCH --error=logs/build_indices_%j.err

# Index Builder for IR Project
# This script builds the BM25 and SPLADE indices needed for experiments.
#
# Supported datasets:
#   - robust04: TREC Robust 2004 (default)
#   - owi: OWI (full)
#   - owi/subsampled: OWI (subsampled)
#
# Usage:
#   sbatch build_indices.sh                        # Build all indices for robust04
#   sbatch build_indices.sh --dataset owi          # Build indices for OWI dataset
#   sbatch build_indices.sh --dataset owi/subsampled  # Build indices for OWI subsampled
#   sbatch build_indices.sh --bm25-only            # Build only BM25 index
#   sbatch build_indices.sh --splade-only          # Build only SPLADE index
#   sbatch build_indices.sh --force                # Force rebuild existing indices

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Data directory - uses the symlink to the shared storage
DATA_DIR="./data"

echo "=============================================="
echo "IR Project - Index Builder"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Data directory: $DATA_DIR"
echo "Arguments: $@"
echo "=============================================="

# Enable HuggingFace offline mode if model is cached locally
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Better CUDA memory management for shared GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the index builder
python build_indices.py --data-dir "$DATA_DIR" "$@"

echo "=============================================="
echo "Index building complete!"
echo "=============================================="
