#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --exclude=cn47,cn48
#SBATCH --output=logs/build_bm25_%j.out
#SBATCH --error=logs/build_bm25_%j.err

# BM25 Index Builder for IR Project (CPU nodes: cn77/cn78)
# This script builds only the BM25 index using CPU-only nodes.
#
# Usage:
#   sbatch build_indices_bm25.sh                # Build BM25 index
#   sbatch build_indices_bm25.sh --force        # Force rebuild existing index

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Data directory - uses the symlink to the shared storage
DATA_DIR="./data"

echo "=============================================="
echo "IR Project - BM25 Index Builder (CPU)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Data directory: $DATA_DIR"
echo "Arguments: $@"
echo "=============================================="

# Run the BM25 index builder (CPU only)
python build_indices.py --data-dir "$DATA_DIR" --bm25-only --device cpu "$@"

echo "=============================================="
echo "BM25 index building complete!"
echo "=============================================="
