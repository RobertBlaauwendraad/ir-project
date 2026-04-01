#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --exclude=cn47,cn48
#SBATCH --array=0-29
#SBATCH --output=logs/%j_shard_%a.out
#SBATCH --error=logs/%j_shard_%a.err

# BM25 Index Builder for IR Project (GPU nodes: cn47/cn48)
# This script builds only the BM25 index.
#
# Usage:
#   sbatch build_indices_bm25_sharding.sh                # Build BM25 index
#   sbatch build_indices_bm25_sharding.sh	         # Force rebuild existing index

# Total number of shards (Has to match array size)
NUM_SHARDS=30

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
python build_indices_sharding.py --data-dir "$DATA_DIR" --bm25-only --shard-id $SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS "$@" --threads 8 --force --dataset owi

echo "=============================================="
echo "SPLADE index building complete!"
echo "=============================================="
