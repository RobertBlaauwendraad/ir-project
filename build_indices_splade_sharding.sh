#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15G
#SBATCH --time=4:00:00
#SBATCH --array=0-29
#SBATCH --output=logs/%j_shard_%a.out
#SBATCH --error=logs/%j_shard_%a.err

# SPLADE Index Builder for IR Project (GPU nodes: cn47/cn48)
# This script builds only the SPLADE index using GPU acceleration.
#
# Usage:
#   sbatch build_indices_splade.sh                # Build SPLADE index
#   sbatch build_indices_splade.sh --force        # Force rebuild existing index

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
echo "IR Project - SPLADE Index Builder (GPU)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Data directory: $DATA_DIR"
echo "Arguments: $@"
echo "=============================================="

# Enable HuggingFace offline mode if model is cached locally
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Better CUDA memory management for shared GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the SPLADE index builder with CUDA device
python build_indices_sharding.py --data-dir "$DATA_DIR" --splade-only --device cuda --shard-id $SLURM_ARRAY_TASK_ID --num-shards $NUM_SHARDS "$@"

echo "=============================================="
echo "SPLADE index building complete!"
echo "=============================================="
