#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15G
#SBATCH --time=4:00:00
#SBATCH --output=logs/build_splade_%j.out
#SBATCH --error=logs/build_splade_%j.err

# SPLADE Index Builder for IR Project (GPU nodes: cn47/cn48)
# This script builds only the SPLADE index using GPU acceleration.
#
# Usage:
#   sbatch build_indices_splade.sh                # Build SPLADE index
#   sbatch build_indices_splade.sh --force        # Force rebuild existing index

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
python build_indices.py --data-dir "$DATA_DIR" --splade-only --device cuda "$@"

echo "=============================================="
echo "SPLADE index building complete!"
echo "=============================================="
