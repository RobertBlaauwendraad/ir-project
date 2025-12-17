#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
##SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err

# IR Experiments Runner
# Usage:
#   sbatch experiments.sh                    # Run all experiments
#   sbatch experiments.sh 1 2 3             # Run specific experiments
#   sbatch experiments.sh 1-5               # Run experiment range
#   sbatch experiments.sh 1-5 10 20         # Run experiments 1-5, 10, and 20

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run experiments
if [ $# -eq 0 ]; then
    # No arguments: run all experiments
    python run_experiments.py --output-dir ./experiment_results
else
    # Arguments provided: pass them as experiment IDs
    python run_experiments.py --exp "$@" --output-dir ./experiment_results
fi