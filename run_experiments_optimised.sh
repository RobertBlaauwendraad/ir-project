#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/run_experiments_%j.out
#SBATCH --error=logs/run_experiments_%j.err

# IR Experiments Runner
#
# Supported datasets:
#   - robust04: TREC Robust 2004 (default)
#   - owi: Dutch government web pages (full)
#   - owi/subsampled: Dutch government web pages (subsampled)
#
# Usage:
#   sbatch run_experiments.sh                              # Run all experiments on robust04
#   sbatch run_experiments.sh --dataset owi                # Run all experiments on OWI
#   sbatch run_experiments.sh --exp 1 2 3                  # Run specific experiments
#   sbatch run_experiments.sh --exp 1-5                    # Run experiment range
#   sbatch run_experiments.sh --dataset owi --exp 1-5      # Run experiments 1-5 on OWI

mkdir -p logs

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

DATA_DIR="./data"

python run_experiments_optimised.py --data-dir "$DATA_DIR" --output-dir ./experiment_results --device cuda "$@"1~python run_experiments.py --data-dir "$DATA_DIR" --output-dir ./experiment_results --device cuda "$@"1~python run_experiments.py --data-dir "$DATA_DIR" --output-dir ./experiment_results --device cuda "$@"
