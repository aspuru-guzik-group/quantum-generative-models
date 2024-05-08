#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=16_pr_be
#SBATCH --output=logs/benchmark_%j.txt
#SBATCH --error=logs/benchmark_%j.err


module load gcccore gompi/2022a python/3.10.4 pytorch syba
python benchmark_models_v0.py --config benchmark_models_settings_$1.json $2

