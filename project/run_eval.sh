#!/bin/bash

#SBATCH --output="SLURM_SUBMIT_DIR/slurm-%j.out"
#SBATCH --error="SLURM_SUBMIT_DIR/slurm-%j.err"
#SBATCH --time=04:00:00
#SBATCH --job-name="Model Evaluation"
#SBATCH --partition=students
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

uv run 05_evaluate_trained_model.py
