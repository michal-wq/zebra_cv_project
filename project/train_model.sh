#!/bin/bash

#SBATCH --output="SLURM_SUBMIT_DIR/slurm-%j.out"  ## Im Verzeichnis aus dem sbatch aufgerufen wird, wird ein Logfile mit dem Namen slurm-[Jobid].out erstellt.
#SBATCH --error="SLURM_SUBMIT_DIR/slurm-%j.err"   ## Ähnlich wie --output. Jedoch ein Log für Fehlermeldungen.
#SBATCH --time=5:00:00           ## Zeitlimite. Diese sollte gleich oder kleiner der Partitions Zeitlimite sein. In diesem Fall ist diese auf 1 Stunde und 30 Minuten gesetzt.
#SBATCH --job-name="Model Training"   ## Job Name.
#SBATCH --partition=students	 ## Partitionsname. Die zur Verfügung stehenden Partitionen können mit dem Befehl sinfo angezeigt werden
#SBATCH --mem=16G               ## Der Arbeitsspeicher, welcher für den Job reserviert wird
#SBATCH --cpus-per-task=16     ## Die Anzahl virtueller Cores, die für den Job reserviert werden
#SBATCH --gres=gpu:a100:1

uv run 03_optuna_training_pipeline.py