#!/bin/bash
# i7-memory (owner 2026-11-08 01:45 part C): the per-alias outcome memory's offline aggregation after a batch - a CPU job submit_fast.py
# queues after every GPU job (--dependency=afterany); it builds a NEW version <root>/alias_memory/v<NNNN> only when no other job of the
# batch (the ledger's rows with this tag) is still queued / running (the last one aggregates). Reads job dirs, writes only the memory root.
#   sbatch -A vision -p svl --exclude=svl9 -c 2 --mem=8G --time=1:00:00 --dependency=afterany:<id> job_memory.sh <root> <tag> <code snapshot> [mode]
# i7-memory2: <mode> production (default: <root>/alias_memory_production, production-mode jobs only) | validation (<root>/alias_memory_validation)
#SBATCH -c 2
#SBATCH --mem=8G
set -u
ROOT=$1
TAG=$2
CODE=$3
MODE=${4:-production}
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
export PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1
echo "ISGF_MEMORY $SLURM_JOB_ID root=$ROOT tag=$TAG code=$CODE mode=$MODE start=$(date -Is)"
python -m ispatialgym.sampling.alias_memory aggregate --root "$ROOT" --batch "$TAG" --if-batch-done --mode "$MODE"
