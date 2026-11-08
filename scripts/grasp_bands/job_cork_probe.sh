#!/bin/bash
# i7-cork 2026-11-08: one GPU probe of the cork routes (cork_rest_probe.py, ...) from a FROZEN snapshot; every output under CP_OUT
# (GBH too: the asset cache / job dir fallback of env_cluster.sh):
#   ISG_GB_ROOT=<snapshot> CP_OUT=<dir> CP_SCRIPT=cork_rest_probe.py [CP_ARGS="--trials 12"] sbatch -A vision -p svl --exclude=svl9 \
#     --gres=gpu:titanrtx:1 --time=01:00:00 -o <dir>/slurm_%j.out job_cork_probe.sh
#SBATCH --job-name=isg_corkprobe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -u
CP_OUT=${CP_OUT:?}; CP_SCRIPT=${CP_SCRIPT:-cork_rest_probe.py}
mkdir -p $CP_OUT/gbh
export GBH=$CP_OUT/gbh
source ${ISG_GB_ROOT:?}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
echo "CP_JOB host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $CP_OUT start $(date +%s)"
python -u $R/scripts/grasp_bands/$CP_SCRIPT --out $CP_OUT ${CP_ARGS:-} >> $CP_OUT/probe.log 2>&1
rc=$?
echo "CP_JOB_DONE rc=$rc end $(date +%s)"
