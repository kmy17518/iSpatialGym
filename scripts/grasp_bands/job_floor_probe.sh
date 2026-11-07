#!/bin/bash
# i7-floorfix 2026-11-07: the floor-finger probe (floor_finger_probe.py: the floor maps' IK with / without the ground slab for the fingers)
# on one GPU, from a FROZEN snapshot. Every output under FP_OUT (GBH too: the asset cache / job dir fallback of env_cluster.sh):
#   ISG_GB_ROOT=<snapshot> FP_OUT=<dir> FP_GLIB=<glib root> [FP_ARGS="--deep"] sbatch -A vision -p svl --exclude=svl9 \
#     --gres=gpu:titanrtx:1 --time=00:30:00 -o <dir>/slurm_%j.out job_floor_probe.sh
#SBATCH --job-name=isg_floorprobe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -u
FP_OUT=${FP_OUT:?}; FP_GLIB=${FP_GLIB:?}
mkdir -p $FP_OUT/gbh
export GBH=$FP_OUT/gbh
source ${ISG_GB_ROOT:?}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
echo "FP_JOB host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $FP_OUT glib $FP_GLIB start $(date +%s)"
python -u $R/scripts/grasp_bands/floor_finger_probe.py --glib $FP_GLIB --out $FP_OUT ${FP_ARGS:---deep} >> $FP_OUT/probe.log 2>&1
rc=$?
echo "FP_JOB_DONE rc=$rc end $(date +%s)"
