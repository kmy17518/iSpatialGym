#!/bin/bash
# python -m ispatialgym.oracle.run_isg in a GPU job, with the warm Kit app data (cluster rule).
#   sbatch scripts/eval_instances/job_run_isg.sh --task alignment_axial --instance-indices 0 1 --mode replay --record video --out <dir>
#   (each dataset instance replays its own demo; --replay <demo.hdf5> names another for one index)
# Logs: datasets/logs/run_isg/ (create it before submitting).
#SBATCH -J isg_run_isg
#SBATCH -p svl
#SBATCH -A vision
#SBATCH --gres=gpu:1
#SBATCH --constraint=turing|ampere|ada
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=viscam7,viscam11,visionlab-dgx1
#SBATCH -o datasets/logs/run_isg/%x_%j.log
set -u
KIT_RC=1
source scripts/eval_instances/env.sh
trap 'kit_appdata_teardown' EXIT
echo "RUN_ISG_JOB ${SLURM_JOB_ID} host=$(hostname -s) gpu=\"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\" args=$* start=$(date -Is)"
python -m ispatialgym.oracle.run_isg "$@"
KIT_RC=$?
echo "RUN_ISG_JOB_EXIT $KIT_RC end=$(date -Is)"
