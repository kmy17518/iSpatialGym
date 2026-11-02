#!/bin/bash
# isgp_*: one physics-only oracle demo of one instance, ALONE on a GPU, under py-spy (per-step cost without GPU contention).
#   sbatch -J isgp_<key> -A vision -p svl --gres=gpu:titanrtx:1 --time=01:00:00 job_profile_collector.sh <job dir of the instance> <tro path> <code> <out dir> [FAST=0|1]
#SBATCH -c 9
#SBATCH --mem=48G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; TRO=$2; CODE=$3; OUT=$4; FAST=${5:-1}
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $F/scripts/env_fast.sh
source $F/scripts/fast_env.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_PHYSICS_ONLY=1 ISPATIALGYM_FAST=$FAST
[ "${STILL_EXIT:-0}" = 1 ] && export ISPATIALGYM_ORACLE_STILL_EXIT=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
mkdir -p $OUT
T0=$(date +%s)
echo "ISGP_START $SLURM_JOB_ID host=$(hostname -s) code=$CODE fast=$FAST tro=$(basename $TRO) start=$(date -Is)" | tee -a $OUT/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"; sleep 30; done ) >> $OUT/gpu.log 2>&1 &
MON=$!
PYSPY=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/py-spy
if [ "${ISGP_PYSPY:-0}" = 1 ]; then $PYSPY record --format raw --rate 25 --duration 1500 --nonblocking -o $OUT/prof.raw -- python -m ispatialgym.oracle.run_isg --instance $TRO --mode oracle --record hdf5 --out $OUT/demo > $OUT/run.log 2>&1; else python -m ispatialgym.oracle.run_isg --instance $TRO --mode oracle --record hdf5 --out $OUT/demo > $OUT/run.log 2>&1; fi
RC=$?
kill $MON 2>/dev/null
KIT_RC=0
echo "ISGP_END $SLURM_JOB_ID rc=$RC wall_s=$(( $(date +%s) - T0 )) end=$(date -Is)" | tee -a $OUT/job.txt
