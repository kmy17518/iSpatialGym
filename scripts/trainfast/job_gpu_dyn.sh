#!/bin/bash
# isgd_* (trainfast follow-up F1, 2026-11-01): one physics-only oracle demo of one instance ALONE on a GPU, CPU PhysX (G=0) or GPU PhysX
# (G=1: ISPATIALGYM_FAST_COLLECTOR_GPU_DYNAMICS, the simulator device staying cpu). Per-step costs come from the demo's meta.json
# (fast.step_costs); /usr/bin/time gives the process's CPU seconds and peak RSS; gpu.log samples memory / utilisation every 10 s.
#   sbatch -J isgd_<key> -A vision -p svl --gres=gpu:titanrtx:1 -c 12 --mem=32G --time=01:30:00 job_gpu_dyn.sh <job dir of the instance> <tro> <code> <out dir> <G>
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; TRO=$2; CODE=$3; OUT=$4; G=$5
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $F/scripts/env_fast.sh
source $F/scripts/fast_env.sh
for v in $(grep -o "^ISPATIALGYM_[A-Z0-9_]*" $JOBDIR/job.env); do export $v; done
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_PHYSICS_ONLY=1 ISPATIALGYM_FAST_COLLECTOR_GPU_DYNAMICS=$G
[ "${STILL_EXIT:-0}" = 1 ] && export ISPATIALGYM_ORACLE_STILL_EXIT=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
mkdir -p $OUT
T0=$(date +%s)
echo "ISGD_START $SLURM_JOB_ID host=$(hostname -s) gpu=\"$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)\" gpu_dynamics=$G tro=$(basename $TRO) start=$(date -Is)" | tee -a $OUT/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"; sleep 10; done ) >> $OUT/gpu.log 2>&1 &
MON=$!
/usr/bin/time -v -o $OUT/time.txt python -m ispatialgym.oracle.run_isg --instance $TRO --mode oracle --record hdf5 --out $OUT/demo > $OUT/run.log 2>&1
RC=$?
kill $MON 2>/dev/null
echo "ISGD_END $SLURM_JOB_ID rc=$RC wall_s=$(( $(date +%s) - T0 )) end=$(date -Is)" | tee -a $OUT/job.txt
