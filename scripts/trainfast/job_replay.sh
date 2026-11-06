#!/bin/bash
# isgr_* (trainfast campaign, REPLAY stage): the kinematic replayer renders the physics-only demos of one job into BEHAVIOR LeRobot v3
# shards (ispatialgym.multienv.replay --watch: every successful demo under the job's demo root, done when its done-file exists).
#   sbatch -J isgr_<key> -A vision -p svl --gres=gpu:titanrtx:1 --time=<T> job_replay.sh <job dir> <code> [slots]
#SBATCH -c 9
#SBATCH --mem=48G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; CODE=$2; SLOTS=${3:-1}
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $F/scripts/env_fast.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache
export ISPATIALGYM_HEAD_CAMERA_ONLY=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
OUT=$JOBDIR/replay; mkdir -p $OUT
T0=$(date +%s)
echo "ISGR_START $SLURM_JOB_ID key=$ALIAS_KEY host=$(hostname -s) code=$CODE demos=$(ls $JOBDIR/demo | wc -l) slots=$SLOTS start=$(date -Is)" | tee -a $JOBDIR/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"; sleep 60; done ) >> $OUT/gpu.log 2>&1 &
MON=$!
python -m ispatialgym.multienv.replay --watch $JOBDIR/demo --done-file $JOBDIR/sampling.done --root $OUT/lerobot --task $ALIAS --slots $SLOTS \
  --batched-usd-edits --async-flush --repo-prefix replay --results $OUT/replay.jsonl > $OUT/replay.log 2>&1
RC=$?
kill $MON 2>/dev/null
KIT_RC=$RC
echo "ISGR_END $SLURM_JOB_ID rc=$RC wall_s=$(( $(date +%s) - T0 )) replayed=$(grep -c REPLAY_DEMO $OUT/replay.jsonl 2>/dev/null) $(grep REPLAY_SUMMARY $OUT/replay.jsonl 2>/dev/null | tail -1 | cut -c1-300) end=$(date -Is)" | tee -a $JOBDIR/job.txt
