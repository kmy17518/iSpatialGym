#!/bin/bash
# isgr_*: P replay processes sharing ONE GPU, each on its share of a job's successful demos (round robin) - replay throughput per GPU.
#   sbatch -J isgr_p<P> -A vision -p svl --gres=gpu:titanrtx:1 -c <9P> --mem=<32P>G --time=<T> job_replay_multi.sh <job dir> <code> <P> <out dir>
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; CODE=$2; P=$3; OUT=$4
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
# one trunk (2026-11-05): the replay code's own scripts; goal occlusion screened here for the families the sampler does not screen
# (multienv/replay.py --goal-occlusion auto, REPLAY_GOAL_OCCLUSION=on|off overrides)
S=$CODE/scripts/trainfast; [ -f $S/env_fast.sh ] || S=$F/scripts
source $S/env_fast.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_HEAD_CAMERA_ONLY=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac; rm -rf ${OMNIGIBSON_APPDATA_PATH}.r* 2>/dev/null' EXIT
mkdir -p $OUT
python3 - "$JOBDIR" "$P" "$OUT" <<'PY'
import glob, json, os, sys
job, P, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ok = sorted(os.path.dirname(m) for m in glob.glob(os.path.join(job, "demo", "*", "meta.json")) if json.load(open(m)).get("success") and os.path.exists(os.path.join(os.path.dirname(m), "demo.hdf5")))
for p in range(P):
    open(os.path.join(out, f"demos_{p}.txt"), "w").write("\n".join(ok[p::P]) + "\n")
print(f"{len(ok)} demos split over {P} processes")
PY
T0=$(date +%s)
echo "ISGR_START $SLURM_JOB_ID host=$(hostname -s) P=$P start=$(date -Is)" | tee -a $OUT/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"; sleep 30; done ) >> $OUT/gpu.log 2>&1 &
MON=$!
PIDS=""
for p in $(seq 0 $(( P - 1 ))); do
  app=${OMNIGIBSON_APPDATA_PATH}.r$p; rsync -a --exclude 'local/logs' $OMNIGIBSON_APPDATA_PATH/ $app/ 2>/dev/null
  ( export OMNIGIBSON_APPDATA_PATH=$app TMPDIR=$app/tmp; mkdir -p $TMPDIR
    python -m ispatialgym.multienv.replay --demos @$OUT/demos_$p.txt --root $OUT/lerobot_$p --task $ALIAS --slots 1 ${REPLAY_BATCHED:---batched-usd-edits} --async-flush \
      --proprio loaded --warmup-s 15 --repo-prefix replay --goal-occlusion ${REPLAY_GOAL_OCCLUSION:-auto} ${REPLAY_EXTRA:-} --results $OUT/replay_$p.jsonl > $OUT/replay_$p.log 2>&1 ) &
  PIDS="$PIDS $!"
done
wait $PIDS
kill $MON 2>/dev/null
KIT_RC=0
echo "ISGR_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T0 )) end=$(date -Is)" | tee -a $OUT/job.txt
for p in $(seq 0 $(( P - 1 ))); do grep REPLAY_SUMMARY $OUT/replay_$p.jsonl | tail -1 | cut -c1-260 | tee -a $OUT/job.txt; done
