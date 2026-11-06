#!/bin/bash
# isgc_*: K physics-only collectors on the queue of an EXISTING job directory (its sampler runs in another job / on another GPU):
# the split test of the fast flow (sampler GPU + collector GPU sharing one queue over NFS).
#   sbatch -J isgc_<key> -A vision -p svl --gres=gpu:titanrtx:1 --time=<T> job_collect.sh <job dir> <code> [K=1]
#SBATCH -c 9
#SBATCH --mem=48G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; CODE=$2; K=${3:-1}
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $F/scripts/env_fast.sh
source $F/scripts/fast_env.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_PHYSICS_ONLY=1
[ "${STILL_EXIT:-0}" = 1 ] && export ISPATIALGYM_ORACLE_STILL_EXIT=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac; rm -rf ${OMNIGIBSON_APPDATA_PATH}.c* 2>/dev/null' EXIT
T_START=$(date +%s)
JOB_END=${SLURM_JOB_END_TIME:-$(( T_START + 14400 ))}
DEADLINE=$(( JOB_END - DEMO_LIMIT_S - 420 ))
mkdir -p $JOBDIR/claims
echo "ISGC_START $SLURM_JOB_ID host=$(hostname -s) k=$K jobdir=$JOBDIR start=$(date -Is)" | tee -a $JOBDIR/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr -d ' ' | tr '\n' ' ')"; sleep 60; done ) >> $JOBDIR/gpu_collect.log 2>/dev/null &
MONPID=$!
collector() {
  local i=$1 q=0 barren=0 rc=0 app=${OMNIGIBSON_APPDATA_PATH}.c$i
  rsync -a --exclude 'local/logs' $OMNIGIBSON_APPDATA_PATH/ $app/ 2>/dev/null || cp -r $OMNIGIBSON_APPDATA_PATH $app
  export OMNIGIBSON_APPDATA_PATH=$app TMPDIR=$app/tmp
  mkdir -p $TMPDIR
  sleep $(( (i - 1) * 45 ))
  while true; do
    [ $(date +%s) -ge $DEADLINE ] && { echo "COLLECT_DEADLINE x$i" | tee -a $JOBDIR/job.txt; break; }
    [ $barren -ge 3 ] && { echo "COLLECT_GIVE_UP x$i" | tee -a $JOBDIR/job.txt; break; }
    if [ -e $JOBDIR/sampling.done ] && [ "$(python $F/scripts/queue_watch.py $JOBDIR left)" -eq 0 ]; then echo "COLLECT_DONE x$i" | tee -a $JOBDIR/job.txt; break; fi
    q=$(( q + 1 ))
    local r0=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l) t1=$(date +%s)
    echo "ISGC_COLLECT $SLURM_JOB_ID x$i q=$q left=$(python $F/scripts/queue_watch.py $JOBDIR left) start=$(date -Is)" | tee -a $JOBDIR/job.txt
    python -m ispatialgym.oracle.run_isg --queue $JOBDIR/inst --claims $JOBDIR/claims --done-file $JOBDIR/sampling.done --poll-s 15 \
      --results $JOBDIR/results.jsonl --deadline $DEADLINE --mode oracle --out $JOBDIR/demo --record hdf5 > $JOBDIR/collect_x${i}_q$q.log 2>&1 &
    local qpid=$! killed=""
    while kill -0 $qpid 2>/dev/null; do
      sleep 20
      local over=$(python $F/scripts/queue_watch.py $JOBDIR over $qpid $DEMO_LIMIT_S)
      if [ -n "$over" ]; then echo "DEMO_TIMEOUT $over x$i" | tee -a $JOBDIR/job.txt; echo "$over" >> $JOBDIR/demo_timeouts.txt; kill -INT $qpid 2>/dev/null; sleep 60; kill -KILL $qpid 2>/dev/null; killed=1; break; fi
    done
    wait $qpid; rc=$?; [ -n "$killed" ] && rc=124
    local r1=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
    echo "COLLECT_EXIT x$i q=$q rc=$rc wall_s=$(( $(date +%s) - t1 )) results=$(( r1 - r0 ))" | tee -a $JOBDIR/job.txt
    if [ "$r1" -eq "$r0" ] && [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then barren=$(( barren + 1 )); else barren=0; fi
    [ "$rc" -eq 0 ] && [ -e $JOBDIR/sampling.done ] && break
  done
}
CP=""; for i in $(seq 1 $K); do collector $i & CP="$CP $!"; done
wait $CP
kill $MONPID 2>/dev/null
KIT_RC=0
echo "ISGC_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T_START )) end=$(date -Is)" | tee -a $JOBDIR/job.txt
