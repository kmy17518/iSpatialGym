#!/bin/bash
# trainfast (adopted from merge_vecnext 2026-11-05, owner: alias validation on the fast flow = job_val.sh KIND=demo): job_fast.sh's collectors only, on a job dir prep_recollect.py made from a finished fast-flow job (its sampled
# instances, the goal overrides reverted): the same environment (env_fast.sh, fast_env.sh, the job.env switches), the same k
# long-lived `run_isg --queue` collectors with job_fast.sh's restart / timeout rules; no sampler (sampling.done from the start), no
# prebuild, no band copies (the collector reads none). Extra collector flags: ISGF_COLLECT_EXTRA (fast_env.sh).
#   submit_fast.py recollect <source job dir> --code <frozen snapshot> [--iids ...] [--root $OFH --job-prefix isgof_] (writes job.env, submits)
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
# the snapshot's scripts and the job's root (job_fast.sh, submit_fast.py)
S=${ISGF_SCRIPTS:-$F/scripts}
ROOT=${ISGF_ROOT:-$F}
K=${KCOLLECT:-2}
T_START=$(date +%s)
JOB_END=${SLURM_JOB_END_TIME:-$(( T_START + TIME_LIMIT_S ))}
KIT_RC=1
export ISGE_CODE
source $S/env_fast.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache
if [ "${START_PER_ATTEMPT:-0}" = 1 ]; then export ISPATIALGYM_START_PER_ATTEMPT=1; else unset ISPATIALGYM_START_PER_ATTEMPT; fi
if [ "${STILL_EXIT:-0}" = 1 ]; then export ISPATIALGYM_ORACLE_STILL_EXIT=1; else unset ISPATIALGYM_ORACLE_STILL_EXIT; fi
source $S/fast_env.sh
# per-job overrides win over env_fast.sh / fast_env.sh (job_fast.sh, 2026-11-05)
set -a; source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' $JOBDIR/job.env); set +a
export ISPATIALGYM_SAMPLING_CACHE=${ISGF_SAMPLING_CACHE:-$F/cache/sampling}
unset ISPATIALGYM_FAST_BAND_SHARE
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac; rm -rf ${OMNIGIBSON_APPDATA_PATH}.c* 2>/dev/null' EXIT
mkdir -p $JOBDIR/claims
echo "$SLURM_JOB_ID" > $JOBDIR/jobid
touch $JOBDIR/sampling.done
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
HEAD="key=$ALIAS_KEY alias=$ALIAS host=$(hostname -s) gpu=\"$GPU\" isg=$(cat $R/.snapshot_isg 2>/dev/null || git -C $R rev-parse --short HEAD) b1k=$(cat $R/.snapshot_b1k 2>/dev/null || git -C $R/Behavior-1K rev-parse --short HEAD)"
echo "ISGF_RECOLLECT $SLURM_JOB_ID $HEAD k=$K collect_extra=\"$COLLECT_EXTRA\" asset_cache=${ISG_VEC_ASSET_CACHE_DIR:-off} job_end=$JOB_END start=$(date -Is)" | tee -a $JOBDIR/job.txt
env | grep "^ISPATIALGYM_\|^ISG_\|^OMNIGIBSON_" | sort > $JOBDIR/env.txt
DEADLINE=$(( JOB_END - DEMO_LIMIT_S - 420 ))
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ') $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr -d ' ' | tr '\n' ' ')"; sleep 60; done ) >> $JOBDIR/gpu.log 2>/dev/null &
MONPID=$!
# COLLECTOR_PIN (job_fast.sh, default on): each collector on CPUs of its own (scripts/cpu_split.py; set 0 = the sampler's, unused here)
PINS=()
if [ "${ISPATIALGYM_FAST:-1}" != 0 ] && [ "${ISPATIALGYM_FAST_COLLECTOR_PIN:-1}" != 0 ]; then
  PINS=( $(python3 $S/cpu_split.py $K ${ISPATIALGYM_FAST_COLLECTOR_CPUS:-3}) )
  [ ${#PINS[@]} -gt 0 ] && echo "PIN $(for i in $(seq 1 $K); do printf "c$i=%s " ${PINS[$i]}; done)" | tee -a $JOBDIR/job.txt || echo "PIN off: $(nproc) CPUs" | tee -a $JOBDIR/job.txt
fi
pin_of() { [ ${#PINS[@]} -gt 0 ] && echo "taskset -c ${PINS[$1]}"; }
collector() {
  local i=$1 q=0 barren=0 rc=0 app=${OMNIGIBSON_APPDATA_PATH}.c$i
  rsync -a --exclude 'local/logs' $OMNIGIBSON_APPDATA_PATH/ $app/ 2>/dev/null || cp -r $OMNIGIBSON_APPDATA_PATH $app
  export OMNIGIBSON_APPDATA_PATH=$app TMPDIR=$app/tmp ISPATIALGYM_PHYSICS_ONLY=1
  mkdir -p $TMPDIR
  sleep $(( (i - 1) * 45 ))   # stagger the Kit starts
  while true; do
    [ $(date +%s) -ge $DEADLINE ] && { echo "COLLECT_DEADLINE c$i" | tee -a $JOBDIR/job.txt; break; }
    [ $barren -ge 3 ] && { echo "COLLECT_GIVE_UP c$i: three processes in a row ended without a result" | tee -a $JOBDIR/job.txt; break; }
    if [ "$(python $S/queue_watch.py $JOBDIR left)" -eq 0 ]; then echo "COLLECT_DONE c$i" | tee -a $JOBDIR/job.txt; break; fi
    q=$(( q + 1 ))
    local t1=$(date +%s) r0=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
    echo "ISGF_COLLECT $SLURM_JOB_ID c$i q=$q left=$(python $S/queue_watch.py $JOBDIR left) deadline=$DEADLINE start=$(date -Is)" | tee -a $JOBDIR/job.txt
    $(pin_of $i) python -m ispatialgym.oracle.run_isg --queue $JOBDIR/inst --claims $JOBDIR/claims --done-file $JOBDIR/sampling.done --poll-s 15 \
      --results $JOBDIR/results.jsonl --deadline $DEADLINE --mode oracle --out $JOBDIR/demo --record hdf5 $COLLECT_EXTRA > $JOBDIR/collect_c${i}_q$q.log 2>&1 &
    local qpid=$! killed=""
    while kill -0 $qpid 2>/dev/null; do
      sleep 20
      local over=$(python $S/queue_watch.py $JOBDIR over $qpid $DEMO_LIMIT_S)
      if [ -n "$over" ]; then
        echo "DEMO_TIMEOUT $over c$i pid=$qpid limit_s=$DEMO_LIMIT_S" | tee -a $JOBDIR/job.txt; echo "$over" >> $JOBDIR/demo_timeouts.txt
        kill -INT $qpid 2>/dev/null; sleep 60; kill -KILL $qpid 2>/dev/null; killed=1; break
      fi
    done
    wait $qpid; rc=$?; [ -n "$killed" ] && rc=124
    local r1=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
    echo "COLLECT_EXIT c$i q=$q rc=$rc wall_s=$(( $(date +%s) - t1 )) results=$(( r1 - r0 ))" | tee -a $JOBDIR/job.txt
    if [ "$r1" -eq "$r0" ] && [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then barren=$(( barren + 1 )); else barren=0; fi
    [ "$rc" -eq 0 ] && break
  done
  echo $rc > $JOBDIR/collect_c$i.rc
}
CPIDS=""
for i in $(seq 1 $K); do collector $i & CPIDS="$CPIDS $!"; done
wait $CPIDS
kill $MONPID 2>/dev/null
QRC=0; for i in $(seq 1 $K); do r=$(cat $JOBDIR/collect_c$i.rc 2>/dev/null || echo 1); [ "$r" -ne 0 ] && QRC=$r; done
python $S/fast_check.py $JOBDIR 2>&1 | tee -a $JOBDIR/job.txt
[ "$QRC" -eq 0 ] && KIT_RC=0
echo "ISGF_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T_START )) kit_rc=$KIT_RC end=$(date -Is)" | tee -a $JOBDIR/job.txt
