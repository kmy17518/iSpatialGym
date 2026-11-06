#!/bin/bash
# isgf_* (trainfast campaign, BASELINE flow): train85's job_train.sh flow on the frozen base snapshot, writing under
# /vision/u/minyeong/projects/isg_home/trainfast. Sampler (multiply_isg --split train) then oracle demos (run_isg --queue, rendered
# as train85 did), one after the other on one GPU. Used as the A side of the speed comparison.
#   sbatch -J isgf_<key> -A vision -p svl --gres=gpu:titanrtx:1 --time=<T> job_base.sh <job dir>   (job.env by submit_fast.py)
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1
F=/vision/u/minyeong/projects/isg_home/trainfast
T85=/vision/u/minyeong/projects/isg_home/train85
source $JOBDIR/job.env
T_START=$(date +%s)
JOB_END=${SLURM_JOB_END_TIME:-$(( T_START + TIME_LIMIT_S ))}
KIT_RC=1
export ISGE_BAND_STORE=1 ISGE_CODE
[ -n "${TRUNK_CHECK:-}" ] && export ISGE_BAND_TRUNK_CHECK=$TRUNK_CHECK
source $F/scripts/env_fast.sh
[ -n "${ISGE_BAND_TRUNK_CHECK:-}" ] && export ISPATIALGYM_BAND_TRUNK_CHECK=$ISGE_BAND_TRUNK_CHECK
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache
if [ "${START_PER_ATTEMPT:-0}" = 1 ]; then export ISPATIALGYM_START_PER_ATTEMPT=1; else unset ISPATIALGYM_START_PER_ATTEMPT; fi
if [ "${STILL_EXIT:-0}" = 1 ]; then export ISPATIALGYM_ORACLE_STILL_EXIT=1; else unset ISPATIALGYM_ORACLE_STILL_EXIT; fi
CUSTOMS_ARG=""
[ "${START_PER_ATTEMPT:-0}" = 1 ] && CUSTOMS_ARG="--customs $ISGE_CODE/ispatialgym/assets/task_specs/isg_customs.json"
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
mkdir -p $JOBDIR/claims
echo "$SLURM_JOB_ID" > $JOBDIR/jobid
# reach bands: train85's per-alias store (read-only source) and this campaign's own
for B in $T85/bands_train/$ALIAS_KEY $F/bands_train/$ALIAS_KEY; do [ -d "$B" ] && cp -n --preserve=timestamps $B/* $ISPATIALGYM_REACH_BANDS/ 2>/dev/null; done
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
HEAD="key=$ALIAS_KEY alias=$ALIAS host=$(hostname -s) gpu=\"$GPU\" isg=$(git -C $R rev-parse --short HEAD) b1k=$(git -C $R/Behavior-1K rev-parse --short HEAD)"
echo "ISGF_SAMPLE $SLURM_JOB_ID $HEAD flow=base variant=${CODE_VARIANT:-base} start_per_attempt=${ISPATIALGYM_START_PER_ATTEMPT:-0} still_exit=${ISPATIALGYM_ORACLE_STILL_EXIT:-0} start_index=$START n=$N max_attempts=$MAX_ATTEMPTS limit_s=$SAMPLE_LIMIT_S job_end=$JOB_END appdata=$OMNIGIBSON_APPDATA_PATH jb=$JB start=$(date -Is)" | tee -a $JOBDIR/job.txt
t0=$(date +%s)
timeout -s INT -k 300 $SAMPLE_LIMIT_S python -m ispatialgym.sampling.multiply_isg --split train --template $TEMPLATE --instances $N --start-index $START \
  --out $JOBDIR/inst --eval-run $EVAL_INDEX --no-write-back --max-attempts $MAX_ATTEMPTS $KEEP_SCENE $CUSTOMS_ARG --log-level INFO > $JOBDIR/sample.log 2>&1
rc=$?
SAMPLE_RC=$rc
echo "SAMPLE_EXIT $rc wall_s=$(( $(date +%s) - t0 )) written=$(grep -c ' written (attempt ' $JOBDIR/sample.log)" | tee -a $JOBDIR/job.txt
mkdir -p $F/bands_train/$ALIAS_KEY
python $F/scripts/publish_job_bands.py "$ISPATIALGYM_REACH_BANDS" "$F/bands_train/$ALIAS_KEY" "$t0" "$SLURM_JOB_ID" 2>&1 | tee -a $JOBDIR/job.txt
python $F/scripts/train_predemo.py $JOBDIR 2>&1 | tee -a $JOBDIR/job.txt
touch $JOBDIR/sampling.done
DEADLINE=$(( JOB_END - DEMO_LIMIT_S - 420 ))
QRC=0
q=0
barren=0
while [ $q -lt $(( N + 3 )) ]; do
  left=$(python $F/scripts/queue_watch.py $JOBDIR left)
  [ "$left" -eq 0 ] && break
  if [ $(date +%s) -ge $DEADLINE ]; then echo "DEMO_DEADLINE left=$left" | tee -a $JOBDIR/job.txt; break; fi
  if [ $barren -ge 2 ]; then echo "DEMO_GIVE_UP left=$left: two demo processes in a row ended without a result" | tee -a $JOBDIR/job.txt; break; fi
  q=$(( q + 1 ))
  echo "ISGF_DEMO $SLURM_JOB_ID $HEAD q=$q left=$left limit_s=$DEMO_LIMIT_S deadline=$DEADLINE start=$(date -Is)" | tee -a $JOBDIR/job.txt
  r0=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
  t1=$(date +%s)
  python -m ispatialgym.oracle.run_isg --queue $JOBDIR/inst --claims $JOBDIR/claims --done-file $JOBDIR/sampling.done \
    --results $JOBDIR/results.jsonl --deadline $DEADLINE --mode oracle --out $JOBDIR/demo --video --video-size 540 --dev > $JOBDIR/demo_q$q.log 2>&1 &
  qpid=$!
  echo "$qpid" > $JOBDIR/queue_q$q.pid
  killed=""
  while kill -0 $qpid 2>/dev/null; do
    sleep 20
    over=$(python $F/scripts/queue_watch.py $JOBDIR over $qpid $DEMO_LIMIT_S)
    if [ -n "$over" ]; then
      echo "DEMO_TIMEOUT $over pid=$qpid limit_s=$DEMO_LIMIT_S" | tee -a $JOBDIR/job.txt
      echo "$over" >> $JOBDIR/demo_timeouts.txt
      kill -INT $qpid 2>/dev/null; sleep 60; kill -KILL $qpid 2>/dev/null
      killed=1
      break
    fi
  done
  wait $qpid
  QRC=$?
  [ -n "$killed" ] && QRC=124
  r1=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)
  echo "DEMO_EXIT q=$q rc=$QRC wall_s=$(( $(date +%s) - t1 )) results=$(( r1 - r0 ))" | tee -a $JOBDIR/job.txt
  if [ "$r1" -eq "$r0" ] && [ "$QRC" -ne 0 ] && [ "$QRC" -ne 75 ]; then barren=$(( barren + 1 )); else barren=0; fi
done
python $F/scripts/fast_check.py $JOBDIR 2>&1 | tee -a $JOBDIR/job.txt
case "$SAMPLE_RC" in 0|2) [ "$QRC" -eq 0 ] && KIT_RC=0 ;; esac
echo "ISGF_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T_START )) kit_rc=$KIT_RC end=$(date -Is)" | tee -a $JOBDIR/job.txt
