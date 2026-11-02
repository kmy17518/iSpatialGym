#!/bin/bash
# isgc_* (trainfast packing test E16, 2026-11-02): N physics-only oracle collectors of ONE instance at the same time in one job (one GPU),
# each with its own Kit app data and a 45 s stagger as in job_fast.sh, the PhysX worker threads capped at T (ISPATIALGYM_FAST_
# COLLECTOR_PHYSX_THREADS; 0 = PhysX's own count). Tells collector-on-collector CPU contention from the sampler's. Per process:
# meta.json (fast.step_costs, loader.physx_threads), /usr/bin/time; ps.log samples each collector's thread count every 10 s.
# MODE (7th arg, default same) separates one part of the job environment per process, to find what two collectors of one job share:
#   same | home (HOME, XDG_CACHE_HOME) | isgcache (ISPATIALGYM_CACHE_DIR and its caches) | jobid (SLURM_JOB_ID only) |
#   sepjob (env_fast.sh / fast_env.sh again under SLURM_JOB_ID=<id>_p<i>: own /tmp folder, HOME, caches, Kit app data - a pair member) |
#   pin (disjoint equal CPU subsets of the job's CPUs, taskset) | pin<k> (k CPUs each, in order of the job's CPU list) |
#   copy (own copy of the job's inst/ per process) | sepjob_copy (both). probe.log: each collector's CPU affinity, the CPUs its threads last ran on
#   and the user's other processes, 240 s and 360 s after the start.
#   sbatch -J isgc_<arm> -A vision -p svl --gres=gpu:titanrtx:1 -c <cpus> --mem=48G --time=01:30:00 job_ct.sh <job dir> <tro> <code> <out> <N> <T> [MODE]
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1; TRO=$2; CODE=$3; OUT=$4; NPROC=$5; PTHREADS=$6; MODE=${7:-same}   # not N / T: job.env sets N (the instance count)
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
export ISGE_CODE=$CODE
source $F/scripts/env_fast.sh
source $F/scripts/fast_env.sh
for v in $(grep -o "^ISPATIALGYM_[A-Z0-9_]*" $JOBDIR/job.env); do export $v; done
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_PHYSICS_ONLY=1 ISPATIALGYM_FAST_COLLECTOR_PHYSX_THREADS=$PTHREADS
[ "${STILL_EXIT:-0}" = 1 ] && export ISPATIALGYM_ORACLE_STILL_EXIT=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
mkdir -p $OUT
T0=$(date +%s)
echo "ISGC_START $SLURM_JOB_ID host=$(hostname -s) cpus=${SLURM_CPUS_PER_TASK:-?} n=$NPROC physx_threads=$PTHREADS mode=$MODE tro=$(basename $TRO) start=$(date -Is)" | tee -a $OUT/job.txt
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr -d ' ')"; sleep 10; done ) >> $OUT/gpu.log 2>&1 &
MON=$!
( while true; do ps -u $(id -u) -o pid=,nlwp=,args= | grep "run_isg --instance" | grep -v grep | awk -v t=$(date +%s) '{print t, $1, $2}'; sleep 10; done ) >> $OUT/ps.log 2>&1 &
PSM=$!
( for t in 240 360; do sleep $(( t - $(date +%s) + T0 )); echo "== +$t s"
    for pid in $(pgrep -u $(id -u) -f "run_isg --instance"); do
      echo "pid $pid $(taskset -cp $pid 2>/dev/null | sed 's/.*: //') threads on CPUs: $(ps -L -o psr= -p $pid | sort -n | uniq -c | sort -rn | head -6 | awk '{printf "%s:%s ", $2, $1}') running: $(ps -L -o stat= -p $pid | grep -c R)"
    done
    ps -u $(id -u) -o pid=,pcpu=,args= | grep -v -E "run_isg|ps -u|grep|sleep|bash|sshd|probe" | cut -c1-160
  done ) > $OUT/probe.log 2>&1 &
PRB=$!
CPUS=$(python3 -c 'import os; print(" ".join(map(str, sorted(os.sched_getaffinity(0)))))')
PIDS=""
for i in $(seq 1 $NPROC); do
  app=${OMNIGIBSON_APPDATA_PATH}.p$i
  [ $MODE = sepjob ] || rsync -a --exclude 'local/logs' $OMNIGIBSON_APPDATA_PATH/ $app/ 2>/dev/null || cp -r $OMNIGIBSON_APPDATA_PATH $app
  mkdir -p $OUT/p$i $app/tmp
  ( export OMNIGIBSON_APPDATA_PATH=$app TMPDIR=$app/tmp
    PIN=""; TRO_I=$TRO
    # copy / sepjob_copy: this process loads its own copy of the job's inst/ (template, scene variants), as pair members do
    case $MODE in *copy) cp -a $JOBDIR/inst $OUT/p$i/inst; TRO_I=$OUT/p$i/${TRO#$JOBDIR/} ;; esac
    case $MODE in
      home) export HOME=$JB/home_p$i XDG_CACHE_HOME=$JB/home_p$i/.cache; mkdir -p $HOME; cp -a $JB/home/. $HOME/ ;;
      isgcache) C=$JB/isg_cache_p$i; mkdir -p $C; cp -a $JB/isg_cache/. $C/
                export ISPATIALGYM_CACHE_DIR=$C ISPATIALGYM_SAMPLING_CACHE=$C/sampling ISPATIALGYM_HELD_HOLD_CACHE=$C/held_hold.json ;;
      jobid) export SLURM_JOB_ID=${SLURM_JOB_ID}_p$i ;;
      sepjob*) export SLURM_JOB_ID=${SLURM_JOB_ID}_p$i
              source $F/scripts/env_fast.sh > $OUT/p$i/env.log 2>&1; source $F/scripts/fast_env.sh >> $OUT/p$i/env.log 2>&1
              for v in $(grep -o "^ISPATIALGYM_[A-Z0-9_]*" $JOBDIR/job.env); do export $v; done
              export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache ISPATIALGYM_PHYSICS_ONLY=1 ISPATIALGYM_FAST_COLLECTOR_PHYSX_THREADS=$PTHREADS
              trap 'kit_appdata_teardown; case "$JB" in /tmp/*) rm -rf "$JB" ;; esac' EXIT ;;
      pin*) PIN="taskset -c $(python3 -c 'import sys; c = sys.argv[3].split(); n, i = int(sys.argv[1]), int(sys.argv[2]); k = int(sys.argv[4] or len(c) // n); print(",".join(c[(i - 1) * k:i * k]))' $NPROC $i "$CPUS" "${MODE#pin}")" ;;
    esac
    echo "ISGC_ENV p$i mode=$MODE SLURM_JOB_ID=$SLURM_JOB_ID HOME=$HOME ISPATIALGYM_CACHE_DIR=$ISPATIALGYM_CACHE_DIR APPDATA=$OMNIGIBSON_APPDATA_PATH pin=${PIN:-none} tro=$TRO_I" >> $OUT/job.txt
    $PIN /usr/bin/time -v -o $OUT/p$i/time.txt python -m ispatialgym.oracle.run_isg --instance $TRO_I --mode oracle --record hdf5 --out $OUT/p$i/demo > $OUT/p$i/run.log 2>&1
    echo "ISGC_PROC p$i rc=$? wall_s=$(( $(date +%s) - T0 ))" >> $OUT/job.txt ) &
  PIDS="$PIDS $!"
  [ $i -lt $NPROC ] && sleep 45
done
wait $PIDS
kill $MON $PSM $PRB 2>/dev/null
echo "ISGC_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - T0 )) end=$(date -Is)" | tee -a $OUT/job.txt
