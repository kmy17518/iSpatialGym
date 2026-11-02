#!/bin/bash
# Build-phase timers (ISG_VEC_BUILD_TIMERS=1) of the after-change world: apple N=8 and hamper N=4, builds only (profile_build.py).
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
R=$HOME/vec_eval85/next; mkdir -p $R/timers
log() { echo "[$(date -Is)] $*"; }
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
for spec in "apple 8 $APPLE" "hamper 4 $HAMPER"; do
  set -- $spec; name=$1; n=$2; inst=$3
  log "timers $name n$n"
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --out $R/timers ) > $R/timers/${name}_n$n.log 2>&1
  log "rc=$? $(grep BUILD_PROFILE $R/timers/${name}_n$n.log | cut -c1-120)"
  grep ISG_VEC_BUILD_TIMER $R/timers/${name}_n$n.log > $R/timers/${name}_n$n.timers.txt
done
log "TIMERS DONE"
