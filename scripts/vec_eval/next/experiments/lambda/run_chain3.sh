#!/bin/bash
# Chain 3: (1) build-phase timers of the after-change world (apple N=8, hamper N=4) with the WARM Kit app-data of the speed checkout
# (the next checkout's own app-data was cold: its first launch compiled shaders, which made the first "before" apple N=1 and the first
# "after" hamper N=1 unfair), (2) fair N=1 re-runs: apple before (speed copy), apple / hamper after (next copy, speed app-data).
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
SPEED=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-speed
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
R=$HOME/vec_eval85/next; mkdir -p $R/timers $R/after_warm $R/baseline_warm
log() { echo "[$(date -Is)] $*"; }
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
run_one() {  # run_one <eval root> <out dir> <log> <instance> <N>
  local root=$1 out=$2 lg=$3 inst=$4 n=$5
  rm -rf $out
  ( export ISG_EVAL_ROOT=$root OMNIGIBSON_APPDATA_PATH=$APPDATA; source $root/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $inst --same-instance-all-slots --num-envs $n --layout grid-centered --scene-margin 5 \
      --wallclock 0 --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:chain3 --out $out ) > $lg 2>&1
  local rc=$?
  log "$(basename $out) rc=$rc $(grep -c RUN_ISG_RESULT $lg) results; $(grep -o 'world built in [0-9.]*s' $lg | head -1); $(grep RUN_ISG_VEC_BATCH $lg | tail -1 | grep -o 'batch_steps/s=[0-9.]* env_steps/s=[0-9.]*')"
}
for spec in "apple 8 $APPLE" "hamper 4 $HAMPER"; do
  set -- $spec; name=$1; n=$2; inst=$3
  log "timers $name n$n (warm app-data)"
  ( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u scripts/vec_eval/profile_build.py --instances $inst --num-envs $n --out $R/timers ) > $R/timers/${name}_n$n.log 2>&1
  log "rc=$? $(grep BUILD_PROFILE $R/timers/${name}_n$n.log | cut -c1-140)"
  grep ISG_VEC_BUILD_TIMER $R/timers/${name}_n$n.log > $R/timers/${name}_n$n.timers.txt
done
log "apple n1 before, warm (speed copy)"; run_one $SPEED $R/baseline_warm/apple_n1 $R/baseline_warm/apple_n1.log $APPLE 1
log "apple n1 after, warm"; run_one $NEXT $R/after_warm/apple_n1 $R/after_warm/apple_n1.log $APPLE 1
log "hamper n1 after, warm"; run_one $NEXT $R/after_warm/hamper_n1 $R/after_warm/hamper_n1.log $HAMPER 1
log "CHAIN3 DONE"
