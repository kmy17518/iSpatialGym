#!/bin/bash
# After the baseline: apply the vec_next overlay to the local Behavior-1K-vec-next checkout, profile the build at N=2, measure apple
# n1 / n8 after the change, then hamper (house_single_floor) before (the speed copy = $S state) and after. Sequential on the one 4090.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
SPEED=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-speed
R=$HOME/vec_eval85/next
log() { echo "[$(date -Is)] $*"; }
run_one() {  # run_one <eval root> <out dir> <log> <instance> <N>
  local root=$1 out=$2 lg=$3 inst=$4 n=$5
  ( export ISG_EVAL_ROOT=$root; source $root/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $inst --same-instance-all-slots --num-envs $n --layout grid-centered --scene-margin 5 \
      --wallclock 0 --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:chain --out $out ) > $lg 2>&1
  local rc=$?
  log "$(basename $out) rc=$rc $(grep -c RUN_ISG_RESULT $lg) results; $(grep -o 'world built in [0-9.]*s' $lg | head -1); $(grep RUN_ISG_VEC_BATCH $lg | tail -1 | grep -o 'batch_steps/s=[0-9.]* env_steps/s=[0-9.]*')"
}
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
log "apply overlay"; bash $NEXT/scripts/vec_eval/apply_b1k_overlay.sh $NEXT/Behavior-1K | cut -c1-200
log "profile n2 apple (after)"
( export ISG_EVAL_ROOT=$NEXT; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs 2 --out $R/profile ) > $R/profile/profile_n2.log 2>&1
log "profile rc=$? $(grep BUILD_PROFILE $R/profile/profile_n2.log | cut -c1-300)"
for n in 1 8; do log "apple n$n after"; run_one $NEXT $R/after/apple_n$n $R/after/apple_n$n.log $APPLE $n; done
for n in 1 8; do log "hamper n$n before (speed copy)"; run_one $SPEED $R/baseline/hamper_n$n $R/baseline/hamper_n$n.log $HAMPER $n; done
for n in 1 8; do log "hamper n$n after"; run_one $NEXT $R/after/hamper_n$n $R/after/hamper_n$n.log $HAMPER $n; done
log "CHAIN DONE"
