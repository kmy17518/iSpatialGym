#!/bin/bash
# Chain 2 (after the NameError fix): apple n1 / n8 after, hamper before (speed copy) / after, then a py-spy build profile at N=8.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
SPEED=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-speed
R=$HOME/vec_eval85/next
PYSPY=$HOME/miniconda3/envs/ispatialgym-vectorized-envs/bin/py-spy
log() { echo "[$(date -Is)] $*"; }
run_one() {  # run_one <eval root> <out dir> <log> <instance> <N>
  local root=$1 out=$2 lg=$3 inst=$4 n=$5
  rm -rf $out
  ( export ISG_EVAL_ROOT=$root; source $root/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $inst --same-instance-all-slots --num-envs $n --layout grid-centered --scene-margin 5 \
      --wallclock 0 --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:chain --out $out ) > $lg 2>&1
  local rc=$?
  log "$(basename $out) rc=$rc $(grep -c RUN_ISG_RESULT $lg) results; $(grep -o 'world built in [0-9.]*s' $lg | head -1); $(grep RUN_ISG_VEC_BATCH $lg | tail -1 | grep -o 'batch_steps/s=[0-9.]* env_steps/s=[0-9.]*')"
}
APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
HAMPER=$HOME/vec_eval85/instances/eval/camera_relocalization-obstructed-hamper_1.json
for n in 1 8; do log "apple n$n after"; run_one $NEXT $R/after/apple_n$n $R/after/apple_n$n.log $APPLE $n; done
for n in 1 8; do log "hamper n$n before (speed copy)"; run_one $SPEED $R/baseline/hamper_n$n $R/baseline/hamper_n$n.log $HAMPER $n; done
for n in 1 8; do log "hamper n$n after"; run_one $NEXT $R/after/hamper_n$n $R/after/hamper_n$n.log $HAMPER $n; done
log "py-spy build profile apple n8 (after)"
( export ISG_EVAL_ROOT=$NEXT; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
  $PYSPY record --rate 50 --format raw --nonblocking -o $R/profile/pyspy_build_n8.txt -- python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs 8 --out $R/profile ) > $R/profile/pyspy_n8.log 2>&1
log "py-spy rc=$? $(grep BUILD_PROFILE $R/profile/pyspy_n8.log | cut -c1-200)"
log "CHAIN2 DONE"
