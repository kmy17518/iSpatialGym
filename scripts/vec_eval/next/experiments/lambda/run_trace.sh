#!/bin/bash
# Who keeps the stack awake? Failing configuration (kielbasa N=2, clone at z=30) with ISG_SLEEP_TRACE: T0 stock; T1 rigid 1e-3; T2 rigid 5e-3 (free objects only); T3 global 5e-3 (every entity incl. the robot).
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next/trace; mkdir -p $VOUT
INST=$HOME/vec_eval85/instances/eval/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table.json
log() { echo "[$(date -Is)] $*"; }
run() { local name=$1; shift; rm -rf $VOUT/$name
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json ISG_SLEEP_TRACE=1 ISG_SLEEP_TRACE_EVERY=30; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --same-instance-all-slots --num-envs 2 --scene-offsets "0,0,30" --wallclock 0 \
      --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:trace:$name "$@" --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  log "$name $* rc=$? $(grep -c RUN_ISG_RESULT $VOUT/$name.log) results"; }
run T0_stock
run T1_rigid1e3 --physics-override rigid_sleep_threshold=0.001
run T2_rigid5e3 --physics-override rigid_sleep_threshold=0.005
run T3_global5e3 --physics-override sleep_threshold=0.005
log "TRACE DONE"
