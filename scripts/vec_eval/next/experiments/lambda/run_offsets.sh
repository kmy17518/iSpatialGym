#!/bin/bash
# Discriminating experiment for the small-object trembling: kielbasa (most sensitive alias), N=2, the same demo in both slots, the clone at
# explicit offsets: A (0,64.6,0) = the failing grid cell; B (0,0,30) = stacked above the origin scene (x/y as at the origin, physically
# separate); C (0,200,0) = far; D (-58.2,0,0) = the passing grid cell. Final overlay, warm app-data.
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next/offsets
INST=$HOME/vec_eval85/instances/eval/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table.json
log() { echo "[$(date -Is)] $*"; }
run() { local name=$1 offs=$2; rm -rf $VOUT/$name
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --same-instance-all-slots --num-envs 2 --scene-offsets "$offs" --wallclock 0 \
      --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:offsets:$name --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  log "$name offs=$offs rc=$? $(grep -c RUN_ISG_RESULT $VOUT/$name.log) results; $(grep -o 'world built in [0-9.]*s' $VOUT/$name.log | head -1); $(grep -o 'RUN_ISG_VEC_SCENE 1 placed at [^(]*' $VOUT/$name.log)"; }
run A_y64 "0,64.6,0"
run B_z30 "0,0,30"
run C_y200 "0,200,0"
run D_xm58 "-58.2,0,0"
log "OFFSETS DONE"
