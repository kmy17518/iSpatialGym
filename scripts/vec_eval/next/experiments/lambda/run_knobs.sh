#!/bin/bash
# Physics knobs on the FAILING configuration (kielbasa, N=2, clone stacked at z=30): K0 repeat (determinism), K1 sleep threshold 0.005 (PhysX
# default; stock 5e-5), K2 stabilization on with threshold 1e-3, K3 128 solver position iterations (stock 32), K4 8 velocity iterations (stock 1).
set -u
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
VOUT=$HOME/vec_eval85/next/knobs; mkdir -p $VOUT
INST=$HOME/vec_eval85/instances/eval/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table.json
log() { echo "[$(date -Is)] $*"; }
run() { local name=$1; shift; rm -rf $VOUT/$name
  ( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
    python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances $INST --same-instance-all-slots --num-envs 2 --scene-offsets "0,0,30" --wallclock 0 \
      --record-view tiled --grid-video --tile-px 240 --external-size 240 --tag next:knobs:$name "$@" --out $VOUT/$name ) > $VOUT/$name.log 2>&1
  log "$name $* rc=$? $(grep -c RUN_ISG_RESULT $VOUT/$name.log) results; $(grep -o 'world built in [0-9.]*s' $VOUT/$name.log | head -1); $(grep -o 'physics_overrides[^}]*}' $VOUT/$name.log | head -1 | cut -c1-160)"; }
run K0_repeat
run K1_sleep --physics-override sleep_threshold=0.005
run K2_stab --physics-override stabilization=0.001
run K3_iters128 --physics-override solver_position_iterations=128
run K4_veliters8 --physics-override solver_velocity_iterations=8
log "KNOBS DONE"
