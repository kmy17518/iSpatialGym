#!/bin/bash
# After the variant probes: copy the play sub-timers in, time an apple N=8 build, then a speedscope (time-ordered) py-spy of apple N=4.
set -u
until grep -q "VARIANT PROBE DONE" $HOME/vec_eval85/next/restore_probe_var.out 2>/dev/null; do sleep 30; done
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
O=$NEXT/scripts/vec_eval/b1k_overlay/OmniGibson/omnigibson; B=/home/minyeongk/Desktop/projects/Behavior-1K-vec-next-6a6194600/OmniGibson/omnigibson
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
V=$HOME/vec_eval85/next/options; APPLE=$HOME/vec_eval85/instances/eval/object_scaling-apple.json
log() { echo "[$(date -Is)] $*"; }
cp $O/simulator.py $B/simulator.py; log "simulator.py copied $(md5sum $B/simulator.py | cut -c1-8)"
( export ISG_EVAL_ROOT=$NEXT ISG_VEC_BUILD_TIMERS=1 OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
  python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs 8 --no-profile --out $V ) > $V/apple_n8_playtimers.log 2>&1
log "playtimers rc=$? $(grep -o 'build_s=[0-9.]*' $V/apple_n8_playtimers.log | head -1)"; grep "ISG_VEC_BUILD_TIMER sim.play\|ISG_VEC_BUILD_TIMER sim.import_play\|ISG_VEC_BUILD_TIMER env.final_play" $V/apple_n8_playtimers.log
( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1; cd $NEXT
  py-spy record -r 50 --format speedscope -o $V/pyspy_apple_n4.speedscope.json -- python -u scripts/vec_eval/profile_build.py --instances $APPLE --num-envs 4 --no-profile --out $V ) > $V/pyspy_apple_n4_ss.log 2>&1
log "pyspy speedscope rc=$?"
log "PROFILE2 DONE"
