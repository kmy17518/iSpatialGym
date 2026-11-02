#!/bin/bash
until grep -q "OFFSETS DONE" $HOME/vec_eval85/next/offsets.out 2>/dev/null; do sleep 20; done
NEXT=/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next
APPDATA=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
INST=$HOME/vec_eval85/instances/eval/inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table.json
( export ISG_EVAL_ROOT=$NEXT OMNIGIBSON_APPDATA_PATH=$APPDATA ISG_ROOMS_OVERRIDE=$NEXT/scripts/vec_eval/rooms/rooms_override.json; source $NEXT/scripts/vec_eval/env_lambda.sh >/dev/null 2>&1
  python -u scripts/vec_eval/probe_sleep.py --instances $INST --num-envs 2 --scene-offsets "0,64.6,0" --steps 300 --every 30 ) > $HOME/vec_eval85/next/probe/sleep_probe.log 2>&1
echo "PROBE_RC=$?" >> $HOME/vec_eval85/next/probe/sleep_probe.log
