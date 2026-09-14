#!/usr/bin/env bash
# Keep at most $MAXP validate_tables.py processes running (GPU memory: ~7.3 GB each on the RTX 4090) and feed them scenes.
# Usage: queue_scenes.sh "<scene1> <scene2> ..." [extra validate_tables.py args...]
set -u
SCENES="$1"; shift
MAXP="${ISG_TABLE_MAXP:-3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${ISG_TABLE_WORK:-/tmp/isg_table_sel}"
mkdir -p "$WORK/logs"
MINFREE_MB="${ISG_TABLE_MINFREE_MB:-9000}"  # a fresh process needs ~7.5-8.5 GB; other agents share the GPU
count() { pgrep -fc "^/home/minyeongk/miniconda3/envs/ispatialgym/bin/python .*validate_tables.py" ; }
freemb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
for s in $SCENES; do
  while [[ "$(count)" -ge "$MAXP" || "$(freemb)" -lt "$MINFREE_MB" ]]; do sleep 60; done
  echo "$(date '+%F %T') launching $s ($*)"
  nohup "$HERE/run_scene.sh" "$s" "$@" > "$WORK/logs/${s}$( [[ " $* " == *" --under "* ]] && echo _under ).wrapper.log" 2>&1 &
  sleep 150  # let the new process allocate its GPU memory before counting again
done
echo "$(date '+%F %T') queue drained"
