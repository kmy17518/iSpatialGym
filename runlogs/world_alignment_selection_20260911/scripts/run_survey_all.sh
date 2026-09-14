#!/usr/bin/env bash
# Run survey_walls.py for a list of scenes, one after the other, restarting a scene after a crash (Isaac Sim segfaults)
# until every room has a result. Usage: run_survey_all.sh "<scene1> <scene2> ..." [extra survey_walls.py args...]
set -u
SCENES="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/.." && pwd)"
WORK="${ISG_WALL_WORK:-/tmp/isg_wall_survey}"
PY="${ISG_PY:-$HOME/miniconda3/envs/ispatialgym/bin/python}"
mkdir -p "$WORK/logs"
export OMNIGIBSON_HEADLESS=1
# the GPU is shared with the table-selection validators (~8 GB each); a survey process needs ~6-11 GB
wait_for_gpu() {
  local need="${ISG_WALL_GPU_FREE_MIB:-10000}"
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ -z "$free" || "$free" -ge "$need" ]] && return 0
    echo "$(date '+%F %T') waiting for GPU memory ($free MiB free < $need)"; sleep 60
  done
}
# never run two survey processes at once (a previous wrapper's process may still be finishing a scene)
while pgrep -f "survey_walls.py --scene" > /dev/null; do sleep 30; done
for s in $SCENES; do
  LOG="$WORK/logs/${s}.log"
  for round in 1 2 3 4; do
    wait_for_gpu
    # a room left '.inprogress' by a crashed process is retried (removed marker), not recorded as crashed
    rm -f "$OUT/results/${s}__"*.json.inprogress
    echo "=== $(date '+%F %T') $s round $round: $*" | tee -a "$LOG"
    "$PY" "$HERE/survey_walls.py" --scene "$s" --out "$OUT" --work "$WORK" --skip-done "$@" >> "$LOG" 2>&1
    rc=$?
    echo "=== $(date '+%F %T') $s round $round exit $rc" | tee -a "$LOG"
    if [[ $rc -eq 0 ]]; then break; fi
    sleep 5
  done
done
echo "=== $(date '+%F %T') all scenes done"
