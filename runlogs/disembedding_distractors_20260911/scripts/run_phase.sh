#!/usr/bin/env bash
# Run the distractor validation (one phase) until every job has a result, re-launching after crashes (Isaac Sim
# segfaults; CUDA out-of-memory and scene-load failures exit with code 3).
#
# GPU sharing: the table-selection queue (runlogs/table_selection_20260911/scripts/queue_scenes.sh, ISG_TABLE_MAXP=2)
# keeps two OmniGibson processes on the RTX 4090 and counts them with `pgrep -f "python .*validate_tables.py"`.
# A third Isaac Sim process does not fit (24 GB). The launcher is therefore named dis_validate_tables.py so that the
# table queue counts it as one of its two slots, and this script only starts it while at most ONE table-selection
# process is alive (and >= ISG_DIS_MIN_FREE_MIB of GPU memory is free).
# Usage: run_phase.sh <render|validate> [extra validate_distractors.py args...]
set -u
PHASE="$1"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(cd "$HERE/.." && pwd)"
WORK="${ISG_DIS_WORK:-/tmp/isg_dis_distractors}"
PY="${ISG_PY:-$HOME/miniconda3/envs/ispatialgym/bin/python}"
MIN_FREE_MIB="${ISG_DIS_MIN_FREE_MIB:-9500}"
MAX_OTHERS="${ISG_DIS_MAX_OTHERS:-9}"
mkdir -p "$WORK/logs"
LOG="$WORK/logs/${PHASE}.log"
export OMNIGIBSON_HEADLESS=1
others() { pgrep -fc "^/home/minyeongk/miniconda3/envs/ispatialgym/bin/python .*validate_tables.py" ; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | tr '\n' ' ' ; }
# Other agents launch Isaac Sim processes on this GPU without checking memory (their next scene starts the moment the
# previous one exits, i.e. exactly when memory frees up). Launch only after the free memory has stayed above the
# threshold for STABLE_S seconds with no new GPU process appearing, so that a process that is still allocating gets
# its memory first (22:00: a simultaneous start with such a process ran this driver out of memory at scene load).
STABLE_S="${ISG_DIS_STABLE_S:-90}"
# ISG_DIS_WAIT_IDLE=1 (default): launch only when NO other GPU compute process exists (existing GPU jobs of other
# sessions run to completion first); 0: the memory / process-count gates alone.
WAIT_IDLE="${ISG_DIS_WAIT_IDLE:-1}"
for round in $(seq 1 12); do
  waited=0
  stable=0
  pids_prev="$(gpu_pids)"
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(others)
    pids_now="$(gpu_pids)"
    idle_ok=1
    if [[ "$WAIT_IDLE" == "1" && -n "${pids_now// /}" ]]; then idle_ok=0; fi
    if [[ "$idle_ok" == "1" && "$free" -ge "$MIN_FREE_MIB" && "$n" -le "$MAX_OTHERS" && "$pids_now" == "$pids_prev" ]]; then
      stable=$((stable + 15))
      if [[ "$stable" -ge "$STABLE_S" ]]; then break; fi
    else
      stable=0
    fi
    pids_prev="$pids_now"
    if (( waited % 600 == 0 )); then
      echo "$(date '+%F %T') waiting for a GPU slot (free ${free} MiB, ${n} validate_tables process(es) alive, GPU pids: ${pids_now})" | tee -a "$LOG"
    fi
    sleep 15; waited=$((waited + 15))
  done
  echo "=== $(date '+%F %T') $PHASE round $round: $*" | tee -a "$LOG"
  "$PY" "$HERE/dis_validate_tables.py" --phase "$PHASE" --out "$OUT" --work "$WORK" --skip-done "$@" >> "$LOG" 2>&1
  rc=$?
  echo "=== $(date '+%F %T') $PHASE round $round exit $rc" | tee -a "$LOG"
  if grep -q "^\[dis_distractors [0-9:]*\] done" <(tail -n 50 "$LOG"); then
    echo "=== $PHASE complete" | tee -a "$LOG"
    exit 0
  fi
  if [[ $rc -eq 0 ]]; then
    continue
  fi
  sleep 20
done
echo "=== $PHASE gave up after 12 rounds" | tee -a "$LOG"
exit 1
