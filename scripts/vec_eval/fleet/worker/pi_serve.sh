#!/bin/bash
# One openpi (pi05_b1k) checkpoint served on this GPU for run_isg_vec's --policy-sweep: fetch the step's eval tar from the
# Hugging Face repo (12 GB), extract it under /work/pi/ckpt/<run>/<step>, then exec serve_b1k.py.
#   pi_serve.sh <hf repo> <step> <port> <task name> [robot]
# The venv: $OPENPI_VENV (default /work/pi/venv, built by the worker with uv from the checkout at $OPENPI_DIR).
# Several slots of one host may serve the same or different steps at once: the download + extraction of a step runs under
# a per-step lock, the tar is checked against the uploader record's size and must extract cleanly (else re-downloaded, up to
# 3 times), a served step directory carries an in-use marker, and only unused step directories are removed for disk space
# (2026-10-24: two slots fetching one step into the same file produced a corrupt tar, and the cleanup of one slot removed
# the step another slot was serving).
set -euo pipefail
REPO=$1; STEP=$2; PORT=$3; TASK=$4; ROBOT=${5:-b1k/R1Pro}
OPENPI_DIR=${OPENPI_DIR:-/vision/u/minyeong/projects/openpi-todo1}
VENV=${OPENPI_VENV:-/work/pi/venv}
RUN=${REPO##*/}
PI_WORK=${ISG_PI_WORK:-/work/pi}          # scratch (downloads, extracted steps, locks); the cluster worker puts it under ISG_WORK
D=$PI_WORK/ckpt/$RUN/$STEP
TAR=$PI_WORK/dl/$RUN-$STEP.tar
NAME=eval/step-$(printf %08d $STEP).tar
URL="https://huggingface.co/$REPO/resolve/main/$NAME"
log() { echo "[pi_serve $(date -u +%FT%TZ)] $*"; }
# ISG_PI_CKPT_ROOT (cluster): the step already extracted on NFS by $V/ckpt/extract_pi05.sh (<root>/<run>/step-XXXXXXXX/.ready);
# nothing is downloaded, extracted or removed
if [ -n "${ISG_PI_CKPT_ROOT:-}" ]; then
  D=$ISG_PI_CKPT_ROOT/$RUN/step-$(printf %08d $STEP)
  [ -f "$D/.ready" ] || { log "no extracted checkpoint at $D (.ready missing): fetch + extract it first"; exit 4; }
  if [ -d "$D/params" ]; then CK=$D; else CK=$(ls -d $D/*/ | head -1); fi   # extract_pi05.sh lifts <STEP>/{params,assets} into the step dir
  log "serving $CK (NFS) on port $PORT task $TASK robot $ROBOT"
  cd $OPENPI_DIR
  export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$PI_WORK/openpi_data} HF_HUB_OFFLINE=1
  exec $VENV/bin/python scripts/b1k/serve_b1k.py --robot "$ROBOT" --task "b1k/$TASK" --policy.config pi05_b1k --policy.dir "${CK%/}" \
    --action-horizon "${PI_ACTION_HORIZON:-16}" --port "$PORT" ${PI_SERVE_EXTRA:-}
fi
mkdir -p $PI_WORK/dl $PI_WORK/ckpt/$RUN $PI_WORK/lock

fetch_and_extract() {
  # the expected size from the uploader record (0 when unknown)
  local want
  want=$(curl -sSL --max-time 60 "https://huggingface.co/$REPO/resolve/main/.b1k-uploader.json" | python3 -c "
import json,sys
try: print(int(json.load(sys.stdin)['eval']['$STEP']['size']))
except Exception: print(0)" 2>/dev/null || echo 0)
  local try
  for try in 1 2 3; do
    for dl in 1 2 3 4; do
      if curl -sSL --retry 5 --retry-delay 10 -C - -o "$TAR" "$URL"; then break; fi
      log "download retry $dl"; sleep 20
    done
    local have
    have=$(stat -c %s "$TAR" 2>/dev/null || echo 0)
    if [ "$want" -gt 0 ] && [ "$have" != "$want" ]; then
      log "tar size $have != expected $want (attempt $try): re-downloading"; rm -f "$TAR"; continue
    fi
    rm -rf "$D"; mkdir -p "$D"
    if tar -xf "$TAR" -C "$D"; then
      rm -f "$TAR"; touch "$D/.ready"
      log "step $STEP extracted ($(du -sh "$D" | cut -f1))"
      return 0
    fi
    log "tar extraction failed (attempt $try): re-downloading"; rm -f "$TAR"; rm -rf "$D"
  done
  log "step $STEP: no usable checkpoint after 3 attempts"
  return 1
}

# download + extraction of this step under its lock (another slot may be doing it right now)
exec 9> "$PI_WORK/lock/$RUN-$STEP.lock"
flock 9
if [ ! -f "$D/.ready" ]; then
  log "fetching $NAME"
  fetch_and_extract
fi
touch "$D/.inuse.$$"
flock -u 9
trap 'rm -f "$D/.inuse.$$"' EXIT
# step directories of the run nobody serves (no in-use marker younger than 6 h), for disk space
for other in $PI_WORK/ckpt/$RUN/*/; do
  [ "$(basename "$other")" = "$STEP" ] && continue
  if [ -z "$(find "$other" -maxdepth 1 -name '.inuse.*' -mmin -360 2>/dev/null | head -1)" ]; then rm -rf "$other"; fi
done
CK=$(ls -d $D/*/ | head -1)   # <STEP>/params + <STEP>/assets
log "serving $CK on port $PORT task $TASK robot $ROBOT"
cd $OPENPI_DIR
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$PI_WORK/openpi_data} HF_HUB_OFFLINE=1
# not exec: the trap must remove the in-use marker when the server ends
$VENV/bin/python scripts/b1k/serve_b1k.py --robot "$ROBOT" --task "b1k/$TASK" --policy.config pi05_b1k --policy.dir "${CK%/}" \
  --action-horizon "${PI_ACTION_HORIZON:-16}" --port "$PORT" ${PI_SERVE_EXTRA:-} &
SERVER=$!
trap 'kill -TERM $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; rm -f "$D/.inuse.$$"' EXIT TERM INT
wait $SERVER
