#!/bin/bash
# Existing pinned worker CPU set and source-native physics; no sampling or allocation.
set -euo pipefail
JOBDIR=${1:?transport job directory}
source "$JOBDIR/job.env"
python3 "$ISGF_REPLAY_HELPER_SNAPSHOT/scripts/campaign_20261110/vast_replay.py" --verify "$JOBDIR"
export ISGF_REPLAY_BACKEND=vast
export ISGF_REPLAY_SCENE_HELPER="$ISGF_REPLAY_HELPER_SNAPSHOT/scripts/campaign_20261110/infra_replay_scene.py"
export ISGF_REPLAY_GATE_HELPER="$ISGF_REPLAY_HELPER_SNAPSHOT/scripts/campaign_20261110/replay_gate_quality.py"
exec bash "$ISGF_REPLAY_HELPER_SNAPSHOT/scripts/campaign_20261110/infra_open_loop.sh" \
  "$ISGF_REPLAY_SOURCE_JOB" "$ISGF_REPLAY_SOURCE_SNAPSHOT" "$JOBDIR/gate" "${ISGF_REPLAY_INSTANCE_ID:-}"
