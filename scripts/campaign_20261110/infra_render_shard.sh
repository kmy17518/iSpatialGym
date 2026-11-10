#!/bin/bash
# One owned GPU process renders accepted episodes; source data is read-only.
set -euo pipefail
CAMPAIGN_ROOT=${1:?campaign root}
RELEASE=${2:?immutable accepted release}
SHARD=${3:?render shard id}
SNAPSHOT=${4:?frozen rendering snapshot}
RENDER_OUT=${5:?new campaign rendering output}
LIMIT=${6:-}
export PYTHONPATH="$SNAPSHOT${PYTHONPATH:+:$PYTHONPATH}"
PREPARE=(python3 -m scripts.campaign_20261110.render_shard --root "$CAMPAIGN_ROOT" --release "$RELEASE" --shard "$SHARD" --snapshot "$SNAPSHOT" --output "$RENDER_OUT")
if [ -n "$LIMIT" ]; then PREPARE+=(--limit "$LIMIT"); fi
mapfile -d '' -t INPUTS < <("${PREPARE[@]}")
[ "${#INPUTS[@]}" -eq 2 ] || { echo 'Rendering preparation failed' >&2; exit 2; }
source "${INPUTS[0]}/job.env"
export ISGE_CODE="$SNAPSHOT"
export ISGF_JB_NFS="$CAMPAIGN_ROOT/jobs"
export KIT_TPL_NFS_ROOT="$CAMPAIGN_ROOT/kit_templates" KIT_TEX_STORE="$CAMPAIGN_ROOT/kit_textures"
export EXP_PATH="$CAMPAIGN_ROOT/isaacsim_overlay/apps"
source "$SNAPSHOT/scripts/trainfast/env_fast.sh"
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
set +e
python -m ispatialgym.multienv.replay --demos "@$RENDER_OUT/demos.txt" --campaign-manifest "$RELEASE/instances.jsonl" \
  --root "$RENDER_OUT/lerobot" --task "${INPUTS[1]}" --slots 1 --scene recorded --batched-usd-edits --async-flush \
  --proprio recorded --warmup-s 60 --repo-prefix replay --goal-occlusion auto --results "$RENDER_OUT/results.jsonl" \
  > "$RENDER_OUT/replay.log" 2>&1
RENDER_RC=$?
set -e
python -m scripts.campaign_20261110.render_shard --output "$RENDER_OUT" --returncode "$RENDER_RC"
KIT_RC=0
