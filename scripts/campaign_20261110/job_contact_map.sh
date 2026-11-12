#!/bin/bash
# Exact positive contact library -> ContactMap only; no trajectory generation.
set -euo pipefail
JOBDIR=${1:?transport job directory}
source "$JOBDIR/job.env"
export ISGE_CODE ISGF_ROOT ISGF_NATIVE_HELPER
export PYTHONPATH="$ISGF_NATIVE_HELPER"
python3 -m scripts.campaign_20261110.vast_contact_map --verify "$JOBDIR"
export ISGF_JB_NFS="$ISGF_ROOT/jobs"
export KIT_TPL_NFS_ROOT=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_templates
export KIT_TEX_STORE=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_textures
unset ISGF_EXP_PATH EXP_PATH
set +u
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
ISGF_GRASPS=v2
source "$ISGE_CODE/scripts/trainfast/fast_env.sh"
set -u
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
python "$ISGF_NATIVE_HELPER/scripts/campaign_20261110/vast_contact_map.py" --run "$JOBDIR"
KIT_RC=0
