#!/bin/bash
# Exact native prefix only: no instance publication, demo, or acceptance.
set -euo pipefail
JOBDIR=${1:?private probe job directory}
source "$JOBDIR/job.env"
export ISGE_CODE ISGF_ROOT ISGF_NATIVE_HELPER
python3 "$ISGF_NATIVE_HELPER/scripts/campaign_20261110/vast_ik_probe.py" --verify "$JOBDIR"
export ISGF_JB_NFS="$ISGF_ROOT/jobs"
export KIT_TPL_NFS_ROOT=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_templates
export KIT_TEX_STORE=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_textures
unset ISGF_EXP_PATH EXP_PATH
set +u
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
ISGF_GRASPS=v2
source "$ISGE_CODE/scripts/trainfast/fast_env.sh"
set -u
set -a; source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' "$JOBDIR/job.env"); set +a
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
python "$ISGF_NATIVE_HELPER/scripts/campaign_20261110/vast_ik_probe.py" --run "$JOBDIR"
KIT_RC=0
