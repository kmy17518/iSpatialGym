#!/bin/bash
# One Slurm GPU job of a batched-replay cluster lane (minyeong or shiyuc): env of cluster_replay.sh, then cluster_lane.py.
#   sbatch ... --export=ALL,CODE=<code dir>,RUN=<run>,LANE=<lane dir>,JBNFS=<writable NFS dir> cluster_lane.sh
set -uo pipefail
R=/vision/u/minyeong/projects/isg_home/campaign_20261110
export ISGE_CODE=${CODE:?code dir}
export ISGF_JB_NFS=${JBNFS:?writable NFS fallback dir}
export KIT_TPL_NFS_ROOT=$R/kit_templates KIT_TEX_STORE=$R/kit_textures
export EXP_PATH=$R/isaacsim_overlay/apps
export OPTIX_CACHE_PATH=/tmp/isgrend_optix_${SLURM_JOB_ID:-local}
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache" ISPATIALGYM_SHARD_FORMAT=legacy ISGREND_SCRATCH=$JB/isgrend
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
# minyeong publishes the Kit template / converted textures to the shared stores at the end; shiyuc only reads them
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
echo "=== $(date -u +%FT%TZ) $(hostname) $USER $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) lane=$LANE"
python -u "$ISGE_CODE/scripts/batched_replay/cluster_lane.py" --run "${RUN:?run}" --lane "${LANE:?lane}"
RC=$?
[ "$USER" = minyeong ] && [ $RC = 0 ] && KIT_RC=0
exit $RC
