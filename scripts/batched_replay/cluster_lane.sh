#!/bin/bash
# One Slurm GPU job of a batched-replay cluster lane (minyeong or shiyuc): env of cluster_replay.sh, then cluster_lane.py.
#   sbatch ... --export=ALL,CODE=<code dir>,RUN=<run>,LANE=<lane dir>,JBNFS=<writable NFS dir> cluster_lane.sh
set -uo pipefail
R=/vision/u/minyeong/projects/isg_home/campaign_20261110
export ISGE_CODE=${CODE:?code dir}
export ISGF_JB_NFS=${JBNFS:?writable NFS fallback dir}
export KIT_TPL_NFS_ROOT=$R/kit_templates KIT_TEX_STORE=$R/kit_textures
# a world-readable copy of $R/isaacsim_overlay (its .kit app files are minyeong-only; shiyuc's Kit could not start)
export EXP_PATH=$R/training_generation/batched_replay_r1/isaacsim_overlay_readable/apps
export OPTIX_CACHE_PATH=/tmp/isgrend_optix_${SLURM_JOB_ID:-local}
# shiyuc: git refuses minyeong's repos ("dubious ownership") and the Kit key would lose its Behavior-1K commit (cold starts)
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
export KIT_KEY KIT_TPL_NFS_ROOT KIT_TEX_STORE
# cache deltas after every unit (cluster_lane.py): minyeong into the shared stores, shiyuc into its own (promoted by finisher.py)
if [ "$USER" = minyeong ]; then export ISGREND_TEX_OUT=$KIT_TEX_STORE ISGREND_TPL_OUT=$KIT_TPL_NFS_ROOT
else export ISGREND_TEX_OUT=/vision/u/shiyuc/minyeong/isgrend/cache_delta/kit_textures ISGREND_TPL_OUT=/vision/u/shiyuc/minyeong/isgrend/cache_delta/kit_templates; fi
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache" ISPATIALGYM_SHARD_FORMAT=legacy ISGREND_SCRATCH=$JB/isgrend
# OmniGibson writes its .kit app file into $EXP_PATH at every start: a job-local copy of the overlay (symlinks + small files)
# (cp -rP: cp -a cannot preserve NFS attributes on /tmp)
rm -rf "$JB/isaacsim_overlay"; cp -rP "$(dirname "$EXP_PATH")" "$JB/isaacsim_overlay"; chmod -R u+w "$JB/isaacsim_overlay"
[ -w "$JB/isaacsim_overlay/apps" ] || { echo "OVERLAY_COPY_FAILED $JB/isaacsim_overlay"; exit 4; }
export EXP_PATH=$JB/isaacsim_overlay/apps
echo "EXP_PATH $EXP_PATH"
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
# minyeong publishes the Kit template / converted textures to the shared stores at the end; shiyuc only reads them
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
echo "=== $(date -u +%FT%TZ) $(hostname) $USER $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) lane=$LANE"
python -u "$ISGE_CODE/scripts/batched_replay/cluster_lane.py" --run "${RUN:?run}" --lane "${LANE:?lane}" --units-file "${UNITS_FILE:?units file}"
RC=$?
[ "$USER" = minyeong ] && [ $RC = 0 ] && KIT_RC=0
exit $RC
