#!/bin/bash
# Batched replay of one template's demos on a cluster GPU (Slurm job), the same command the Vast worker runs.
#   CODE=<code dir> LIST=<demo list> OUT=<out dir> PURPOSE=train|eval [EXTRA="..."] sbatch ... cluster_replay.sh
# train: --scene-pert remove --lighting-pert replay (camera relocalization: no replay-time removal, lighting yes); eval: both off.
set -uo pipefail
R=/vision/u/minyeong/projects/isg_home/campaign_20261110
RV=$R/training_generation/batched_replay_r1
export ISGE_CODE=${CODE:?code dir}
LIST=${LIST:?demo list}; OUT=${OUT:?out dir}; PURPOSE=${PURPOSE:-train}
export ISGF_JB_NFS=$RV/jobs
export KIT_TPL_NFS_ROOT=$R/kit_templates KIT_TEX_STORE=$R/kit_textures
export EXP_PATH=$R/isaacsim_overlay/apps
export OPTIX_CACHE_PATH=/tmp/isgrend_optix_${SLURM_JOB_ID:-local}
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache" ISPATIALGYM_SHARD_FORMAT=legacy
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
trap 'kit_appdata_teardown' EXIT
mkdir -p "$OUT"
case $PURPOSE in
  train) PERT="--scene-pert remove --scene-pert-plan auto --lighting-pert replay --goal-occlusion auto" ;;
  eval)  PERT="--scene-pert off --lighting-pert off --goal-occlusion off" ;;
  *) echo "PURPOSE train|eval"; exit 2 ;;
esac
echo "=== $(date -u +%FT%TZ) $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) purpose=$PURPOSE"
python -m ispatialgym.multienv.replay --demos "@$LIST" --root "$OUT/lerobot" --task "${TASK:-isg}" --repo-prefix "${PREFIX:-replay}" \
  --slots 1 --batched-usd-edits --async-flush --proprio loaded --warmup-s 15 --scene full \
  --image-size 224 --settle-pad 0.2 --settle-pad-min 50 --goal-images on ${INSTANCE_META:+--instance-meta $INSTANCE_META} \
  --results "$OUT/results.jsonl" $PERT ${EXTRA:-} > "$OUT/replay.log" 2>&1
RC=$?
echo "=== rc=$RC $(date -u +%FT%TZ)"
exit $RC
