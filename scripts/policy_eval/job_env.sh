# Environment of a policy-eval Slurm job (minyeong or shiyuc): the batched-replay lane's Kit / overlay setup (cluster_lane.sh), the
# code snapshot ISGE_CODE (= $CODE), the v2 eval dataset. Sourced by job scripts; sets JB (job-local scratch) via env_fast.sh.
# (env_fast.sh reassigns R: the paths below use ISGEV_* names.)
ISGEV_CAMP=/vision/u/minyeong/projects/isg_home/campaign_20261110
ISGEV_REC=$ISGEV_CAMP/eval_policy_r1
export ISGE_CODE=${CODE:?code dir}
export ISGF_JB_NFS=${JBNFS:-$ISGEV_REC/jobs}
export KIT_TPL_NFS_ROOT=$ISGEV_CAMP/kit_templates KIT_TEX_STORE=$ISGEV_CAMP/kit_textures
export EXP_PATH=$ISGEV_CAMP/training_generation/batched_replay_r1/isaacsim_overlay_readable/apps
export OPTIX_CACHE_PATH=/tmp/isgeval_optix_${SLURM_JOB_ID:-local}
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
source "$ISGE_CODE/scripts/trainfast/env_fast.sh"
export KIT_KEY KIT_TPL_NFS_ROOT KIT_TEX_STORE
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"
rm -rf "$JB/isaacsim_overlay"; cp -rP "$(dirname "$EXP_PATH")" "$JB/isaacsim_overlay"; chmod -R u+w "$JB/isaacsim_overlay"
export EXP_PATH=$JB/isaacsim_overlay/apps
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
# eval lighting = the batched replay's (stock lights; collection-time lighting draws of promoted train demos ignored)
export ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING=${ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING:-1}
# each instance's exact collection-time BDDL problem (scripts/policy_eval/build_bddl_overlay.py; the v2 export's ids are the templates' own)
export ISPATIALGYM_EVAL_BDDL_OVERLAY=${ISPATIALGYM_EVAL_BDDL_OVERLAY:-$ISGEV_REC/inputs/bddl_overlay_v2}
export ISPATIALGYM_INSTANCES_PATH=${ISPATIALGYM_INSTANCES_PATH:-$ISGEV_CAMP/training_generation/batched_replay_r1/merged/v2/eval-layout/ispatialgym-eval-instances}
for _p in "$ISPATIALGYM_EVAL_BDDL_OVERLAY/overlay.json" "$ISPATIALGYM_INSTANCES_PATH/metadata/instances.jsonl"; do
  [ -f "$_p" ] || { echo "JOB_ENV_MISSING $_p"; exit 5; }
done
# ISGEV_DEFAULTS_ONLY=1: no overlay / lighting switch - the dataset (files.bddl_root, eval_lighting) and the evaluator's defaults alone
if [ "${ISGEV_DEFAULTS_ONLY:-0}" = 1 ]; then unset ISPATIALGYM_EVAL_BDDL_OVERLAY ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING; fi
echo "JOB_ENV overlay=$ISPATIALGYM_EVAL_BDDL_OVERLAY instances=$ISPATIALGYM_INSTANCES_PATH"
KIT_RC=1
