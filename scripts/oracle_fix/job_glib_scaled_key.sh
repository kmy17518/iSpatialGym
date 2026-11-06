#!/bin/bash
# oracle-fix 2026-11-06 (owner on the ontop_small#02 steak demo: "want to scale down steak so it can be grasped lying down & fits inside the
# floor of the frying pan"): ONE legacy grasp-library key at a spawn scale, made the way spatula__hkvsep__t076 / lid__pbmsji__s065 were
# (isg_home/eval85/scratch_spatula076/og_job.sh, 17815412: 4:21 on a TITAN RTX): the library's own scripts (runlogs/grasp_library_20260905/
# scripts) with this job's paths, in ONE job (one Kit process at a time in one kit_appdata folder), on the oracle-fix worktree's code:
#   export  the collision mesh at the spawn scale -> glib/meshes/<key>.{obj,json}                                  (Kit)
#   cands   gen_candidates.py a1 (CPU) -> glib/cands/<key>.yaml; the job fails (rc 3) unless the rest is GL_REST (canonical: lying)
#   ag      assisted-grasp attach + hold-under-gravity test, renders -> glib/results/<key>.json, glib/frames/          (Kit)
#   strips  make_model_strips.py -> glib/strips/<key>.png (the review's input)
#   GL_MODELS=$R/scripts/oracle_fix/glib_steak060/models_steak060.json sbatch -J isgof_glib_steak060 -A vision -p svl --exclude=svl9 \
#     --gres=gpu:titanrtx:1 --time=1:00:00 scripts/oracle_fix/job_glib_scaled_key.sh ["export cands ag strips"]
# Writes only $GL_WORK (default $OFH/glib_scaled/<key>) and the slurm log. Then (no simulator): the review, the files into
# runlogs/grasp_library_20260905/{cands,results,review}/ and the grasps.csv rows - scripts/oracle_fix/grasp_rows.py.
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/slurm/%x_%j.log
set -u
STEPS=${1:-export cands ag strips}
export OFH=/vision/u/minyeong/projects/isg_home/oracle_fix
source ${ISG_OF_ROOT:-/vision/u/minyeong/projects/ispatialgym-oracle-fix}/scripts/oracle_fix/env_cluster.sh   # R, the conda env, kit_appdata
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$OFH/jobs/*) rm -rf "$JB" ;; esac' EXIT
M=${GL_MODELS:?the models json (one job entry: key, category, model, scale, pipeline a1, constraint)}
KEY=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))[0]['key'])" $M)
W=${GL_WORK:-$OFH/glib_scaled/$KEY}
REST=${GL_REST:-canonical}
mkdir -p $W/glib/meshes $W/glib/cands $W/glib/results $W/glib/frames $W/glib/strips $W/glib/graspgen $W/scripts $W/slurm
cp $M $W/glib/models.json
cp $R/runlogs/grasp_library_20260905/test_pose.json $W/glib/test_pose.json
export GLIB_OUT=$W/glib JOBS=$W/glib/models.json
# the library's scripts with this job's paths - the only differences to isg_home/eval85/scratch_spatula076/scripts (their output dir, the
# dataset, the gripper sphere file and the Isaac grasp sampler of this env; the vectorized OmniGibson's env.robots[0][0], 6 frames per view)
SAMPLER=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.replicator.grasping/isaacsim/replicator/grasping/sampler_utils.py
for f in export_meshes_batch.py gen_candidates.py ag_validate_render.py make_model_strips.py; do
  sed -e 's#OUT = "/tmp/glib"#OUT = os.environ["GLIB_OUT"]#' \
      -e 's#EXP = "/tmp/grasp_exp"#EXP = "'$R'/runlogs/grasp_exp_20260904"#' \
      -e 's#"/tmp/grasp_exp/r1pro_right_gripper_eef.json"#"'$R'/runlogs/grasp_exp_20260904/r1pro_right_gripper_eef.json"#' \
      -e 's#ASSETS = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets/behavior-1k-assets/objects"#ASSETS = "'$R'/Behavior-1K/datasets/behavior-1k-assets/objects"#' \
      -e 's#^SAMPLER = ".*"$#SAMPLER = "'$SAMPLER'"#' \
      -e 's#robot = env.robots\[0\]; scene#robot = env.robots[0][0]; scene#' \
      -e 's#    for _ in range(3): og.sim.render()#    for _ in range(6): og.sim.render()#' \
      $R/runlogs/grasp_library_20260905/scripts/$f > $W/scripts/$f
done
if grep -n '/tmp/glib\|/tmp/grasp_exp\|/home/minyeongk' $W/scripts/*.py; then echo "ISGOF_GLIB_SCALED unpatched paths left in $W/scripts"; exit 2; fi
[ -f "$SAMPLER" ] || { echo "ISGOF_GLIB_SCALED no Isaac grasp sampler at $SAMPLER"; exit 2; }
GPU=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
echo "ISGOF_GLIB_SCALED $SLURM_JOB_ID key=$KEY steps=\"$STEPS\" host=$(hostname -s) gpu=\"$GPU\" isg=$(git -C $R rev-parse --short HEAD) b1k=$(git -C $R/Behavior-1K rev-parse --short HEAD) work=$W start=$(date -Is)"
for STEP in $STEPS; do
  t0=$(date +%s)
  case "$STEP" in
    export) python $W/scripts/export_meshes_batch.py > $W/slurm/export_${SLURM_JOB_ID}.log 2>&1 ;;
    cands)  python $W/scripts/gen_candidates.py a1 > $W/slurm/cands_${SLURM_JOB_ID}.log 2>&1 && \
            python -c "import sys, yaml; d = yaml.safe_load(open(sys.argv[1]))['ispatialgym']; print('rest', d.get('rest_pose'), d.get('rest_quat_wxyz')); sys.exit(0 if d.get('rest_pose') == sys.argv[2] else 3)" \
              $W/glib/cands/$KEY.yaml $REST ;;
    ag)     python $W/scripts/ag_validate_render.py 0 1 > $W/slurm/ag_${SLURM_JOB_ID}.log 2>&1 ;;
    strips) python $W/scripts/make_model_strips.py $KEY > $W/slurm/strips_${SLURM_JOB_ID}.log 2>&1 ;;
    *)      echo "unknown step $STEP"; exit 2 ;;
  esac
  rc=$?
  echo "ISGOF_GLIB_SCALED_STEP step=$STEP rc=$rc wall=$(( $(date +%s) - t0 ))s $(date -Is)"
  [ $rc -eq 0 ] || exit $rc
done
python -c "import json, sys; r = json.load(open(sys.argv[1])); print('ISGOF_GLIB_SCALED_RESULT', r.get('key'), 'n_valid', r.get('n_valid'), 'of', len(r.get('tested', [])), 'rest', (r.get('candidates_meta') or {}).get('rest_pose'))" \
  $W/glib/results/$KEY.json
echo "ISGOF_GLIB_SCALED_END $(date -Is)"
exit 0
