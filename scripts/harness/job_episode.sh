#!/bin/bash
# One harness episode: run_isg --mode policy with ispatialgym.harness.policy:HarnessPolicy on <episode dir>/instance and
# <episode dir>/harness_config.json (prepared by python -m ispatialgym.harness.run). Follows the cluster's Kit app-data rule
# (warm NFS template, KIT_RC); the datasets are linked into a per-job directory, so the checkout holds no dataset symlinks.
# Writes <episode dir>/{job.txt,run.log,code_paths.txt,run/<iid>/...}.
#   ISG_HARNESS_CODE=<checkout> sbatch [opts] scripts/harness/job_episode.sh <episode dir>
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
set -u
EP=$(realpath "$1")
R=${ISG_HARNESS_CODE:-/vision/u/minyeong/projects/ispatialgym-eval}
LIMIT=${ISG_HARNESS_TIMEOUT_S:-13800}
SHARED=${ISG_SHARED_DATASETS:-/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets}
KIT_RC=1
WT=$R
source /vision/u/minyeong/projects/cluster_tools/kit_appdata.sh
kit_appdata_setup
JB=/tmp/$USER/isgh_${SLURM_JOB_ID:-$$}
mkdir -p $JB 2>/dev/null || JB=$EP/job_tmp_${SLURM_JOB_ID:-$$}
mkdir -p $JB/datasets $JB/home
trap 'kit_appdata_teardown; rm -rf "$JB"' EXIT
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
for d in 2026-challenge-task-instances behavior-1k-assets omnigibson-robot-assets omnigibson.key; do ln -s $SHARED/$d $JB/datasets/$d; done
ln -s $R/ispatialgym/assets/datasets/ispatialgym-assets $JB/datasets/ispatialgym-assets
export PYTHONPATH=$R:$R/Behavior-1K/OmniGibson:$R/Behavior-1K/bddl3 PYTHONDONTWRITEBYTECODE=1
export OMNIGIBSON_DATA_PATH=$JB/datasets OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1
export OMNIGIBSON_TRACK_STATE_UPDATES=1 OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0
# the production / eval settings of isg_home/eval85/env.sh (exact opt-ins on, experimental switches off)
export ISPATIALGYM_FAST_GEOMETRY=1 ISPATIALGYM_FAST_SUPPORT=1 ISPATIALGYM_FAST_ASSIGN=1 ISPATIALGYM_SHARED_GOAL_EVAL=1 \
  ISPATIALGYM_GC_FREEZE=1 ISPATIALGYM_BATCHED_WORLD_SYNC=1 ISPATIALGYM_SHARD_FORMAT=legacy ISPATIALGYM_SEEDED_SPHERE_FIT=off \
  ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
unset ISPATIALGYM_EXECUTOR_MESH_CACHE ISPATIALGYM_PHYSICS_ONLY ISPATIALGYM_ASYNC_SHARD ISPATIALGYM_PER_SCENE_STATE ISPATIALGYM_ASYNC_FRAMES \
  ISPATIALGYM_RECORD_LINK_POSES ISPATIALGYM_SAMPLER_MESH_CACHE ISPATIALGYM_SAMPLER_STEP1_ONCE ISPATIALGYM_HEAD_CAMERA_ONLY ISG_PROGRAM_HOOK \
  ISPATIALGYM_LAZY_FRAMES ISPATIALGYM_MESH_CACHE ISPATIALGYM_RB_DUMP_WORLD ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO ISPATIALGYM_SHARD_NO_DEPTH \
  ISPATIALGYM_SHARD_RGB_ENCODER ISPATIALGYM_LAB_CACHE ISPATIALGYM_BAND_TRUNK_CHECK
export HOME=$JB/home XDG_CACHE_HOME=$JB/home/.cache TORCHINDUCTOR_CACHE_DIR=$JB/inductor TRITON_CACHE_DIR=$JB/triton NUMBA_CACHE_DIR=$JB/numba \
  HF_LEROBOT_HOME=$JB/lerobot_home ISPATIALGYM_REACH_BANDS=$JB/reach_bands ISPATIALGYM_CACHE_DIR=$JB/isg_cache \
  ISPATIALGYM_SAMPLING_CACHE=$JB/isg_cache/sampling ISPATIALGYM_HELD_HOLD_CACHE=$JB/isg_cache/held_hold.json ISPATIALGYM_REACH_VOLUME=$JB/reach_volume
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $NUMBA_CACHE_DIR $ISPATIALGYM_REACH_BANDS $ISPATIALGYM_CACHE_DIR $ISPATIALGYM_REACH_VOLUME
export ISG_HARNESS_CONFIG=$EP/harness_config.json
cd $R
# the conda env's editable installs point at another checkout: refuse to run unless PYTHONPATH wins for every package
python - "$R" > $EP/code_paths.txt 2>&1 <<'EOF'
import importlib.util, os, sys
root = os.path.realpath(sys.argv[1])
bad = 0
for name in ("ispatialgym", "omnigibson", "bddl"):
    origin = os.path.realpath(importlib.util.find_spec(name).origin)
    ok = origin.startswith(root + os.sep)
    bad += not ok
    print(f"{name} {origin} {'ok' if ok else 'OUTSIDE ' + root}")
sys.exit(1 if bad else 0)
EOF
[ $? -eq 0 ] || { echo "CODE_PATH_CHECK_FAILED"; cat $EP/code_paths.txt; exit 4; }
tro=$(ls $EP/instance/instances/*-tro_state.json 2>/dev/null | head -1)
[ -n "$tro" ] || { echo "no instance under $EP/instance"; exit 3; }
STEPS=$(python -c "import json; print(int(json.load(open('$EP/harness_config.json')).get('max_steps', 18000)) + 50)")
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "ISG_HARNESS_EPISODE ${SLURM_JOB_ID:-local} ep=$EP host=$(hostname -s) gpu=\"$GPU\" isg=$(git -C $R rev-parse --short HEAD) b1k=$(git -C $R/Behavior-1K rev-parse --short=9 HEAD) limit_s=$LIMIT start=$(date -Is)" | tee $EP/job.txt
t0=$(date +%s)
timeout $LIMIT python -m ispatialgym.oracle.run_isg --instance "$tro" --mode policy --policy ispatialgym.harness.policy:HarnessPolicy \
  --max-steps $STEPS --out $EP/run --record video --record-view head,external --video-size 540 --no-goal-override > $EP/run.log 2>&1
KIT_RC=$?
echo "RUN_EXIT $KIT_RC wall_s=$(( $(date +%s) - t0 ))" | tee -a $EP/job.txt
