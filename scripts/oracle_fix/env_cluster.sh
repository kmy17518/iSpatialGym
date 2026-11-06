# Stanford svl / viscam: environment of the oracle-fix worktree (sourced by job_val.sh). A copy of trainfast/scripts/env_fast.sh (the
# production train env) with R = the oracle-fix worktree; writes only the job's own directories and $OFH. Reach bands are COPIED into the
# job (the committed runlog bands, the band session's store, eval85's bands_eval, train85 / trainfast bands_train of the alias): the
# sources are read only, never written (ISPATIALGYM_REACH_BANDS is job-local).
R=${ISG_OF_ROOT:-/vision/u/minyeong/projects/ispatialgym-oracle-fix}
OFH=${OFH:-/vision/u/minyeong/projects/isg_home/oracle_fix}
E=/vision/u/minyeong/projects/isg_home/eval85
if [ -n "${SLURM_JOB_ID:-}" ]; then
  WT=$R
  source /vision/u/minyeong/projects/cluster_tools/kit_appdata.sh
  kit_appdata_setup
fi
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
export PYTHONPATH=$R:$R/Behavior-1K/OmniGibson:$R/Behavior-1K/bddl3
export OMNIGIBSON_DATA_PATH=$R/Behavior-1K/datasets
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1 OMNIGIBSON_TRACK_STATE_UPDATES=1
export OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1
export ISPATIALGYM_FAST_GEOMETRY=1 ISPATIALGYM_FAST_SUPPORT=1 ISPATIALGYM_FAST_ASSIGN=1 ISPATIALGYM_SHARED_GOAL_EVAL=1 \
  ISPATIALGYM_GC_FREEZE=1 ISPATIALGYM_BATCHED_WORLD_SYNC=1
export ISPATIALGYM_SHARD_FORMAT=legacy ISPATIALGYM_SEEDED_SPHERE_FIT=off
unset ISPATIALGYM_EXECUTOR_MESH_CACHE ISPATIALGYM_PHYSICS_ONLY ISPATIALGYM_ASYNC_SHARD ISPATIALGYM_PER_SCENE_STATE ISPATIALGYM_ASYNC_FRAMES \
  ISPATIALGYM_RECORD_LINK_POSES ISPATIALGYM_SAMPLER_MESH_CACHE ISPATIALGYM_SAMPLER_STEP1_ONCE ISPATIALGYM_HEAD_CAMERA_ONLY ISG_PROGRAM_HOOK \
  ISPATIALGYM_LAZY_FRAMES ISPATIALGYM_MESH_CACHE ISPATIALGYM_RB_DUMP_WORLD ISPATIALGYM_RECORD_BEHAVIOR_PROPRIO ISPATIALGYM_SHARD_NO_DEPTH \
  ISPATIALGYM_SHARD_RGB_ENCODER ISPATIALGYM_LAB_CACHE ISPATIALGYM_BAND_TRUNK_CHECK ISPATIALGYM_GRASP_LIBRARY_V2 ISPATIALGYM_REACH_MAPS
export NUMPY_MADVISE_HUGEPAGE=0 ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ "$(df --output=avail -k /tmp 2>/dev/null | tail -1)" -gt 30000000 ] 2>/dev/null && mkdir -p /tmp/$USER/isgof_$SLURM_JOB_ID 2>/dev/null; then
    JB=/tmp/$USER/isgof_$SLURM_JOB_ID
  else
    JB=$OFH/jobs/$SLURM_JOB_ID
  fi
  mkdir -p $JB/home
  export HOME=$JB/home XDG_CACHE_HOME=$JB/home/.cache TORCHINDUCTOR_CACHE_DIR=$JB/inductor TRITON_CACHE_DIR=$JB/triton HF_LEROBOT_HOME=$JB/lerobot_home
  export ISPATIALGYM_REACH_BANDS=$JB/reach_bands ISPATIALGYM_CACHE_DIR=$JB/isg_cache ISPATIALGYM_SAMPLING_CACHE=$JB/isg_cache/sampling \
    ISPATIALGYM_HELD_HOLD_CACHE=$JB/isg_cache/held_hold.json ISPATIALGYM_REACH_VOLUME=$JB/reach_volume NUMBA_CACHE_DIR=$JB/numba_cache
  mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $ISPATIALGYM_REACH_BANDS $ISPATIALGYM_CACHE_DIR $ISPATIALGYM_REACH_VOLUME
  cp --preserve=timestamps $R/runlogs/reach_bands_20260915/bands/* $ISPATIALGYM_REACH_BANDS/ 2>/dev/null
  cp -n --preserve=timestamps /vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/reach_bands/* $ISPATIALGYM_REACH_BANDS/ 2>/dev/null
  cp -n --preserve=timestamps $E/bands_eval/*.json $E/bands_eval/*.npz $ISPATIALGYM_REACH_BANDS/ 2>/dev/null
  # oracle-fix 2026-11-04 (owner item 5): floor bands computed with the COPY's grasp library ($OFH/glib_v2), persisted by job_val.sh
  # PERSIST_BANDS; first so that a band of the same name from the stores above does not win
  [ -n "${OF_BANDS:-}" ] && cp --preserve=timestamps $OF_BANDS/*.json $OF_BANDS/*.npz $ISPATIALGYM_REACH_BANDS/ 2>/dev/null
  if [ -n "${ALIAS_KEY:-}" ]; then
    for B in /vision/u/minyeong/projects/isg_home/train85/bands_train/$ALIAS_KEY /vision/u/minyeong/projects/isg_home/trainfast/bands_train/$ALIAS_KEY; do
      [ -d "$B" ] && cp -n --preserve=timestamps $B/* $ISPATIALGYM_REACH_BANDS/ 2>/dev/null
    done
  fi
  cp --preserve=timestamps /vision/u/minyeong/projects/isg_home/.cache/ispatialgym-vecenv/reach_volume/* $ISPATIALGYM_REACH_VOLUME/ 2>/dev/null
  NCPU=${SLURM_CPUS_PER_TASK:-8}
  export MKL_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU
fi
cd $R
