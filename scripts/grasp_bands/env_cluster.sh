# Stanford svl / viscam: environment of the grasp-bands worktree (sourced by job_glib.sh). Writes only the job's own directories and $GBH.
R=${ISG_GB_ROOT:-/vision/u/minyeong/projects/ispatialgym-grasp-bands}
GBH=${GBH:-/vision/u/minyeong/projects/isg_home/grasp_bands}
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
export OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0 ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
# vec_next's build speed-ups (exact): asset preparation / mesh caches, and the persistent content-keyed geometry cache (one shard per process)
export ISG_VEC_PREPARE_CACHE=1 ISG_VEC_MESH_CACHE=1 ISG_VEC_ASSET_CACHE_DIR=$GBH/asset_cache ISG_VEC_ASSET_CACHE_MODE=rw
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ "$(df --output=avail -k /tmp 2>/dev/null | tail -1)" -gt 30000000 ] 2>/dev/null && mkdir -p /tmp/$USER/isggb_$SLURM_JOB_ID 2>/dev/null; then
    JB=/tmp/$USER/isggb_$SLURM_JOB_ID
  else
    JB=$GBH/jobs/$SLURM_JOB_ID
  fi
  mkdir -p $JB/home
  export HOME=$JB/home XDG_CACHE_HOME=$JB/home/.cache TORCHINDUCTOR_CACHE_DIR=$JB/inductor TRITON_CACHE_DIR=$JB/triton
  export ISPATIALGYM_REACH_BANDS=$JB/reach_bands ISPATIALGYM_CACHE_DIR=$JB/isg_cache ISPATIALGYM_SAMPLING_CACHE=$JB/isg_cache/sampling \
    ISPATIALGYM_HELD_HOLD_CACHE=$JB/isg_cache/held_hold.json ISPATIALGYM_REACH_VOLUME=$JB/reach_volume
  mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $ISPATIALGYM_REACH_BANDS $ISPATIALGYM_CACHE_DIR $ISPATIALGYM_REACH_VOLUME $ISG_VEC_ASSET_CACHE_DIR
fi
cd $R
