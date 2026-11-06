# lambda (RTX 4090) environment of the oracle-fix worktree: source scripts/oracle_fix/env_lambda.sh [appdata slot, default 0]
# Copy of scripts/grasp_bands/env_lambda.sh; every data path points at ~/oracle_fix (copies of ~/grasp_bands made 2026-11-03), never at
# ~/grasp_bands, so the grasp-bands worktree's reach bands / grasp library / caches are never touched from here.
R=${ISG_OF_ROOT:-/home/minyeongk/Desktop/projects/ispatialgym-oracle-fix}
source /home/minyeongk/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
export PYTHONPATH=$R:$R/Behavior-1K/OmniGibson:$R/Behavior-1K/bddl3
export OMNIGIBSON_DATA_PATH=$R/Behavior-1K/datasets
export OMNIGIBSON_APPDATA_PATH=${OMNIGIBSON_APPDATA_PATH:-$HOME/oracle_fix/appdata/p${1:-0}}
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1
export OMNIGIBSON_TRACK_STATE_UPDATES=1 OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0
export ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
# private caches (never the shared ~/.cache/ispatialgym, never ~/grasp_bands)
export ISPATIALGYM_CACHE_DIR=$HOME/oracle_fix/cache ISPATIALGYM_SAMPLING_CACHE=$HOME/oracle_fix/cache/sampling
export ISPATIALGYM_REACH_BANDS=$HOME/oracle_fix/cache/reach_bands ISPATIALGYM_REACH_VOLUME=$HOME/oracle_fix/cache/reach_volume
export ISPATIALGYM_HELD_HOLD_CACHE=$HOME/oracle_fix/cache/held_hold.json
export ISG_VEC_ASSET_CACHE_DIR=${ISG_VEC_ASSET_CACHE_DIR:-$HOME/oracle_fix/cache/asset_cache} ISG_VEC_PREPARE_CACHE=1 ISG_VEC_MESH_CACHE=1
export ISG_OF_OUT=${ISG_OF_OUT:-$HOME/oracle_fix/out}
mkdir -p $ISG_OF_OUT $ISPATIALGYM_CACHE_DIR
cd $R
