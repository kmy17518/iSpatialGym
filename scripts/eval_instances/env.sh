# Environment for Isaac Sim jobs of this repo (sourced by the job scripts in scripts/eval_instances/).
# In a Slurm batch job: Kit app data warm from the NFS template of this GPU + driver + Behavior-1K commit and the shared
# texture store (cluster_tools/kit_appdata.sh; the job script sets the EXIT trap and KIT_RC), HOME and the torch / triton
# caches inside that per-job folder. Not in an interactive allocation's shell (scontrol: BatchFlag=0; the editor's shells
# run in one), where it would move HOME for the rest of the session: ISG_KIT_APPDATA=1 sets it up there anyway (run
# kit_appdata_teardown when done), ISG_KIT_APPDATA=0 never. The simulator data path is the repo's Behavior-1K/datasets
# (shared datasets linked in, see README of datasets/ispatialgym-eval-instances). The settings below are the ones the eval
# demos were recorded with.
R=${ISG_EVAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
_isg_kit=${ISG_KIT_APPDATA:-}
if [ -z "$_isg_kit" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  case "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)" in
    *BatchFlag=0*) _isg_kit=0; echo "env.sh: interactive allocation, no Kit app data (ISG_KIT_APPDATA=1 sets it up)" >&2 ;;
    *) _isg_kit=1 ;;
  esac
fi
if [ "$_isg_kit" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  WT=$R
  source /vision/u/minyeong/projects/cluster_tools/kit_appdata.sh
  kit_appdata_setup
  mkdir -p $OMNIGIBSON_APPDATA_PATH/home
  export HOME=$OMNIGIBSON_APPDATA_PATH/home XDG_CACHE_HOME=$OMNIGIBSON_APPDATA_PATH/home/.cache \
    TORCHINDUCTOR_CACHE_DIR=$OMNIGIBSON_APPDATA_PATH/inductor TRITON_CACHE_DIR=$OMNIGIBSON_APPDATA_PATH/triton
fi
unset _isg_kit
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
export PYTHONPATH=$R:$R/Behavior-1K/OmniGibson:$R/Behavior-1K/bddl3
export OMNIGIBSON_DATA_PATH=${ISG_EVAL_DATA_PATH:-$R/Behavior-1K/datasets}
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1
export OMNIGIBSON_TRACK_STATE_UPDATES=1 OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0
export ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
cd $R
