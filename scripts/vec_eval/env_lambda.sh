# the lambda 4090 box: environment of the vectorized evaluator (scripts/eval_instances/env.sh without SLURM / Kit app-data helper)
R=${ISG_EVAL_ROOT:-/home/minyeongk/Desktop/projects/ispatialgym-eval-vec-next}
source /home/minyeongk/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
export PYTHONPATH=$R:$R/Behavior-1K/OmniGibson:$R/Behavior-1K/bddl3
export OMNIGIBSON_DATA_PATH=$R/Behavior-1K/datasets
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1
export OMNIGIBSON_TRACK_STATE_UPDATES=1 OMP_NUM_THREADS=1 OMNIGIBSON_CHANGE_DETECTION_NUMPY=1 NUMPY_MADVISE_HUGEPAGE=0
export ISPATIALGYM_CUROBO_WARMUP_ITERS=0 PYTHONHASHSEED=0
export ISG_VEC_ROOT=/home/minyeongk/vec_eval85
mkdir -p $ISG_VEC_ROOT
cd $R
