# Sandbox of the lab-eval worktree's conda env on the cluster (lab_tools/cluster/SETUP_PROMPT.md): copy to
# $CONDA_PREFIX/etc/conda/activate.d/ispatialgym_lab.sh. The lab machine's hook plus what scripts/isg/prod_job.sh does on
# the cluster: the sampler / oracle caches and the reach-band files private to this worktree (the reach-band samplers write
# feasible sets back into the files they read), Kit's portable root on node-local disk (one on NFS stalled Kit starts and
# ran out of lock slots; the first launch on a node builds it, ~3-5 min).
export ISPATIALGYM_CACHE_DIR="${ISPATIALGYM_LAB_CACHE:-$HOME/.cache/ispatialgym-lab-vec-eval}"
export ISPATIALGYM_SAMPLING_CACHE="$ISPATIALGYM_CACHE_DIR/sampling"
export ISPATIALGYM_REACH_BANDS="$ISPATIALGYM_CACHE_DIR/reach_bands"
export ISPATIALGYM_REACH_VOLUME="$ISPATIALGYM_CACHE_DIR/reach_volume"
export ISPATIALGYM_HELD_HOLD_CACHE="$ISPATIALGYM_CACHE_DIR/held_hold.json"
export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"
export OMNI_KIT_ACCEPT_EULA=YES
export HDF5_USE_FILE_LOCKING=FALSE
_isg_scratch="${TMPDIR:-/tmp}"
[ -d "$_isg_scratch" ] && [ -w "$_isg_scratch" ] || _isg_scratch=/tmp
export OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-$_isg_scratch/${USER:-isg}/isg_lab_appdata}"
mkdir -p "$OMNIGIBSON_APPDATA_PATH" 2>/dev/null
unset _isg_scratch
