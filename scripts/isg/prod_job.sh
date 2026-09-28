#!/bin/bash
# One production job (slurm_sweep produce): the thin job.sh sbatch was given resolves the snapshot link, exports PROD_* and
# execs this script from the snapshot, so a change here reaches every job that starts after the link moves.
#
# PROD_SNAP (snapshot root: ispatialgym/ + Behavior-1K/), PROD_ALIAS, PROD_START, PROD_COUNT, PROD_OUT, PROD_SEED,
# PROD_MAX_ATTEMPTS, PROD_DEMO_WORKERS, PROD_DEMO_MODE, PROD_HOURS, PROD_GRACE_MIN, PROD_THREADS, PROD_HOME, PROD_CONDA_SH,
# PROD_ENV, PROD_DATA_PATH, PROD_FALLBACK_TMP, PROD_MIN_SCRATCH_KB
export HOME="$PROD_HOME"
free_kb() { df -Pk "$1" 2>/dev/null | awk 'NR==2{print $4}'; }
if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && [ -w "$TMPDIR" ] && [ "$(free_kb "$TMPDIR")" -gt "$PROD_MIN_SCRATCH_KB" ]; then
  SCRATCH="$TMPDIR"
elif [ -w /dev/shm ] && [ "$(free_kb /dev/shm)" -gt "$PROD_MIN_SCRATCH_KB" ]; then
  SCRATCH="/dev/shm/$USER"  # a node whose /tmp is full: RAM-backed scratch (charged to the job's --mem)
else
  SCRATCH=""
fi
if [ -n "$SCRATCH" ]; then
  JOB_TMP="$SCRATCH/isg_$SLURM_JOB_ID"
  # Kit's portable root (shader / derived-data caches) shared by the jobs on this node's local disk: only the first
  # process on a node builds it (~3 min); one shared on NFS stalled Kit starts and ran out of lock slots
  APPDATA="$SCRATCH/isg_appdata_shared"
else
  JOB_TMP="$PROD_FALLBACK_TMP/isg_$SLURM_JOB_ID"
  APPDATA="$JOB_TMP/appdata"
fi
mkdir -p "$JOB_TMP/reach_bands" "$APPDATA"
source "$PROD_CONDA_SH"
conda activate "$PROD_ENV"
CODE="$PROD_SNAP/ispatialgym"; B1K="$PROD_SNAP/Behavior-1K"
cd "$CODE" || exit 2
export PYTHONPATH="$CODE:$B1K/OmniGibson:$B1K/bddl3${PYTHONPATH:+:$PYTHONPATH}"
export OMNIGIBSON_DATA_PATH="$PROD_DATA_PATH" OMNIGIBSON_APPDATA_PATH="$APPDATA" OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
# OmniGibson's per-step change detection (gm.TRACK_STATE_UPDATES) stays on: the executor's per-step hdf5 dump filter reads
# is_active, and with it off an object whose state changed while it slept is left out of that step's recorded state, which
# BEHAVIOR's playback loads step by step. Off is ~half of an env step faster; PROD_TRACK_STATE_UPDATES=0 turns it off.
export OMNIGIBSON_TRACK_STATE_UPDATES="${PROD_TRACK_STATE_UPDATES:-1}"
export TMPDIR="$JOB_TMP"
# the samplers write feasible sets back into the band files they read, in place: a private copy per job (cp -p keeps the
# mtimes the loader breaks ties with)
cp -p "$CODE"/runlogs/reach_bands_20260915/bands/* "$TMPDIR/reach_bands/" 2>/dev/null
export ISPATIALGYM_REACH_BANDS="$TMPDIR/reach_bands" ISPATIALGYM_CACHE_DIR="$TMPDIR/isg_cache" ISPATIALGYM_SAMPLING_CACHE="$TMPDIR/isg_cache/sampling" \
  ISPATIALGYM_HELD_HOLD_CACHE="$TMPDIR/isg_cache/held_hold.json"
export OMP_NUM_THREADS="$PROD_THREADS" MKL_NUM_THREADS="$PROD_THREADS" OPENBLAS_NUM_THREADS="$PROD_THREADS"
echo "PROD_JOB $SLURM_JOB_ID host=$(hostname -s) gpu=\"$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)\" TMPDIR=$TMPDIR appdata=$APPDATA code=$CODE isg_commit=$(git -C "$CODE" rev-parse HEAD) b1k_commit=$(git -C "$B1K" rev-parse HEAD) track_state_updates=$OMNIGIBSON_TRACK_STATE_UPDATES start=$(date -Is)"
deadline=$(( $(date +%s) + ${PROD_HOURS%.*} * 3600 - ${PROD_GRACE_MIN%.*} * 60 ))
python "$CODE/scripts/isg/prod_worker.py" --alias "$PROD_ALIAS" --start "$PROD_START" --count "$PROD_COUNT" --out "$PROD_OUT" --seed "$PROD_SEED" \
  --max-attempts "$PROD_MAX_ATTEMPTS" --demo-workers "$PROD_DEMO_WORKERS" --demo-mode "$PROD_DEMO_MODE" --deadline "$deadline"
rc=$?
echo "PROD_JOB_END $SLURM_JOB_ID rc=$rc end=$(date -Is)"
case "$JOB_TMP" in /vision/*) ;; *) rm -rf "$JOB_TMP" ;; esac
exit $rc
