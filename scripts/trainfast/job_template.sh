#!/bin/bash
# one trunk (owner 2026-11-05): the template-making step that stays of job_val.sh KIND=eval - sample_isg at an eval seed on the frozen
# snapshot with the fast flow's environment (env_fast.sh, fast_env.sh, the job.env overrides), no demo. Next: register_alias.py (the
# template copy, eval index, registry row) and submit_fast.py submit (validation / collection through the fast flow).
#   submit_fast.py template <alias> <key> --code <snapshot> --seed-index K [--scene S] [--support SUP] [--support-row ROW] [--n N] [--budget-s B]
#SBATCH -c 9
#SBATCH --mem=48G
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
JOBDIR=$1
F=/vision/u/minyeong/projects/isg_home/trainfast
source $JOBDIR/job.env
S=${ISGF_SCRIPTS:-$F/scripts}
KIT_RC=1
export ISGE_BAND_STORE=1 ISGE_CODE
source $S/env_fast.sh
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$JB/numba_cache
if [ "${START_PER_ATTEMPT:-1}" = 1 ]; then export ISPATIALGYM_START_PER_ATTEMPT=1; else unset ISPATIALGYM_START_PER_ATTEMPT; fi
source $S/fast_env.sh
set -a; source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' $JOBDIR/job.env); set +a
export ISPATIALGYM_SAMPLING_CACHE=${ISGF_SAMPLING_CACHE:-$F/cache/sampling}
mkdir -p $ISPATIALGYM_SAMPLING_CACHE
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$F/jobs/*) rm -rf "$JB" ;; esac' EXIT
for B in ${BAND_STORES:-}; do [ -d "$B" ] && cp --preserve=timestamps $B/*.json $B/*.npz $ISPATIALGYM_REACH_BANDS/ 2>/dev/null; done
echo "$SLURM_JOB_ID" > $JOBDIR/jobid
echo "ISGF_TEMPLATE $SLURM_JOB_ID alias=$ALIAS seed=${SEED_SPLIT:-eval}:${SEED_INDEX:-0} host=$(hostname -s) isg=$(cat $R/.snapshot_isg 2>/dev/null) start=$(date -Is)" | tee -a $JOBDIR/job.txt
env | grep "^ISPATIALGYM_\|^ISG_\|^OMNIGIBSON_" | sort > $JOBDIR/env.txt
t0=$(date +%s)
timeout -s INT -k 300 $(( ${BUDGET_S:-1800} + 900 )) python -m ispatialgym.sampling.sample_isg --variant $ALIAS ${SCENE:+--scene $SCENE} \
  ${SUPPORT:+--support $SUPPORT} ${SUPPORT_ROW:+--support-row $SUPPORT_ROW} --templates 1 --instances-per-template ${N:-1} \
  --max-attempts ${MAX_ATTEMPTS:-1000} --time-budget-s ${BUDGET_S:-1800} --seed-split ${SEED_SPLIT:-eval} --seed-index ${SEED_INDEX:-0} \
  --no-write-back --out $JOBDIR/inst --log-level INFO > $JOBDIR/sample.log 2>&1
RC=$?
[ $RC -eq 0 ] && KIT_RC=0
# PERSIST_BANDS=<dir> (job_val.sh's): the bands this job computed on a miss go into a STATIC store (registry band_stores / BAND_STORES
# of later fast-flow jobs read it, never write it: no sharing between jobs)
if [ -n "${PERSIST_BANDS:-}" ]; then
  mkdir -p $PERSIST_BANDS
  for f in $(grep -lE '"computed_by": "(iors_large compute-on-miss|sampling.v2.tabletop.compute_bands)"' $ISPATIALGYM_REACH_BANDS/*.json 2>/dev/null); do
    cp -n --preserve=timestamps $f ${f%.json}.npz $PERSIST_BANDS/ 2>/dev/null && echo "PERSIST_BAND $(basename $f)" | tee -a $JOBDIR/job.txt
  done
fi
echo "SAMPLE_EXIT $RC wall_s=$(( $(date +%s) - t0 )) written=$(grep -c ' written (attempt ' $JOBDIR/sample.log 2>/dev/null; true) tro=$(find $JOBDIR/inst -name '*-tro_state.json' | wc -l)" | tee -a $JOBDIR/job.txt
