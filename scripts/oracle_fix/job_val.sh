#!/bin/bash
# isgof_* (oracle-fix validation, 2026-11-03): sample (KIND=train: multiply_isg from a COPY of a train85 template; KIND=eval: sample_isg at
# an eval seed) or copy existing instances (KIND=demo), then oracle demos (run_isg --queue) on the oracle-fix worktree. Source instances /
# templates are copied into the job dir first (run_isg writes the settled goal back into the TRO it runs). Writes only <job dir> and $OFH.
#   sbatch -J isgof_<name> -A vision -p svl --gres=gpu:titanrtx:1 --time=<T> scripts/oracle_fix/job_val.sh <job dir>   (job.env inside)
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/slurm/%x_%j.log
set -u
JOBDIR=$1
source $JOBDIR/job.env
for kv in ${ENV_EXTRA:-}; do export "$kv"; done
source /vision/u/minyeong/projects/ispatialgym-oracle-fix/scripts/oracle_fix/env_cluster.sh
KIT_RC=1
trap 'kit_appdata_teardown; case "$JB" in /tmp/*|$OFH/jobs/*) rm -rf "$JB" ;; esac' EXIT
if [ "${START_PER_ATTEMPT:-0}" = 1 ]; then export ISPATIALGYM_START_PER_ATTEMPT=1; else unset ISPATIALGYM_START_PER_ATTEMPT; fi
if [ "${STILL_EXIT:-1}" = 1 ]; then export ISPATIALGYM_ORACLE_STILL_EXIT=1; else unset ISPATIALGYM_ORACLE_STILL_EXIT; fi
mkdir -p $JOBDIR/claims $JOBDIR/inst
echo "$SLURM_JOB_ID" > $JOBDIR/jobid
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "ISGOF_START $SLURM_JOB_ID kind=$KIND host=$(hostname -s) gpu=\"$GPU\" env_extra=\"${ENV_EXTRA:-}\" start=$(date -Is)" | tee -a $JOBDIR/job.txt
t0=$(date +%s)
case "$KIND" in
  demo)
    for src in $SRC_INST; do rsync -a "$src"/ $JOBDIR/inst/; done  # several roots merge (distinct aliases: distinct <variant>/<template> dirs)
    if [ -n "${IIDS:-}" ]; then  # demo only these instances: a skip claim for every other one
      for f in $(find $JOBDIR/inst -name "*-tro_state.json"); do
        iid=$(basename $f -tro_state.json)
        case " $IIDS " in *" $iid "*) ;; *) echo "skip not in IIDS" > $JOBDIR/claims/$iid ;; esac
      done
    fi
    SAMPLE_RC=0 ;;
  train)
    rsync -a "$TEMPLATE_SRC"/ $JOBDIR/template_inst/
    timeout -s INT -k 300 ${SAMPLE_LIMIT_S:-7200} python -m ispatialgym.sampling.multiply_isg --split train --template $JOBDIR/template_inst/$TEMPLATE_REL \
      --instances $N --start-index $START --out $JOBDIR/inst --eval-run $EVAL_INDEX --no-write-back --max-attempts ${MAX_ATTEMPTS:-30} \
      ${KEEP_SCENE:-} --log-level INFO > $JOBDIR/sample.log 2>&1
    SAMPLE_RC=$? ;;
  eval)
    timeout -s INT -k 300 $(( ${BUDGET_S:-1800} + 900 )) python -m ispatialgym.sampling.sample_isg --variant $ALIAS ${SCENE:+--scene $SCENE} \
      ${SUPPORT:+--support $SUPPORT} ${SUPPORT_ROW:+--support-row $SUPPORT_ROW} --templates 1 --instances-per-template ${N:-1} \
      --max-attempts ${MAX_ATTEMPTS:-1000} --time-budget-s ${BUDGET_S:-1800} --seed-split ${SEED_SPLIT:-eval} --seed-index ${SEED_INDEX:-0} \
      --no-write-back --out $JOBDIR/inst --log-level INFO > $JOBDIR/sample.log 2>&1
    SAMPLE_RC=$? ;;
  *) echo "unknown KIND=$KIND"; exit 2 ;;
esac
if [ -n "${PERSIST_BANDS:-}" ]; then  # oracle-fix 2026-11-04 (owner item 5): keep the bands this job computed on a miss (job-local otherwise)
  mkdir -p $PERSIST_BANDS
  for f in $(grep -lE '"computed_by": "(iors_large compute-on-miss|sampling.v2.tabletop.compute_bands)"' $ISPATIALGYM_REACH_BANDS/*.json 2>/dev/null); do  # + tabletop bands (round 3: the mouse desk band, 25 min per job)
    cp -n --preserve=timestamps $f ${f%.json}.npz $PERSIST_BANDS/ 2>/dev/null && echo "PERSIST_BAND $(basename $f)" | tee -a $JOBDIR/job.txt
  done
fi
echo "SAMPLE_EXIT $SAMPLE_RC wall_s=$(( $(date +%s) - t0 )) written=$(grep -c ' written (attempt ' $JOBDIR/sample.log 2>/dev/null; true) tro=$(find $JOBDIR/inst -name '*-tro_state.json' | wc -l)" | tee -a $JOBDIR/job.txt
touch $JOBDIR/sampling.done
[ "${NO_DEMO:-0}" = 1 ] && exit 0
q=0; QRC=0
while [ $q -lt ${DEMO_PROCS:-4} ]; do
  left=$(( $(find $JOBDIR/inst -name '*-tro_state.json' | wc -l) - $(ls $JOBDIR/claims | wc -l) ))
  [ $left -le 0 ] && break
  q=$(( q + 1 )); t1=$(date +%s)
  echo "ISGOF_DEMO q=$q left=$left start=$(date -Is)" | tee -a $JOBDIR/job.txt
  timeout -s INT -k 120 $(( left * ${DEMO_LIMIT_S:-1800} + 600 )) python -m ispatialgym.oracle.run_isg --queue $JOBDIR/inst --claims $JOBDIR/claims \
    --done-file $JOBDIR/sampling.done --results $JOBDIR/results.jsonl --mode oracle --out $JOBDIR/demo --video-size 540 \
    > $JOBDIR/demo_q$q.log 2>&1
  QRC=$?
  echo "DEMO_EXIT q=$q rc=$QRC wall_s=$(( $(date +%s) - t1 )) results=$(cat $JOBDIR/results.jsonl 2>/dev/null | wc -l)" | tee -a $JOBDIR/job.txt
done
[ "$QRC" -eq 0 ] && KIT_RC=0
echo "ISGOF_END $SLURM_JOB_ID wall_s=$(( $(date +%s) - t0 )) end=$(date -Is)" | tee -a $JOBDIR/job.txt
