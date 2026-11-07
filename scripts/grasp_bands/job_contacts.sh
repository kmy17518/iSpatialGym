#!/bin/bash
# One shard of the contact-library pipeline (i7-contact 2026-11-07, step 2): build (CPU) and physics-validate (Isaac Sim) the PUSH / BRACE
# libraries of the manifest's objects (ispatialgym/grasping/contact_library.py, validate_contacts.py). Everything goes under $OFH - never the
# grasp-bands home: GBH (the env's asset cache / job dirs) defaults to $OFH/contact_home here.
#   GC_MANIFEST=<jobs.json> GC_GLIB=<v2 glib root: the objects' model dirs> GC_ROOT=$OFH/contact_libs_<tag> GC_NSHARD=N \
#     [GC_STAGES="build validate"] [GC_KINDS="push brace"] [GC_REPLAY_SPECS=<file>: GC_STAGES=replay] ISG_GB_ROOT=<frozen code snapshot> \
#     sbatch --array=0-$((N-1)) -o $GC_ROOT/logs/%x_%A_%a.out scripts/grasp_bands/job_contacts.sh
# build: python -m ispatialgym.grasping.contact_library build (no simulator); validate: one Isaac Sim world per kind and arm (push: right,
# brace: left), the shard's manifest ids (the build's partition). Logs: $GC_ROOT/logs/{build,validate}_<i>of<N>.jsonl + <stage>_<i>.log.
# A failure inside python can still exit 0: the stages count only with their done lines (GB_CONTACT_BUILD_DONE / GB_CONTACT_VALIDATE_DONE).
#SBATCH --job-name=isgcontact
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
set -u
OFH=${OFH:-/vision/u/minyeong/projects/isg_home/oracle_fix}
export GBH=${GBH:-$OFH/contact_home}
source ${ISG_GB_ROOT:-/vision/u/minyeong/projects/ispatialgym-grasp-bands}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
I=${SLURM_ARRAY_TASK_ID:-0}; N=${GC_NSHARD:?}; M=${GC_MANIFEST:?}; GLIB=${GC_GLIB:?}; ROOT=${GC_ROOT:?}
case "$ROOT" in /vision/u/minyeong/projects/isg_home/grasp_bands*) echo "GC_ROOT must not be under the grasp-bands home"; exit 2;; esac
mkdir -p $ROOT/logs
STAGES=${GC_STAGES:-build validate}
echo "GC_JOB shard $I/$N host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R root $ROOT glib $GLIB start $(date +%s)"
RC=0
if [[ " $STAGES " == *" build "* ]]; then
  t0=$(date +%s)
  python -u -m ispatialgym.grasping.contact_library build --jobs $M --glib $GLIB --root $ROOT --shard $I/$N >> $ROOT/logs/build_$I.log 2>&1
  rc=$?; grep -q "GB_CONTACT_BUILD_DONE" $ROOT/logs/build_$I.log || rc=${rc/#0/1}; RC=$rc
  echo "GC_STAGE build rc=$rc s=$(( $(date +%s) - t0 ))"
fi
if [[ " $STAGES " == *" replay "* ]]; then
  # one recorded demo replay per array task: line I+1 of GC_REPLAY_SPECS = "<push|brace> <v2 model dir of the contacted object> <instance .oracle.json> [...]"
  t0=$(date +%s)
  SPEC=$(sed -n "$((I + 1))p" ${GC_REPLAY_SPECS:?})
  read -r RK RM RI <<< "$SPEC"
  python -u -m ispatialgym.grasping.validate_contacts replay --kind $RK --model-dir $RM --instance $RI >> $ROOT/logs/replay_$I.log 2>&1
  rc=$?; grep -q "GB_CONTACT_REPLAY" $ROOT/logs/replay_$I.log || rc=${rc/#0/1}; RC=$rc
  echo "GC_STAGE replay rc=$rc s=$(( $(date +%s) - t0 ))"
fi
if [[ " $STAGES " == *" validate "* ]]; then
  for KIND in ${GC_KINDS:-push brace}; do
    t0=$(date +%s)
    python -u -m ispatialgym.grasping.validate_contacts --root $ROOT --kind $KIND --glib $GLIB --jobs $M --shard $I/$N >> $ROOT/logs/validate_${I}_$KIND.log 2>&1
    rc=$?; grep -q "GB_CONTACT_VALIDATE_DONE" $ROOT/logs/validate_${I}_$KIND.log || rc=${rc/#0/1}; [ $RC = 0 ] && RC=$rc
    echo "GC_STAGE validate kind=$KIND rc=$rc s=$(( $(date +%s) - t0 ))"
  done
fi
echo "GC_JOB_DONE shard $I rc=$RC end $(date +%s)"
