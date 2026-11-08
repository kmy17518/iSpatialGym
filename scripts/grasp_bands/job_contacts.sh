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
# maps (i7-contact 2026-11-07, step 3): the contact maps of the root's libraries (python -m ispatialgym.sampling.contact_maps compute) per
# GC_MAP_KINDS (default "touch_push touch_brace push_blade axial_brace"; a kind whose libraries the root does not hold is a no-op), the
# library entries with a verdict in GC_MAP_VERDICTS (default "valid"; "all" = unvalidated too), at GC_MAP_HEIGHTS (default each library's
# validate.support_z), written to GC_MAPS_OUT (default $GC_ROOT/maps: <entry>/cmap__<kind>__*; the sampler reads them from the maps root
# ISPATIALGYM_REACH_MAPS beside the grasp maps). Logs $GC_ROOT/logs/maps_<i>_<kind>.log + $GC_MAPS_OUT/logs/maps_<kind>_<i>of<N>.jsonl:
#   GC_STAGES=maps GC_MANIFEST=none GC_GLIB=none GC_ROOT=<library root> GC_NSHARD=N ISG_GB_ROOT=<snapshot> sbatch --array=0-$((N-1)) job_contacts.sh
# A re-validation in a new order that keeps the earlier verdicts (i7-contact 2026-11-07: the axial (end, spin) groups): GC_VALIDATE_ARGS=--skip-validated
# on a cp -a of the library root.
# owner A6 (i7-contactfu 2026-11-08): STRATIFIED validation - GC_VALIDATE_ARGS="--per-group-valid 3 --per-group-max 8 [--skip-validated]" (2-3 valid
# per (yaw x end x spin) group; without --skip-validated a fresh run: unreached entries lose old verdicts); maps from VALID entries only (the
# default GC_MAP_VERDICTS); GC_MAP_OVERLAY=<merged contact overlay root> rebuilds a map with the entries real oracle successes confirmed
# (contact_overlay.py harvest -> map_tilt merge; the overlay version enters the map identity).
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
  python -u -m ispatialgym.grasping.validate_contacts replay --kind $RK --model-dir $RM --instance $RI ${GC_REPLAY_ARGS:-} >> $ROOT/logs/replay_$I.log 2>&1
  rc=$?; grep -q "GB_CONTACT_REPLAY" $ROOT/logs/replay_$I.log || rc=${rc/#0/1}; RC=$rc
  echo "GC_STAGE replay rc=$rc s=$(( $(date +%s) - t0 ))"
fi
if [[ " $STAGES " == *" validate "* ]]; then
  # i7-brace (owner 3.4): GC_VALIDATE_ARMS="left right" runs one validation per arm (a wall_brace library: each entry its own brace arm;
  # log validate_<i>_<kind>_<arm>.log); unset = one run at the kind's default arm (the log name as before)
  for KIND in ${GC_KINDS:-push brace}; do
    for ARM in ${GC_VALIDATE_ARMS:-default}; do
      t0=$(date +%s)
      if [ "$ARM" = default ]; then AARG=""; VLOG=$ROOT/logs/validate_${I}_$KIND.log; else AARG="--arm $ARM"; VLOG=$ROOT/logs/validate_${I}_${KIND}_$ARM.log; fi
      python -u -m ispatialgym.grasping.validate_contacts --root $ROOT --kind $KIND --glib $GLIB --jobs $M --shard $I/$N $AARG ${GC_VALIDATE_ARGS:-} >> $VLOG 2>&1
      rc=$?; grep -q "GB_CONTACT_VALIDATE_DONE" $VLOG || rc=${rc/#0/1}; [ $RC = 0 ] && RC=$rc
      echo "GC_STAGE validate kind=$KIND arm=$ARM rc=$rc s=$(( $(date +%s) - t0 ))"
    done
  done
fi
if [[ " $STAGES " == *" maps "* ]]; then
  MOUT=${GC_MAPS_OUT:-$ROOT/maps}; mkdir -p $MOUT
  # i7-contactfu 2026-11-08: with a manifest the maps stage maps THIS shard's manifest ids (the validate stage's partition) - the library ids'
  # own sorted partition let a shard map another shard's library before its validation ended (contact_libs_20261108s: napkins / cereal
  # touch_push maps read the old verdicts)
  MSEL="--shard $I/$N"
  if [ -f "$M" ]; then
    # (the validate stage's --only in GC_VALIDATE_ARGS filters before the partition: the same here)
    MIDS=$(python -c "
import sys
from ispatialgym.grasping import contact_library as CL
a = sys.argv[3].split(); only = []
if '--only' in a:
    for t in a[a.index('--only') + 1:]:
        if t.startswith('--'):
            break
        only.append(t)
print(' '.join(j['id'] for j in CL._jobs(sys.argv[1], only or None, sys.argv[2])))" $M $I/$N "${GC_VALIDATE_ARGS:-}")
    MSEL="--only ${MIDS:-__none__}"
  fi
  for KIND in ${GC_MAP_KINDS:-touch_push touch_brace push_blade axial_brace}; do
    t0=$(date +%s)
    python -u -m ispatialgym.sampling.contact_maps compute --root $ROOT --kind $KIND --out $MOUT $MSEL --verdicts ${GC_MAP_VERDICTS:-valid} \
      ${GC_MAP_HEIGHTS:+--heights $GC_MAP_HEIGHTS} ${GC_MAP_OVERLAY:+--overlay-root $GC_MAP_OVERLAY} >> $ROOT/logs/maps_${I}_$KIND.log 2>&1
    rc=$?; grep -q "GB_CONTACT_MAPS_DONE" $ROOT/logs/maps_${I}_$KIND.log || rc=${rc/#0/1}; [ $RC = 0 ] && RC=$rc
    echo "GC_STAGE maps kind=$KIND rc=$rc s=$(( $(date +%s) - t0 ))"
  done
fi
echo "GC_JOB_DONE shard $I rc=$RC end $(date +%s)"
