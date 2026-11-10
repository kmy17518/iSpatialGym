#!/bin/bash
# i7-artic 2026-11-07 (owner: handles stay with the ARTICULATION library): the articulation prefix maps of small-articulation aliases in one job -
#   rest     validate_artic rest    (the alias's profile by the sampler's rule, the stable rest at the sampler's placement rest)
#   motions  validate_artic motions (the oracle's motion per handle variant in physics -> artic_library__<alias>.json)
#   prefix   reach_maps prefix      (one map per library x GB_HEIGHTS x arm the task offers; a non-offered arm is skipped and logged)
#
#   GB_ALIASES="<customs alias> ..." GB_OUT=<dir> GB_ANCHORS=<anchors.json> GB_HEIGHTS="0.8126 0.7853" [GB_STAGES="rest motions prefix"] [GB_Q_STARTS=grid] \
#   [GB_ARMS="right left"] [GB_VAL_ARM=right|left] ISG_GB_ROOT=<frozen snapshot> GBH=<campaign home> sbatch -o <log> job_artic.sh
#
# Owner 2026-11-07: an arm's prefix map is built ONLY from segments validated with that arm. GB_VAL_ARM (default right) = the arm of the
# rest / motions stages (validate_artic --arm; outputs artic_library__<alias>[__left].json); the prefix stage reads, per arm of GB_ARMS,
# that arm's libraries only (reach_maps refuses another arm's segments anyway).
#
# Outputs: $GB_OUT/alib/models/<category>__<model>__s1__articulation/{geometry.json, motions/<alias>{,__rest}.json, artic_library__<alias>.json},
# $GB_OUT/maps/_articulation/prefix__*.{json,npz} (the sampler reads <ISPATIALGYM_REACH_MAPS>/_articulation, or ISPATIALGYM_ARTIC_PREFIX_MAPS),
# $GB_OUT/logs/{rest,motions,prefix_<arm>}_<job id>.log (one log per job: a re-run never reads an older done line). A stage counts only
# with its done line (a Kit start failure exits 0).
#SBATCH --job-name=isgartic
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
set -u
GBH=${GBH:?}
source ${ISG_GB_ROOT:?}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
OUT=${GB_OUT:?}; ALIASES=${GB_ALIASES:?}
mkdir -p $OUT/alib $OUT/maps/_articulation $OUT/logs
STAGES=${GB_STAGES:-rest motions prefix}
RC=0
JID=${SLURM_JOB_ID:-local$$}
echo "GB_ARTIC_JOB host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $OUT aliases $ALIASES start $(date +%s)"
for ST in rest motions; do
  if [[ " $STAGES " == *" $ST "* ]]; then
    t0=$(date +%s)
    python -u -m ispatialgym.grasping.validate_artic $ST --aliases $ALIASES --root $OUT/alib --arm ${GB_VAL_ARM:-right} --q-starts ${GB_Q_STARTS:-motion} >> $OUT/logs/${ST}_$JID.log 2>&1
    rc=$?; grep -q "GB_ARTIC_DONE" $OUT/logs/${ST}_$JID.log || rc=${rc/#0/1}
    [ $RC = 0 ] && RC=$rc
    echo "GB_STAGE $ST rc=$rc s=$(( $(date +%s) - t0 ))"
  fi
done
if [[ " $STAGES " == *" prefix "* ]]; then
  for ARM in ${GB_ARMS:-right left}; do
    SFX=$([ $ARM = right ] && echo "" || echo "__$ARM")
    LIBS=()
    for A in $ALIASES; do
      for f in $OUT/alib/models/*/artic_library__$A$SFX.json; do [ -e "$f" ] && LIBS+=("$f"); done
      # i7-artsmall (GB_Q_STARTS=grid): the per-start push-close libraries artic_library__<alias>__q<q>[__<arm>].json of this arm
      for f in $OUT/alib/models/*/artic_library__${A}__q*$SFX.json; do
        [ -e "$f" ] || continue
        [ $ARM = right ] && [[ "$f" == *__left.json ]] && continue
        LIBS+=("$f")
      done
    done
    if [ ${#LIBS[@]} = 0 ]; then
      echo "GB_STAGE prefix arm=$ARM: no artic_library__<alias>$SFX.json under $OUT/alib (validate_artic motions --arm $ARM first) - nothing to map"
    else
      t0=$(date +%s)
      python -u -m ispatialgym.sampling.reach_maps prefix --artic-lib "${LIBS[@]}" --anchors ${GB_ANCHORS:?} --heights ${GB_HEIGHTS:?} --arm $ARM \
        --out $OUT/maps/_articulation --skip-existing >> $OUT/logs/prefix_${ARM}_$JID.log 2>&1
      rc=$?; grep -q "GB_PREFIX_DONE" $OUT/logs/prefix_${ARM}_$JID.log || rc=${rc/#0/1}
      [ $RC = 0 ] && RC=$rc
      echo "GB_STAGE prefix arm=$ARM rc=$rc s=$(( $(date +%s) - t0 ))"
    fi
  done
fi
echo "GB_ARTIC_JOB_DONE rc=$RC end $(date +%s)"
exit $RC
