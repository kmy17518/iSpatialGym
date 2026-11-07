#!/bin/bash
# One shard of the grasp-library v2 + reach-map pipeline (sbatch --array=0..N-1):
#   GB_MANIFEST=<jobs.json> GB_NSHARD=N GB_TAG=<run tag> [GB_STAGES="glib maps"] [GB_HEIGHTS="0.8126 ..."] sbatch --array=0-$((N-1)) job_glib.sh
# glib = export + candidates + physics validation in one persistent Isaac Sim world (pipeline all); maps = reach maps of the shard's
# models at GB_HEIGHTS (anchor keys; default the three probe heights) in a robot-only world. Outputs: $GBH/out/$GB_TAG/{glib,maps,logs}.
# of-v2maps 2026-11-06 (a campaign's own copy; the grasp-bands record stays read-only): GBH (default the grasp-bands home: asset cache, NFS
# job dirs), ISG_GB_ROOT (the code root: a frozen snapshot), GB_OUT (default $GBH/out/$GB_TAG), GB_GLIB (the library the maps read,
# default $GB_OUT/glib), GB_HEIGHTS_BY_ID (json {entry id: [anchor keys]}), GB_INCLUDE_DERIVED (default 1), GB_ARMS (default "right": one maps pass per arm). Pass
# -o <log> on the sbatch line to keep the .out under the campaign (the #SBATCH --output below is grasp-bands').
# i7-generic 2026-11-07: GB_KIND (default tabletop; brace | generic) = the maps stage's reach_maps --kind. generic: the manifest's generic
# pseudo-entries ([{"id": "_generic", "kind": "generic"}], optional dz_m / fingers_q / dir) at GB_HEIGHTS (or GB_HEIGHTS_BY_ID by entry id),
# no library (GB_STAGES=maps; GB_GLIB unused), written to $GB_OUT/maps/_generic/, logs maps_<shard>_<arm>_generic.log.
#SBATCH --job-name=isggb
#SBATCH --account=vision
#SBATCH --partition=svl
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/vision/u/minyeong/projects/isg_home/grasp_bands/logs/%x_%A_%a.out
set -u
GBH=${GBH:-/vision/u/minyeong/projects/isg_home/grasp_bands}
source ${ISG_GB_ROOT:-/vision/u/minyeong/projects/ispatialgym-grasp-bands}/scripts/grasp_bands/env_cluster.sh
trap 'kit_appdata_teardown' EXIT
I=${SLURM_ARRAY_TASK_ID:-0}; N=${GB_NSHARD:?}; TAG=${GB_TAG:?}; M=${GB_MANIFEST:?}
OUT=${GB_OUT:-$GBH/out/$TAG}; mkdir -p $OUT/glib $OUT/maps $OUT/logs
GLIB=${GB_GLIB:-$OUT/glib}
STAGES=${GB_STAGES:-glib maps}
echo "GB_JOB shard $I/$N tag $TAG host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $OUT glib $GLIB start $(date +%s)"
KIT_RC=0
if [[ " $STAGES " == *" glib "* ]]; then
  t0=$(date +%s)
  python -u -m ispatialgym.grasping.pipeline all --jobs $M --root $OUT/glib --shard $I/$N --workers ${SLURM_CPUS_PER_TASK:-8} >> $OUT/logs/glib_$I.log 2>&1
  KIT_RC=$?
  # a Kit start that failed (an extension missing on the node) still exits 0: the stage counts only with its done line
  grep -q "GB_VALIDATE_DONE" $OUT/logs/glib_$I.log || KIT_RC=${KIT_RC/#0/1}
  echo "GB_STAGE glib rc=$KIT_RC s=$(( $(date +%s) - t0 ))"
fi
if [[ " $STAGES " == *" maps "* ]]; then
  BYID=(); [ -n "${GB_HEIGHTS_BY_ID:-}" ] && BYID=(--heights-by-id $GB_HEIGHTS_BY_ID)
  DERIVED=--include-derived; [ "${GB_INCLUDE_DERIVED:-1}" = 0 ] && DERIVED=""   # 0: only the manifest's entries (a campaign library holds others)
  KIND=${GB_KIND:-tabletop}; LOGK=""; [ "$KIND" != tabletop ] && LOGK="_$KIND"
  for ARM in ${GB_ARMS:-right}; do
    t0=$(date +%s)
    python -u -m ispatialgym.sampling.reach_maps compute --jobs $M --glib $GLIB --anchors ${GB_ANCHORS:-$GBH/anchors.json} --heights ${GB_HEIGHTS:-0.4230 0.8126 1.0505} \
      "${BYID[@]}" --arm $ARM --kind $KIND --shard $I/$N --out $OUT/maps --skip-existing $DERIVED >> $OUT/logs/maps_${I}_${ARM}${LOGK}.log 2>&1
    rc=$?; [ $KIT_RC = 0 ] && KIT_RC=$rc
    echo "GB_STAGE maps kind=$KIND arm=$ARM rc=$rc s=$(( $(date +%s) - t0 ))"
  done
fi
echo "GB_JOB_DONE shard $I rc=$KIT_RC end $(date +%s)"
