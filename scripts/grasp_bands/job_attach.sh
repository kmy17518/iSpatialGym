#!/bin/bash
# i7-attachmap 2026-11-07: the ATTACH reach maps of attach pairs in one job (python -m ispatialgym.sampling.attach_maps build):
#   spec  per pair of GB_PAIRS: the male entry (GB_GLIB/models/<male_entry>) and the partner spawned attachable, the pair measured by the
#         attachment task's own code (attachment_links, insertion_axis_rule, predicates.seat_axis, the partner's collision pieces) ->
#         $GB_OUT/specs/<pair id>/{attach_spec.json, partner_geometry.json} (kept: a rerun re-uses it unless GB_REMEASURE=1)
#   maps  one attach map per pair height (anchor keys of GB_ANCHORS) x hover rule (GB_RULES, default "box swept") x arm (GB_ARMS, default
#         right: the attachment task's only attach arm) -> $GB_OUT/maps/<male entry>/amap__attach__*.{json,npz}
#
#   GB_PAIRS=<pairs.json> GB_GLIB=<v2 glib root> GB_ANCHORS=<anchors.json> GB_OUT=<dir> ISG_GB_ROOT=<frozen snapshot> GBH=<campaign home> \
#     sbatch -A vision -p svl --exclude=svl9 --gres=gpu:titanrtx:1 -o <log> job_attach.sh
#
# A failure inside python can still exit 0: the job counts only with ATTACH_DONE in its log and every ATTACH_SPEC / ATTACH_MAP line
# "ok": true (the job's own summary line GB_ATTACH_JOB_DONE reports both).
#SBATCH --job-name=isgattach
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
OUT=${GB_OUT:?}; PAIRS=${GB_PAIRS:?}; GLIB=${GB_GLIB:?}; ANCHORS=${GB_ANCHORS:?}
case "$OUT" in /vision/u/minyeong/projects/isg_home/grasp_bands*) echo "GB_OUT must not be under the grasp-bands home"; exit 2;; esac
mkdir -p $OUT/specs $OUT/maps $OUT/logs
JID=${SLURM_JOB_ID:-local$$}
echo "GB_ATTACH_JOB host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) code $R out $OUT pairs $PAIRS glib $GLIB start $(date +%s)"
RC=0
for ARM in ${GB_ARMS:-right}; do
  t0=$(date +%s)
  LOG=$OUT/logs/build_${ARM}_$JID.log
  python -u -m ispatialgym.sampling.attach_maps build --pairs $PAIRS --glib $GLIB --anchors $ANCHORS --out $OUT --arm $ARM \
    --rules ${GB_RULES:-box swept} $([ "${GB_REMEASURE:-0}" = 1 ] && echo --remeasure) >> $LOG 2>&1
  rc=$?
  grep -q "ATTACH_DONE" $LOG || rc=${rc/#0/1}
  nbad=$(grep -E "^ATTACH_(SPEC|MAP) " $LOG | grep -c '"ok": false')
  nok=$(grep -E "^ATTACH_MAP " $LOG | grep -c '"ok": true')
  [ "$nbad" != 0 ] && rc=${rc/#0/1}
  [ $RC = 0 ] && RC=$rc
  echo "GB_STAGE build arm=$ARM rc=$rc maps_ok=$nok not_ok=$nbad s=$(( $(date +%s) - t0 ))"
done
echo "GB_ATTACH_JOB_DONE rc=$RC end $(date +%s)"
exit $RC
