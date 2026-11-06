#!/bin/bash
# RECORD_CHANGED_ONLY verdict (2026-11-01): head-camera frames of rf3_rec (full-scene recollection with the changed-only recording filter,
# replayed on the recorded full scene) against the cropped path (rp_full), the old full-scene replay (rf_rec / rf_rec2) and the noise floor
F=/vision/u/minyeong/projects/isg_home/trainfast
K=$F/out/pv/inter_object_relation_satisfaction-nextto_02
P=/vision/u/minyeong/miniconda3/envs/behavior/bin/python
cd $K/cmp3
for pr in "rf3_rec rp_full" "rf3_rec rf_rec" "rp_full rp_full2" "rf3_rec rf_rec2"; do
  set -- $pr
  echo "== $1 vs $2"
  $P -u $F/scripts/compare_frames.py $K/$1/lerobot_0 $K/$2/lerobot_0 $K/cmp3/png_$1_vs_$2 > $K/cmp3/$1_vs_$2.txt 2>&1
  echo "exit $?"
done
