#!/bin/bash
# RECORD_CHANGED_ONLY, nextto_02 100322 (episode 1) over frames 0-223, before full3's trajectory first differs (step 224)
F=/vision/u/minyeong/projects/isg_home/trainfast
K=$F/out/pv/inter_object_relation_satisfaction-nextto_02
P=/vision/u/minyeong/miniconda3/envs/behavior/bin/python
for pr in "rf3_rec rp_full" "rf_rec rp_full" "rp_full rp_full2" "rf_rec rf_rec2" "rf3_rec rf_rec"; do
  set -- $pr
  echo "== $1 vs $2: $($P -u $F/scripts/cmp_prefix.py $K/$1/lerobot_0 $K/$2/lerobot_0 1 224 2>&1 | grep '^episode')"
done
