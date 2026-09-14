#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ispatialgym
cd /tmp/glib
for k in $(python3 -c "
import glob, os
NS={'blender','carton','cedar_chest','cooler','hinged_jar','laptop','recycling_bin','toolbox','violin_case','mailbox','backpack'}
print(' '.join(sorted(os.path.basename(p)[:-5] for p in glob.glob('/tmp/glib/artic/handles/*.json') if os.path.basename(p).split('__')[0] in NS)))"); do
  [ -f "artic/results/$k.json" ] && continue
  ONLY=$k timeout 1500 python -u artic_validate_table.py 0 1 >> artic_vt4.log 2>&1
done
for sc in Rs_int house_single_floor house_double_floor_lower house_double_floor_upper restaurant_diner office_cubicles_right; do
  for i in 1 2 3 4 5 6 7 8; do
    n=$(python3 -c "
import json, os
a=[x for x in json.load(open('/tmp/glib/artic/scene_assign.json')) if x['scene']=='$sc' and not os.path.exists('/tmp/glib/artic/results/'+x['category']+'__'+x['model']+'.json')]
print(len(a))")
    [ "$n" = "0" ] && break
    LIMIT=4 timeout 3600 python -u artic_validate_scene.py "$sc" >> "artic_vs4_${sc}.log" 2>&1
  done
done
echo "CHAIN4 DONE $(date)" > artic_all4.done
