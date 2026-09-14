#!/bin/bash
# after chain2: re-run scene tests whose results predate the hanging-arms stance fix
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ispatialgym
cd /tmp/glib
while [ ! -f artic_all2.done ]; do sleep 60; done
T=$(cat artic/scene_fix_time); mkdir -p artic/results_prefix
python3 - <<'PY'
import json, os, glob, shutil
T = int(open("/tmp/glib/artic/scene_fix_time").read())
assign = json.load(open("/tmp/glib/artic/scene_assign.json")); keys = {f"{a['category']}__{a['model']}" for a in assign}
n = 0
for k in keys:
    p = f"/tmp/glib/artic/results/{k}.json"
    if os.path.exists(p) and os.path.getmtime(p) < T:
        r = json.load(open(p))
        if not any(j.get("opened") for j in r["joints"]): shutil.move(p, f"/tmp/glib/artic/results_prefix/{k}.json"); n += 1
print("moved", n, "pre-fix scene results for re-run")
PY
for sc in Rs_int house_single_floor house_double_floor_lower house_double_floor_upper restaurant_diner office_cubicles_right; do
  for i in 1 2 3 4 5 6 7 8; do
    n=$(python3 -c "
import json, os
a=[x for x in json.load(open('/tmp/glib/artic/scene_assign.json')) if x['scene']=='$sc' and not os.path.exists('/tmp/glib/artic/results/'+x['category']+'__'+x['model']+'.json')]
print(len(a))")
    [ "$n" = "0" ] && break
    LIMIT=4 timeout 3600 python -u artic_validate_scene.py "$sc" >> "artic_vs_${sc}.log" 2>&1
  done
done
echo "CHAIN3 DONE $(date)" > artic_all3.done
