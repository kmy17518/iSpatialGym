#!/bin/bash
# after the main chain: re-run the tabletop models whose results predate the yaw fix
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ispatialgym
cd /tmp/glib
while [ ! -f artic_all.done ]; do sleep 60; done
T=$(cat artic/yaw_fix_time)
for f in artic/results/*.json; do
  k=$(basename "$f" .json); c=${k%%__*}
  case "$c" in blender|carton|cedar_chest|cooler|hinged_jar|laptop|recycling_bin|toolbox|violin_case|mailbox|backpack) ;; *) continue;; esac
  if [ "$(stat -c %Y "$f")" -lt "$T" ]; then mkdir -p artic/results_prefix; mv "$f" artic/results_prefix/; fi
done
for k in $(python3 -c "
import glob, os
NS={'blender','carton','cedar_chest','cooler','hinged_jar','laptop','recycling_bin','toolbox','violin_case','mailbox','backpack'}
print(' '.join(sorted(os.path.basename(p)[:-5] for p in glob.glob('/tmp/glib/artic/handles/*.json') if os.path.basename(p).split('__')[0] in NS)))"); do
  [ -f "artic/results/$k.json" ] && continue
  ONLY=$k timeout 1500 python -u artic_validate_table.py 0 1 >> artic_vt.log 2>&1
done
echo "CHAIN2 DONE $(date)" > artic_all2.done
