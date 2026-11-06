F=/vision/u/minyeong/projects/isg_home/trainfast
for k in disembedding_03 object_scaling-bell_pepper; do
  JOBDIR=/nonexistent; ALIAS_KEY=$k
  TEMPLATE_ID=$(grep "^TEMPLATE_ID=" $(ls -d $F/out/pilot/$k/g* | head -1)/job.env | cut -d= -f2 | tr -d "'")
  n=0; SRC=""
  for d in $(ls -td $F/out/*/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID /vision/u/minyeong/projects/isg_home/train85/out/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID 2>/dev/null); do
    n=$((n+1)); case "$d" in $JOBDIR/*) continue ;; esac
    if [ -f "$d/template.json" ] && ls $d/instances/*-tro_state.json >/dev/null 2>&1; then SRC=$d; break; fi
  done
  echo "$k: candidates scanned=$n SRC=${SRC#$F/}"
  [ -n "$SRC" ] && grep -o '"scene_file": "[^"]*"' $SRC/template_meta.json | head -2
done
