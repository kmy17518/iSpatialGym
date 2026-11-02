#!/bin/bash
# partial-scene validation (2026-11-01): copy N successful instances of a source job into out/pv/<key>/{partial,full} queues
#   mk_pv.sh <key> <source tag> [N=2]
set -eu
F=/vision/u/minyeong/projects/isg_home/trainfast
KEY=$1; SRCTAG=$2; NI=${3:-2}
SRC=$(ls -d $F/out/$SRCTAG/$KEY/g* | grep -v "\.cancelled\|failed" | head -1)
ids=$(grep "\"success\": true" $SRC/results.jsonl | head -$NI | python3 -c "import sys,json; print(' '.join(json.loads(l)['instance'].rsplit('__',1)[1] for l in sys.stdin))")
echo "$KEY ($SRC): $ids"
for side in partial full; do
  D=$F/out/pv/$KEY/$side
  mkdir -p $D/claims
  cp $SRC/job.env $D/job.env
  [ $side = full ] && echo "export ISPATIALGYM_FAST_PARTIAL_SCENE=0" >> $D/job.env
  grep "^ISPATIALGYM_" $SRC/job.env | sed 's/^/export /' >> $D/job.env || true
  (cd $SRC/inst && find . -maxdepth 1 -mindepth 1 -type d) | while read a; do
    mkdir -p $D/inst/$a
    [ "$a" = ./scene_variants ] && { cp -r $SRC/inst/$a/. $D/inst/$a/; continue; }
    for t in $SRC/inst/$a/*/; do
      [ -d $t/instances ] || { cp -r $t $D/inst/$a/; continue; }
      tn=$(basename $t); mkdir -p $D/inst/$a/$tn/instances
      find $t -maxdepth 1 -type f -exec cp {} $D/inst/$a/$tn/ \;
      [ -d $t/activity_definitions ] && cp -r $t/activity_definitions $D/inst/$a/$tn/
      for i in $ids; do cp $t/instances/*__$i[-.]* $D/inst/$a/$tn/instances/ 2>/dev/null || true; done
    done
  done
  touch $D/sampling.done
done
