#!/bin/bash
# build recollect queue dirs from crop failures: mk_recollect.sh <key> <reason> <dest tag>
set -eu
F=/vision/u/minyeong/projects/isg_home/trainfast
KEY=$1; REASON=$2; TAG=$3; SRCTAG=${4:-crop}
SRC=$(ls -d $F/out/$SRCTAG/$KEY/g* | head -1)
ids=$(grep -E "\"reason\": \"($REASON)\"" $SRC/results.jsonl | python3 -c "import sys,json; print(' '.join(json.loads(l)['instance'].rsplit('__',1)[1] for l in sys.stdin))")
echo "$KEY $REASON: $ids"
for side in partial full; do
  D=$F/out/$TAG/$KEY/$side
  mkdir -p $D/claims
  cp $SRC/job.env $D/job.env
  [ $side = full ] && echo "export ISPATIALGYM_FAST_PARTIAL_SCENE=0" >> $D/job.env
  grep "^ISPATIALGYM_" $SRC/job.env | sed 's/^/export /' >> $D/job.env
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
  [ -d $SRC/inst/../activity_definitions ] && true
  touch $D/sampling.done
  echo "$D: $(ls $D/inst/*/*/instances/ 2>/dev/null | wc -l) files"
done
