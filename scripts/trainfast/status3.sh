#!/bin/bash
# crop3 / full3 / ev_off: one status line, then per key-alias attempts/written for the split-cone aliases, tracebacks, committed-band writes
F=/vision/u/minyeong/projects/isg_home/trainfast
TAGS="crop4 full4 pilot2 pilot3 pilot4" bash $F/scripts/status.sh
echo "committed_band_changes=$(cd /vision/u/minyeong/projects/ispatialgym-train-fast && git status --short runlogs/ | wc -l)"
for k in disembedding_03 alignment-axial_01 configuration_matching-attachment_02 configuration_matching-block_design_01 inter_object_relation_satisfaction-inside_02; do
  line="$k:"
  for t in crop4 full4 ev_off rule2; do
    J=$(ls -d $F/out/$t/$k/g* 2>/dev/null | head -1); [ -z "$J" ] && continue
    line="$line $t $(grep -c ' written (attempt' $J/sample.log 2>/dev/null)/$(grep -c 'rejected in\| written (attempt' $J/sample.log 2>/dev/null)"
  done
  echo "$line"
done
for t in crop3 full3 ev_off; do for J in $F/out/$t/*/g*; do
  n=$(cat $J/sample.log $J/collect_*.log 2>/dev/null | grep -c "^Traceback"); [ "$n" -gt 0 ] && echo "TB $t $(basename $(dirname $J)) $n: $(cat $J/sample.log $J/collect_*.log 2>/dev/null | grep -A4 '^Traceback' | grep -E 'Error|Exception' | sort -u | head -2 | tr '\n' ' ' | cut -c1-200)"
done; done
