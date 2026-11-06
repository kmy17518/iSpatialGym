#!/bin/bash
# one line: per tag written / demos / successes / tracebacks / ended, replay tests done, my queue
F=/vision/u/minyeong/projects/isg_home/trainfast
c(){ if [ -f "$2" ]; then grep -c "$1" "$2"; else echo 0; fi; }
out="$(date +%H:%M)"
for t in ${TAGS:-crop crop_off}; do
  W=0; D=0; S=0; T=0; E=0; N=0
  for J in $F/out/$t/*/g*; do
    case $J in *.cancelled) continue;; esac
    [ -f $J/job.txt ] || continue
    N=$((N+1)); W=$((W+$(c " written (attempt" $J/sample.log))); D=$((D+$(c instance $J/results.jsonl))); S=$((S+$(c '"success": true' $J/results.jsonl)))
    T=$((T+$(cat $J/sample.log $J/collect_*.log $J/demo_q*.log 2>/dev/null | grep -c "^Traceback"))); E=$((E+$(c "^ISGF_END" $J/job.txt)))
  done
  out="$out | $t j=$N w=$W d=$D s=$S tb=$T end=$E"
done
r=$(ls $F/out/replay_test/*/job.txt 2>/dev/null | xargs grep -l ISGR_END 2>/dev/null | wc -l)
out="$out | replay_done=$r/$(ls -d $F/out/replay_test/* | wc -l) | queue R=$(squeue -u minyeong -h -t R -o %j | grep -c ^isg) PD=$(squeue -u minyeong -h -t PD -o %j | grep -c ^isg)"
echo "$out"
