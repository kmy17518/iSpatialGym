#!/bin/bash
# Status of a policy-eval run from R2: done units per checkpoint, each instance's last worker events and heartbeat.
#   source $EV/tool/creds.sh; bash run_status.sh <run> [instance ...]
RUN=${1:?run}; shift
echo "done: $($S5 ls "$JOB_S3/runs/$RUN/done/*" 2>/dev/null | awk '{print $NF}' | sed 's/__.*//' | sort | uniq -c | tr '\n' ' ')"
for i in "$@"; do
  echo "== $i"
  $S5 cat "$JOB_S3/runs/$RUN/logs/$i/worker.log" 2>/dev/null | grep -E "DONE|FAIL|unusable|CKPT|WORKER END" | tail -6 | cut -c1-240
  $S5 cat "$JOB_S3/runs/$RUN/hb/$i.json" 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d['phase'], 'gpu', d.get('gpu_util'), d.get('gpu_mem_mb'), 'cpu', d.get('cpu_busy'), 'done', d.get('done'), 'failed', d.get('failed'),
      [(r.get('unit'), r.get('phase'), r.get('done_instances')) for r in d.get('running', [])])"
done
