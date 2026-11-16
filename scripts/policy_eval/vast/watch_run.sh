#!/bin/bash
# One line per new worker event of a policy-eval run (boot / checkpoint / server / unit start, done, fail / self-destroy), read from R2.
#   source $EV/tool/creds.sh; bash watch_run.sh <run> [interval_s]
RUN=${1:?run}; EVERY=${2:-90}
T=/tmp/isgeval_watch_$RUN; mkdir -p $T; touch $T/seen
while true; do
  $S5 cp "$JOB_S3/runs/$RUN/logs/*/worker.log" "$T/" >/dev/null 2>&1
  $S5 cp "$JOB_S3/runs/$RUN/logs/*/boot.log" "$T/" >/dev/null 2>&1
  for f in $(find $T -name 'worker.log' -o -name 'boot.log' 2>/dev/null); do
    inst=$(basename $(dirname $f))
    grep -h -E "WORKER |CKPT |SERVER |START |DONE |FAIL |KILL |SELF_DESTROY|WORKER END|unusable|FETCH_FAILED|checkpoint .* (ready|failed)|servers of|policy env|self-test|bundle unpacked" $f \
      | sed "s|^|$inst |" ; done | sort -u > $T/now
  comm -13 <(sort $T/seen) $T/now | cut -c1-260
  cp $T/now $T/seen
  n_done=$($S5 ls "$JOB_S3/runs/$RUN/done/*" 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M)] done=$n_done" > $T/status
  sleep $EVERY
done
