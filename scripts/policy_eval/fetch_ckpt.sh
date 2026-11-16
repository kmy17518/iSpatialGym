#!/bin/bash
# Fetch eval checkpoints of the isg-train224 runs from Hugging Face into $CK/<run>/ (pi: the step tar extracted to step-XXXXXXXX/,
# with .ready; ACT / DP: the .pt). Size checked against the uploader journal (.b1k-uploader.json) when it records one.
#   fetch_ckpt.sh <run>:<step> ...        e.g. nov15-pi05-isg224-goal-gbs256-s0:10000 nov15-act-isg224-goal-gbs1024-s0:80000
set -uo pipefail
CK=${CK:-/vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1/ckpt}
for spec in "$@"; do
  run=${spec%%:*}; step=${spec##*:}; s8=$(printf %08d $step)
  mkdir -p $CK/$run
  case $run in *pi05*) name=eval/step-$s8.tar ;; *) name=eval/step-$s8.pt ;; esac
  dst=$CK/$run/$(basename $name)
  if [ -f $CK/$run/step-$s8/.ready ] || { [ -f $dst ] && [[ $dst == *.pt ]]; }; then echo "HAVE $run $step"; continue; fi
  t0=$SECONDS
  want=$(curl -sSL --max-time 60 "https://huggingface.co/kmy17518/$run/resolve/main/.b1k-uploader.json" | python3 -c "
import json,sys
try:
  d=json.load(sys.stdin); print(int(d['eval']['$step']['size']))
except Exception: print(0)" 2>/dev/null)
  curl -sSL --retry 5 --retry-delay 10 -C - -o $dst.part "https://huggingface.co/kmy17518/$run/resolve/main/$name" || { echo "FAIL download $spec"; continue; }
  have=$(stat -c %s $dst.part)
  if [ "${want:-0}" -gt 0 ] && [ "$have" != "$want" ]; then echo "FAIL size $spec $have != $want"; continue; fi
  mv $dst.part $dst
  if [[ $dst == *.tar ]]; then
    mkdir -p $CK/$run/step-$s8 && tar -xf $dst -C $CK/$run/step-$s8 && rm -f $dst && touch $CK/$run/step-$s8/.ready || { echo "FAIL extract $spec"; continue; }
  fi
  echo "OK $spec $((SECONDS - t0)) s $(du -sh $CK/$run | cut -f1)"
done
