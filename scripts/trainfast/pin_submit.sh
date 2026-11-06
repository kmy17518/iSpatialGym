#!/bin/bash
# trainfast E16 job-level A/B (2026-11-02): COLLECTOR_PIN (job_fast.sh: the sampler and each collector on CPUs of their own,
# ISPATIALGYM_FAST_COLLECTOR_CPUS=3 each) against the unpinned k2 / k3 jobs of the 11-01 packing fleet (pk_*) and a fresh unpinned
# control; n=24 per job as the fleet. Tags pin_k2_titan, pin_k3_titan, ctl_k2_titan, pin_k2_l40s, pin_k3_l40s.
# `pin_submit.sh pair` (04:42, after COLLECTOR_PIN went default on and job_pair.sh put each member on half the cores): one pinned pair of
# de03 members with 2 collectors each on an L40S against the fleet's unpinned pk_pair2_l40s, tag pinpair2_l40s.
F=/vision/u/minyeong/projects/isg_home/trainfast
W=/vision/u/minyeong/projects/ispatialgym-train-fast
LIM="--sample-limit-s 16000 --time-s 21600"
TITAN="--gres titanrtx --acct vision --part svl"
L40S="--gres l40s --acct viscam --part viscam"
PIN="--extra-env ISPATIALGYM_FAST_COLLECTOR_PIN=1"
NOPIN="--extra-env ISPATIALGYM_FAST_COLLECTOR_PIN=0"   # the control ran with no switch while the default was off (02:34)
DE=disembedding_03
cd $F
S() { python3 scripts/submit_fast.py submit "$@" --flow fast --code $W $LIM; }
if [ "${1:-}" = pair ]; then
  python3 scripts/submit_pair.py $DE 24 --code $W --kcollect 2 --cpus 24 --mem 96G $LIM $L40S --tag pinpair2_l40s
  exit
fi
for r in 1 2; do S $DE 24 --kcollect 2 --cpus 12 --mem 32G $TITAN $PIN --tag pin_k2_titan; done
S inter_object_relation_satisfaction-ontop_small_scale_01 24 --kcollect 2 --cpus 12 --mem 32G $TITAN $PIN --tag pin_k2_titan
S mental_rotation-mug 24 --kcollect 2 --cpus 12 --mem 32G $TITAN $PIN --tag pin_k2_titan
S $DE 24 --kcollect 3 --cpus 16 --mem 48G $TITAN $PIN --tag pin_k3_titan
S $DE 24 --kcollect 2 --cpus 12 --mem 32G $TITAN $NOPIN --tag ctl_k2_titan
S $DE 24 --kcollect 2 --cpus 12 --mem 32G $L40S $PIN --tag pin_k2_l40s
S $DE 24 --kcollect 3 --cpus 16 --mem 48G $L40S $PIN --tag pin_k3_l40s
