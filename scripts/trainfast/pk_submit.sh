#!/bin/bash
# trainfast packing test (owner item 2, 2026-11-01 evening): more processes per GPU. Arms per key, n=24 per sampler, latest code ($W):
#   pk_k2_<gpu>    sampler + 2 collectors (the default; control)      12 CPUs / 32G
#   pk_k3_<gpu>    sampler + 3 collectors                             16 CPUs / 48G
#   pk_pair1_<gpu> 2 x (sampler + 1 collector) on one GPU (job_pair)  24 CPUs / 64G
#   pk_pair2_<gpu> 2 x (sampler + 2 collectors) on one GPU            24 CPUs / 96G
# gpu = titan (svl TITAN RTX 24 GB) or l40s (viscam L40S 48 GB). Keys: ontop_small_01 (sampler-bound), disembedding_03 (collector-
# bound), mental_rotation-mug (fast sampler, short demos).
#   pk_submit.sh singles | pairs
F=/vision/u/minyeong/projects/isg_home/trainfast
W=/vision/u/minyeong/projects/ispatialgym-train-fast
KEYS="inter_object_relation_satisfaction-ontop_small_scale_01 disembedding_03 mental_rotation-mug"
LIM="--sample-limit-s 16000 --time-s 21600"
TITAN="--gres titanrtx --acct vision --part svl"
L40S="--gres l40s --acct viscam --part viscam"
cd $F
case "$1" in
  singles)
    for k in $KEYS; do
      for r in 1 2; do python3 scripts/submit_fast.py submit $k 24 --flow fast --code $W --kcollect 2 --cpus 12 --mem 32G $LIM $TITAN --tag pk_k2_titan; done
      python3 scripts/submit_fast.py submit $k 24 --flow fast --code $W --kcollect 3 --cpus 16 --mem 48G $LIM $TITAN --tag pk_k3_titan
      python3 scripts/submit_fast.py submit $k 24 --flow fast --code $W --kcollect 2 --cpus 12 --mem 32G $LIM $L40S --tag pk_k2_l40s
      python3 scripts/submit_fast.py submit $k 24 --flow fast --code $W --kcollect 3 --cpus 16 --mem 48G $LIM $L40S --tag pk_k3_l40s
    done ;;
  pairs)
    for k in $KEYS; do
      [ "$k" = mental_rotation-mug ] || python3 scripts/submit_pair.py $k 24 --code $W --kcollect 1 --cpus 24 --mem 64G $LIM $TITAN --tag pk_pair1_titan
      python3 scripts/submit_pair.py $k 24 --code $W --kcollect 1 --cpus 24 --mem 64G $LIM $L40S --tag pk_pair1_l40s
      python3 scripts/submit_pair.py $k 24 --code $W --kcollect 2 --cpus 24 --mem 96G $LIM $L40S --tag pk_pair2_l40s
    done ;;
esac
