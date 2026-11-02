#!/bin/bash
# partial-scene validation: 4 replays per alias once both its partial and full collections ended (marker replays.submitted)
#   rp_full = partial demos --scene full; rp_rec = partial demos --scene recorded; rf_rec = full demos --scene recorded;
#   rp_full_delta = partial demos --scene full --delta-load
F=/vision/u/minyeong/projects/isg_home/trainfast; W=/vision/u/minyeong/projects/ispatialgym-train-fast
for K in $F/out/pv/*; do
  k=$(basename $K); [ -e $K/replays.submitted ] && continue
  grep -q ISGC_END $K/partial/job.txt 2>/dev/null && grep -q ISGC_END $K/full/job.txt 2>/dev/null || continue
  short=$(echo $k | sed "s/inter_object_relation_satisfaction-//; s/configuration_matching-//" | cut -c1-12)
  sub() { REPLAY_EXTRA="$3" sbatch --parsable --export=ALL -J isgr_pv_${short}_$2 -A vision -p svl --gres=gpu:titanrtx:1 -c 12 --mem=40G --time=02:00:00 \
            $F/scripts/job_replay_multi.sh $K/$1 $W 1 $K/$2; }
  ids="$(sub partial rp_full '--scene full') $(sub partial rp_rec '--scene recorded') $(sub full rf_rec '--scene recorded') $(sub partial rp_full_delta '--scene full --delta-load')"
  echo "$ids" > $K/replays.submitted; echo "$k: $ids"
done
