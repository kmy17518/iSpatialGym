#!/bin/bash
# isgp_* (trainfast packing test, 2026-11-01): two fast-flow jobs (job_fast.sh: a sampler and its KCOLLECT collectors each) on ONE GPU.
# Each member runs exactly as job_fast.sh alone, under its own suffixed job id (<id>_a / <id>_b: its own /tmp job folder, Kit app data
# and exit cleanup, which job_fast.sh / env_fast.sh / kit_appdata.sh key on SLURM_JOB_ID) and half the CPUs. The pair's GPU-hours are
# this slurm job's wall (PAIR_END in each member's pair.txt), not the sum of the members' walls.
#   sbatch ... job_pair.sh <job dir a> <job dir b>   (job dirs and job.env by submit_pair.py)
#SBATCH -o /vision/u/minyeong/projects/isg_home/trainfast/slurm/%x_%j.log
set -u
F=/vision/u/minyeong/projects/isg_home/trainfast
A=$1
B=$2
ID=$SLURM_JOB_ID
HALF=$(( ${SLURM_CPUS_PER_TASK:-24} / 2 ))
T0=$(date +%s)
# 2026-11-02: each member on half of this job's physical cores (taskset; scripts/cpu_split.py halves), so its COLLECTOR_PIN (job_fast.sh)
# splits only its own half - on the full CPU list both members pinned their sampler and collectors onto the same CPUs. The 11-01 pk_pair*
# members ran before this, unpinned on all the CPUs. Fewer than two cores: both members on all the CPUs, COLLECTOR_PIN off
HALVES=( $(python3 $F/scripts/cpu_split.py halves) )
if [ ${#HALVES[@]} -eq 2 ]; then
  TA="taskset -c ${HALVES[0]}"; TB="taskset -c ${HALVES[1]}"
else
  TA=""; TB=""; export ISPATIALGYM_FAST_COLLECTOR_PIN=0
fi
for D in $A $B; do echo "PAIR_START $ID a=$A b=$B cpus_each=$HALF cpus_a=${HALVES[0]:-all} cpus_b=${HALVES[1]:-all} host=$(hostname -s) start=$(date -Is)" >> $D/pair.txt; done
( export SLURM_JOB_ID=${ID}_a SLURM_CPUS_PER_TASK=$HALF; exec $TA bash $F/scripts/job_fast.sh $A ) > $F/slurm/pair_${ID}_a.log 2>&1 &
PA=$!
sleep 30   # stagger the two members' Kit starts
( export SLURM_JOB_ID=${ID}_b SLURM_CPUS_PER_TASK=$HALF; exec $TB bash $F/scripts/job_fast.sh $B ) > $F/slurm/pair_${ID}_b.log 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
for D in $A $B; do echo "PAIR_END $ID wall_s=$(( $(date +%s) - T0 )) rc_a=$RA rc_b=$RB end=$(date -Is)" >> $D/pair.txt; done
