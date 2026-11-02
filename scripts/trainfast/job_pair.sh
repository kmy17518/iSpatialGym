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
for D in $A $B; do echo "PAIR_START $ID a=$A b=$B cpus_each=$HALF host=$(hostname -s) start=$(date -Is)" >> $D/pair.txt; done
( export SLURM_JOB_ID=${ID}_a SLURM_CPUS_PER_TASK=$HALF; exec bash $F/scripts/job_fast.sh $A ) > $F/slurm/pair_${ID}_a.log 2>&1 &
PA=$!
sleep 30   # stagger the two members' Kit starts
( export SLURM_JOB_ID=${ID}_b SLURM_CPUS_PER_TASK=$HALF; exec bash $F/scripts/job_fast.sh $B ) > $F/slurm/pair_${ID}_b.log 2>&1 &
PB=$!
wait $PA; RA=$?
wait $PB; RB=$?
for D in $A $B; do echo "PAIR_END $ID wall_s=$(( $(date +%s) - T0 )) rc_a=$RA rc_b=$RB end=$(date -Is)" >> $D/pair.txt; done
