#!/bin/bash
# Round D (2026-10-30): the 34 cluster aliases at N=8 with the final physics (sleep 0.005 + solver iterations [+ sleep assist]), N=1 parity vs runs_B.
#   roundD.sh "<ISG_EXTRA_ARGS>"      e.g. roundD.sh "--rigid-solver-iterations 64+8 --sleep-assist 30+1+3"
set -u
EXTRA=${1:?extra args}
V=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$V/campaigns/vec_next
B=/vision/u/minyeong/projects/Behavior-1K-vec-next-b-6a6194600
cd $C/jobs && sbatch -A viscam -p viscam --gres=gpu:1 --constraint=48G --array=0-33%10 \
  --export="ALL,ISG_TASK_LIST=$C/inputs/aliases_C_cluster.txt,ISG_B1K_DIR=$B,ISG_RUNS_DIR=runs_D,ISG_VERIFY_DIR=verify_D,ISG_REF_RUNS_DIR=runs_B,ISG_PHASES=n1+compare_n1+n8+compare,ISG_EXTRA_ARGS=$EXTRA" verify.sbatch
