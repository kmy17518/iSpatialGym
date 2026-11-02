#!/bin/bash
# Round E (2026-10-31): all 53 aliases at N=8 on svl Titan RTX (24 GB, driver 580) with the restore-sequence fix (one world step for
# the batch instead of one per scene reset; vec_task.py + vec_loader.py) and the sleep-assist contact-window fix (run_isg_vec.py).
# N=1 parity vs runs_D (expected bit-identical except where the assist used to sleep an object right at its release).
#   roundE.sh            (submit only after the three files are synced to $EVAL and the lambda restore probe showed slot k == N=1)
set -u
V=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$V/campaigns/vec_next
B=/vision/u/minyeong/projects/Behavior-1K-vec-next-b-6a6194600
cd $C/jobs && sbatch -A vision -p svl --gres=gpu:titanrtx:1 --array=0-52%16 \
  --export="ALL,ISG_TASK_LIST=$C/inputs/aliases.txt,ISG_B1K_DIR=$B,ISG_RUNS_DIR=runs_E,ISG_VERIFY_DIR=verify_E,ISG_REF_RUNS_DIR=runs_D,ISG_PHASES=n1+compare_n1+n8+compare,ISG_EXTRA_ARGS=--sleep-assist 30+1+3" verify.sbatch
