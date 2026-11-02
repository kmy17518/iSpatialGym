#!/bin/bash
# Round F (2026-10-31): the 53 aliases at N=8 with option 3 (clone caches: world volume, collision_mesh_cpu_data; lazy clone emitters)
# on top of round E's code; N=1 parity vs runs_E. Aliases 0-38 on svl Titan RTX, 39-52 on viscam 48 GB (svl QOS cap 48 GPUs/user).
# Plus option 1 on the cluster: the mp phase (K processes x N scenes) for apple on a Titan RTX and on an L40S.
set -u
V=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$V/campaigns/vec_next
B=/vision/u/minyeong/projects/Behavior-1K-vec-next-b-6a6194600
EXP="ISG_TASK_LIST=$C/inputs/aliases.txt,ISG_B1K_DIR=$B,ISG_RUNS_DIR=runs_F,ISG_VERIFY_DIR=verify_F,ISG_REF_RUNS_DIR=runs_E,ISG_PHASES=n1+compare_n1+n8+compare,ISG_EXTRA_ARGS=--sleep-assist 30+1+3"
cd $C/jobs
sbatch -A vision -p svl --gres=gpu:titanrtx:1 --array=0-38 --export="ALL,$EXP" verify.sbatch
sbatch -A viscam -p viscam --gres=gpu:1 --constraint=48G --array=39-52 --export="ALL,$EXP" verify.sbatch
MP="ISG_TASK=object_scaling-apple,ISG_B1K_DIR=$B,ISG_RUNS_DIR=mp_cluster_titan,ISG_VERIFY_DIR=verify_mp,ISG_PHASES=mp,ISG_MP_LAYOUTS=1x8+2x4+4x2,ISG_EXTRA_ARGS=--sleep-assist 30+1+3"
sbatch -A vision -p svl --gres=gpu:titanrtx:1 -c 16 --mem=128G -J vnmp_titan --export="ALL,$MP" verify.sbatch
MP2="ISG_TASK=object_scaling-apple,ISG_B1K_DIR=$B,ISG_RUNS_DIR=mp_cluster_l40s,ISG_VERIFY_DIR=verify_mp,ISG_PHASES=mp,ISG_MP_LAYOUTS=1x8+2x4+4x2+8x1,ISG_EXTRA_ARGS=--sleep-assist 30+1+3"
sbatch -A viscam -p viscam --gres=gpu:l40s:1 -c 16 --mem=128G -J vnmp_l40s --export="ALL,$MP2" verify.sbatch
