#!/bin/bash
# Grasp ladder step 3, the cluster's CPU part (select + prepare --regen) as its own slurm job (pam_slurm_adopt: never in an ssh session):
#   AG_BATCH=<batch dir> AG_MANIFESTS="<m.json ...>" AG_ROOTS="<glib root ...>" [AG_EXCLUDE="<id ...>"] ISG_GB_ROOT=<frozen snapshot> \
#     sbatch -A vision -p svl --exclude=svl9 -c 4 --mem=16G --time=04:00:00 -J isgof_agent_prep -o <log> job_prepare.sh
set -uo pipefail
source ${ISG_GB_ROOT:?}/scripts/grasp_bands/env_cluster.sh
EX=(); [ -n "${AG_EXCLUDE:-}" ] && EX=(--exclude $AG_EXCLUDE)
python -u $ISG_GB_ROOT/scripts/grasp_ladder/agent_step.py select --manifest $AG_MANIFESTS --roots $AG_ROOTS --batch ${AG_BATCH:?} "${EX[@]}"
python -u $ISG_GB_ROOT/scripts/grasp_ladder/agent_step.py prepare --batch $AG_BATCH --regen
echo "AGENT_PREPARE_JOB_DONE rc=$?"
