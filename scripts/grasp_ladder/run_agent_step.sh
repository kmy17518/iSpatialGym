#!/bin/bash
# Grasp ladder step 3 unattended, from lambda (codex is installed and logged in here; the cluster runs everything else):
#   run_agent_step.sh <batch name> <frozen snapshot sha> "<manifests>" "<glib roots>" ["<exclude ids>"]
# 1 cluster CPU job: select every entry at 0 physics-valid after the given rounds + prepare its inputs / strips (job_prepare.sh);
# 2 the batch's JSON / PNG (no meshes) to lambda; 3 codex proposes per entry (agent_step.py propose --runner codex: codex exec
#   --approve-for-me, the contract's JSON schema, answers checked, one retry); 4 the proposals back; 5 cluster: the proposals through
#   the same gates + physics (one glib job, a NEW root), waited on by its explicit job id; 6 the report (proposals -> gates -> valid).
set -euo pipefail
NAME=$1; SHA=$2; MANIFESTS=$3; ROOTS=$4; EXCLUDE=${5:-}
H=minyeong@svl11
O=/vision/u/minyeong/projects/isg_home/oracle_fix
S=$O/code/trunk_$SHA
B=$O/ladder_agent/$NAME
OUT=$O/ladder_agent/${NAME}_build
LB=${AG_LOCAL:-$HOME/oracle_fix/ladder_agent}/$NAME
PY=${AG_PY:-/home/minyeongk/miniconda3/envs/ispatialgym/bin/python}
here=$(cd "$(dirname "$0")" && pwd)
wait_job() { while ssh $H "sacct -j $1 -X -n -o State" | grep -qE "PENDING|RUNNING|REQUEUED"; do sleep 60; done; ssh $H "sacct -j $1 -X -n -o JobID,State,ExitCode"; }
ssh $H "mkdir -p $B $O/slurm"
j=$(ssh $H "AG_BATCH=$B AG_MANIFESTS='$MANIFESTS' AG_ROOTS='$ROOTS' AG_EXCLUDE='$EXCLUDE' ISG_GB_ROOT=$S sbatch --parsable -A vision -p svl --exclude=svl9 -c 4 --mem=16G --time=04:00:00 -J isgof_agent_prep -o $O/slurm/isgof_agent_prep_%j.out $S/scripts/grasp_ladder/job_prepare.sh")
echo "PREPARE_JOB $j"; wait_job $j
mkdir -p $LB
rsync -a --include='entries.json' --include='CONTRACT.md' --include='contract.schema.json' --include='inputs/***' --include='strips/***' --exclude='*' $H:$B/ $LB/
$PY $here/agent_step.py propose --batch $LB --runner codex
rsync -a $LB/proposals $H:$B/
ssh $H "cd $O && python3 $S/scripts/grasp_ladder/agent_step.py submit --batch $B --code $S --out $OUT --run" | tee $LB/submit.log
jid=$(grep -oP 'AGENT_JOB \K[0-9]+' $LB/submit.log || true)
[ -n "$jid" ] && wait_job $jid
ssh $H "cd $O && python3 $S/scripts/grasp_ladder/agent_step.py report --batch $B --out $OUT" | tee $LB/report.log
