#!/bin/bash
#SBATCH -J isgof_vastctl
#SBATCH -A vision
#SBATCH -p svl
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 3-00:00:00
#SBATCH --exclude=svl9
#SBATCH -o /vision/u/minyeong/projects/isg_home/oracle_fix/vast/logs/%x_%j.log
# i7-vast: the ONE coordinator of the Vast backend as a slurm CPU job (never an ssh session):
#   sbatch $OFH/vast/tool/job_coord.sh          (scancel <its job id> stops it; rented instances keep running: vastctl.py stop-all)
VTOOL=${VTOOL:-/vision/u/minyeong/projects/isg_home/oracle_fix/vast/tool}
VTOOL=$(readlink -f $VTOOL)   # this job keeps the tool copy it started with
source $VTOOL/creds.sh
echo "COORD $SLURM_JOB_ID tool=$VTOOL ($(cat $VTOOL/.commit)) host=$(hostname -s) start=$(date -Is)"
exec $VPY -u $VTOOL/vastctl.py loop
