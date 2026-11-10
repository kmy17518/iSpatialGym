# i7-vast: sourced on the cluster (svl11 or a slurm CPU job of the coordinator), NEVER on a Vast instance. Loads the shared
# credentials (VAST_GUIDE.md section 3) into the shell without printing them and sets this job's names / prefixes.
#   JOB_ID   og-isg261110 (instance label prefix og-isgof-; one job for the whole round-4 Vast backend)
#   VJ       $OFH/vast: queue/ state/ ledger/ logs/ market/ bundle/ results/ (cluster record of the backend)
#   JOB_S3   s3://isg-demos/jobs/$JOB_ID: bundle/ jobs/ assign/ hb/ out/ done/ logs/ ctl/ (R2, this job only)
#   REPLAY_S3 the replay prefix, READ ONLY (env parts, assets, textures, kit templates, the asset key)
set +x
OFH=/vision/u/minyeong/projects/isg_home/campaign_20261110
VJ=$OFH/vast
REPLAY_WORK=/vision/u/minyeong/projects/isg_home/train85/work/vast_replay
export JOB_ID=og-isg261110 OFH VJ REPLAY_WORK
set -a; source /vision/u/minyeong/.config/isg-demos/.env; set +a
export AWS_CONFIG_FILE=/vision/u/minyeong/.aws/config AWS_SHARED_CREDENTIALS_FILE=/vision/u/minyeong/.aws/credentials
export AWS_REGION=auto AWS_DEFAULT_REGION=auto
export PATH=$REPLAY_WORK/tools/bin:$REPLAY_WORK/tools/venv/bin:$PATH
export XDG_CONFIG_HOME=$VJ/.config XDG_CACHE_HOME=$VJ/.cache DOCKER_CONFIG=$VJ/.docker
S3_ENDPOINT_URL=$("$REPLAY_WORK/tools/venv/bin/python" - <<'PY'
import configparser, os
c = configparser.ConfigParser(); c.read(os.environ["AWS_CONFIG_FILE"])
p = os.environ["AWS_PROFILE"]
print(c[f"profile {p}" if f"profile {p}" in c else p]["endpoint_url"])
PY
)
export S3_ENDPOINT_URL
export JOB_S3=s3://$S3_BUCKET/jobs/$JOB_ID REPLAY_S3=s3://$S3_BUCKET/isg-1020-b1k
VPY=$REPLAY_WORK/tools/venv/bin/python
