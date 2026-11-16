#!/bin/bash
# One Slurm GPU job: a policy server on the job's GPU (KIND pi | act | dp, CKPT = the step dir / .pt) and run_isg --instance-list over
# LIST in one world against it (the cluster form of eval_worker.py's unit).
#   sbatch ... --export=ALL,CODE=<code dir>,KIND=act,CKPT=<.pt>,SERVER_ARGS="--temporal-agg",LIST=<tros>,OUT=<dir>,ARGS_FILE=<run_isg args>,
#              POLICY_KWARGS_FILE=<json> job_policy.sh
set -uo pipefail
ARGS=$(cat "${ARGS_FILE:?args file}")
PK=$(cat "${POLICY_KWARGS_FILE:?policy kwargs json file}")
Q=/vision/u/minyeong/projects/isg-train224-eval
PORT=${PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 1000))}
case ${KIND:?kind} in
  pi)  (cd $Q/openpi && XLA_PYTHON_CLIENT_PREALLOCATE=false OPENPI_DATA_HOME=$Q/openpi_data HF_HUB_OFFLINE=1 exec .venv-jax/bin/python \
          scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/${TASK_ALIAS:-alignment-axial_01} --policy.config pi05_b1k \
          --policy.dir "${CKPT:?ckpt}" --action-horizon ${PI_HORIZON:-16} --port $PORT ${SERVER_ARGS:-}) > "$OUT.server.log" 2>&1 & ;;
  act) (cd $Q/act && exec .venv/bin/python scripts/b1k/serve_b1k.py --model-path "${CKPT:?ckpt}" --port $PORT ${SERVER_ARGS:-}) > "$OUT.server.log" 2>&1 & ;;
  dp)  (cd $Q/diffusion_policy && exec .venv/bin/python scripts/b1k/serve_b1k.py --model-path "${CKPT:?ckpt}" --port $PORT ${SERVER_ARGS:-}) > "$OUT.server.log" 2>&1 & ;;
esac
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
source "$CODE/scripts/policy_eval/job_env.sh"
trap 'kill $SERVER 2>/dev/null; kit_appdata_teardown' EXIT
export ISG_POLICY_URL=ws://127.0.0.1:$PORT
export ISG_TRAIN_TASKS=${ISG_TRAIN_TASKS:-/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1/merged/v2/train-lerobot/meta/tasks.jsonl}
for i in $(seq 1 180); do grep -q "listening on" "$OUT.server.log" 2>/dev/null && break; kill -0 $SERVER 2>/dev/null || { echo "SERVER_DIED"; tail -20 "$OUT.server.log"; exit 3; }; sleep 5; done
echo "=== $(date -u +%FT%TZ) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) server $KIND $CKPT on $PORT"
mkdir -p "$OUT"
t0=$(date +%s.%N)
python -u -m ispatialgym.run_isg --instance-list "$LIST" --out "$OUT" --policy ispatialgym.eval.ws_run_policy:WSPolicy --policy-kwargs "$PK" $ARGS
echo "JOBLIST_DONE rc=$? wall_s=$(python3 -c "import time; print(round(time.time() - $t0, 1))") list=$LIST"
KIT_RC=0
