#!/bin/bash
# Upload a merged batched-replay dataset (merge_train.py output) to a PRIVATE Hugging Face dataset repo (adapted from
# train85/work/vast_replay/bin/job_upload_hf.sh). Run as a CPU Slurm job; only with the owner's go-ahead.
#   sbatch -A vision -p visionlab -c 16 --mem=32G -t 12:00:00 -J isgrend_upload_hf -o $RV/logs/%x_%j.log \
#     --export=ALL,SRC=<dataset dir>,REPO=kmy17518/<name> upload_hf.sh
set -euo pipefail
source /vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1/tool/creds.sh >/dev/null
SRC=${SRC:?dataset dir}; REPO=${REPO:?hf repo id}
STAGE=$RV/hf_staging/$(basename "$SRC")
mkdir -p "$(dirname "$STAGE")"
# hard-link mirror so upload_large_folder's .cache/huggingface/upload stays out of the dataset
[ -d "$STAGE" ] || cp -al "$SRC" "$STAGE"
export STAGE REPO SRC PYTHONDONTWRITEBYTECODE=1
/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python -u - <<'EOF'
import os, time
from pathlib import Path
from huggingface_hub import HfApi
repo, stage, src = os.environ["REPO"], Path(os.environ["STAGE"]), Path(os.environ["SRC"])
local = sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())
api = HfApi()
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
assert api.dataset_info(repo_id=repo).private is True, f"refusing: {repo} is not private"
t0 = time.time()
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=stage, private=True, num_workers=8, print_report=True,
                        print_report_every=60)
info = api.dataset_info(repo_id=repo, files_metadata=False)
remote = {s.rfilename for s in (info.siblings or [])}
missing = [f for f in local if f not in remote]
print(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(remote)} remote / {len(local)} local files, missing {len(missing)}, private {info.private}")
if missing:
    raise SystemExit(f"missing on the Hub: {missing[:10]}")
EOF
