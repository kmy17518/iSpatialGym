#!/bin/bash
# Upload an eval dataset folder to a PRIVATE Hugging Face dataset repo (scripts/batched_replay/upload_hf.sh with the policy-eval record's
# staging and credentials). CPU Slurm job (svl, never visionlab); only with the owner's go-ahead.
#   sbatch -A vision -p svl -c 8 --mem=16G -t 4:00:00 --exclude=visionlab-dgx1 -J isgeval_upload_hf -o $EV/logs/%x_%j.log \
#     --export=ALL,SRC=<dataset dir>,REPO=kmy17518/<name> upload_hf.sh
set -euo pipefail
source /vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1/tool/creds.sh >/dev/null
SRC=${SRC:?dataset dir}; REPO=${REPO:?hf repo id}
STAGE=$EV/hf_staging/${REPO//\//__}
mkdir -p "$(dirname "$STAGE")"
if [ -d "$STAGE" ] && [ "$(cat "$STAGE.src" 2>/dev/null)" != "$(readlink -f "$SRC")" ]; then rm -rf "$STAGE"; fi
[ -d "$STAGE" ] || { cp -al "$SRC" "$STAGE" && readlink -f "$SRC" > "$STAGE.src"; }
export STAGE REPO SRC PYTHONDONTWRITEBYTECODE=1 HF_HOME=${TMPDIR:-/tmp}/isgeval_hf_home_$$
/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python -u - <<'EOF'
import collections, os, time
from pathlib import Path
from huggingface_hub import HfApi
repo, stage, src = os.environ["REPO"], Path(os.environ["STAGE"]), Path(os.environ["SRC"])
local = sorted(str(p.relative_to(src)) for p in src.rglob("*") if p.is_file())
big = {d: n for d, n in collections.Counter(os.path.dirname(f) for f in local).items() if n > 10000}
if big:
    raise SystemExit(f"refusing: folders over 10,000 files {big}")
api = HfApi()
api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
assert api.dataset_info(repo_id=repo).private is True, f"refusing: {repo} is not private"
before = api.dataset_info(repo_id=repo).sha
t0 = time.time()
api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=stage, private=True, num_workers=8, print_report=True, print_report_every=60)
info = api.dataset_info(repo_id=repo, files_metadata=False)
remote = {s.rfilename for s in (info.siblings or [])}
missing = [f for f in local if f not in remote]
extra = sorted(f for f in remote - set(local) if f != ".gitattributes")
print(f"DONE in {(time.time() - t0) / 60:.1f} min: {len(remote)} remote / {len(local)} local files, missing {len(missing)}, extra {len(extra)}, "
      f"private {info.private}, revision {before} -> {info.sha}")
if extra or missing:
    raise SystemExit(f"Hub != source: missing {missing[:5]}, extra {extra[:5]}")
EOF
