# Prompt 1 of 2 — set up an isolated `lab-vec-eval` worktree for the ISG lab evaluator on the cluster

You are a coding agent on the Stanford cluster (`sc.stanford.edu`, SLURM partition `viscam`). Build an isolated copy of the
ISG lab's evaluation stack — the one that runs on the lab machine `yhang-lambda` (RTX 4090) — in a new git worktree with its
own conda env, simulator checkout, ACT policy server and data, without touching anything that already runs here. Finish
with a smoke test and a report. Do not merge anything: merging and the full sanity battery are prompt 2
(`lab_tools/cluster/MERGE_SANITY_PROMPT.md`).

## What you are reproducing

| Component | Pin |
| --- | --- |
| iSpatialGym (`git@github.com:kmy17518/iSpatialGym.git`) | branch `origin/lab-vec-eval` = `221c6e03` (the code that collected the datasets `kmy17518/isg-init` and `isg-init-eval`) + 7 vectorized-evaluator commits + `lab_tools/` (the evaluation drivers, the evaluator patches, the ACT bridge, the gates) + `lab_tools/cluster/` (this kit) |
| BEHAVIOR-1K (`git@github.com:kmy17518/BEHAVIOR-1K.git`; OmniGibson 3.9.2, bddl 3.7.0) | `bc7180afce217f2af1dcd992c3c7a257e9f87720`, the parent of `origin/v2.0-vectorized`'s tip. Not the tip `a0e81fb68` (it changes the assisted-grasp release) |
| ACT policy server (`https://github.com/kmy17518/act.git`) | `978ae9bdb759fbbfed2f8f9394235784b7156dbb` (branch `isg-wave2-dev`). `lab/env.json` still says `8333d01`: stale, the lab's server runs `978ae9b` |
| Simulator env | Python 3.11.16, isaacsim 5.1.0.0, torch 2.7.0+cu128, torchvision 0.22.0+cu128, nvidia-curobo 0.8.0 (NVlabs `4ea77366`), numpy 1.26.0, warp-lang 1.12.1, av 15.1.0, pillow 11.0.0, datasets 4.8.5, lerobot (wensi-ai `436812bd`), msgpack 1.2.3, websockets 16.1.1, pytest. Full list: `lab_tools/cluster/env_snapshot/pip-freeze-ispatialgym-v2.0-vectorized.txt` |
| ACT venv | Python 3.10, torch 2.10.0+cu128, torchvision 0.25.0+cu128 (`lab_tools/cluster/env_snapshot/act-venv-packages.txt`) |
| BEHAVIOR assets | behavior-1k-assets 3.9.0rc9 on the lab machine (3.9.0 on the RunPod pods), omnigibson-robot-assets 3.8.2, 2026-challenge-task-instances, omnigibson.key: use the cluster's existing copy, the one its production jobs use |
| Test set | `kmy17518/isg-init-eval` at `c379f77dafa0054bcadc9e90d65260e00b06895b` (whole, ~2 GB) |
| Training set (part) | `kmy17518/isg-init` at `3d1e0fd661bc9f03c80057404f4cfbc642fc9092`: data, meta, isg_meta, the relocalization videos (~1.7 GB) |
| Instances + lab records | `isg-lab-eval-bundle-20261010.tar.gz` (73.7 MB; sha256 `767274304f5b388ee3b3085984f17baabaa14c67002cc950eb7993ef626f54e8`, or the `.sha256` file next to it if the user rebuilt it). The user copies it from the lab machine (`/tmp/v20eval-transfer/` there); ask where it is. It holds the TRO instance files `lab/instances.json` / `lab/instances-train.json` point at (regenerated on the lab machine, nowhere else), the restored training instances, the wrist goal PNGs, and `lab/` (instances maps, `env.json`, `notes.md` = the lab's full log of findings, `diagnostics/` = the reference records) |

## Hard rules (production runs on this cluster: nothing you do may affect it)

1. **Read-only**: `/vision/u/minyeong/projects/isg_prod/**`, every existing iSpatialGym / BEHAVIOR-1K checkout and code snapshot (e.g.
   `/vision/u/minyeong/projects/ispatialgym-vectorized-envs`, `isg_prod/code_*`), their conda envs, the shared BEHAVIOR datasets.
   No writes, no `pip install`, no `git checkout|reset|rebase|stash|clean|commit`, no scripts run from there.
2. **Processes / SLURM**: never cancel, signal or requeue jobs or processes you did not start (`squeue -u $USER` lists production
   jobs too). Never `pkill -f python|isaac|kit|omni`. Isaac Sim only on compute nodes you allocated (`srun` / `salloc` /
   `sbatch`), never on a login node; one Isaac Sim process at a time from this worktree.
3. **GPU**: Isaac Sim's RTX renderer needs RT cores, and the policy sees rendered images: use `viscam` nodes with feature `ada`
   (the lab machine's architecture) and at least 24 GB (`scripts/isg/slurm_sweep.py` lists the node features `24G / 48G,
   turing / ampere / ada`). Never A100 / H100 / V100. One Isaac Sim process takes ~10-15 GB of GPU memory and ~10 GB RAM.
4. **Git**: the object store is shared with the production checkouts. `git worktree add` is fine; never `git worktree
   remove|prune|move`, `git gc --prune`, `git reflog expire`, `git stash`; never check out, move or delete a branch that is
   checked out elsewhere. Do not push.
5. **BEHAVIOR-1K setup**: never pass `--dev` to `setup.sh` (BEHAVIOR-1K's setup runs `pre-commit install`, which rewrites the
   hook shared by every worktree of that repo) and never `--dataset`; never call `omnigibson.utils.asset_utils.download_*`.
6. **Credentials**: the Hugging Face token goes into `<WT>/.env` (gitignored, mode 600) as `HF_TOKEN=...`, taken from the
   cluster's existing login (`~/.cache/huggingface/token`) or from the user. Never print it, never commit it. No W&B.
7. **Never run here**: `lab_tools/tierb_loop.py` (the lab's production queue runner), `lab_tools/status_loop.sh` and
   `lab_tools/lab_sync.py upload|sync` (they upload `lab/` to the shared coordination repo on the Hub and would overwrite the
   lab machine's records), `lab_tools/*.sh` (they `cd` into the lab machine's path: do not "fix" them), anything under
   `scripts/runpod/`, `scripts/isg/slurm_sweep.py` / `prod_*.sh|py`, `scripts/isg/slack_bridge.py`, `scripts/slack_sidechat/*`.
8. **Disk**: check the quota first (~4 GB of data, ~20 GB for a conda env, ~6 GB for the ACT venv and a checkpoint).
9. When unsure whether an action reaches outside the new worktree, stop and ask the user.

## Step 0 — discover, then confirm with the user (read-only)

Collect and show:

- `hostname`; `sinfo -p viscam -o '%n %G %f %m %T' | head -30`; free space / quota where the worktree will live (`df -h`, `quota -s`).
- The iSpatialGym clone to add the worktree to (look under `/vision/u/minyeong/projects/`): `git -C <clone> remote -v`,
  `git -C <clone> worktree list`, `git -C <clone> branch -a --sort=-committerdate | head -20`, `git -C <clone> fetch --dry-run origin`
  (does GitHub access work from here?).
- The BEHAVIOR-1K clone the cluster uses and whether it has the pin: `git -C <b1k> cat-file -e bc7180af^{commit}`.
- The dataset root of the production jobs (`PROD_DATA_PATH`: `slurm_sweep.py` defaults it to `realpath(<repo>/Behavior-1K/datasets)`;
  the `PROD_JOB` lines of job logs print the code and commits) and the asset versions (`cat <root>/behavior-1k-assets/VERSION
  <root>/omnigibson-robot-assets/VERSION`).
- The production conda env (`PROD_ENV`; `conda env list`) and its versions:
  `<env>/bin/python -m pip list 2>/dev/null | grep -iE '^(isaacsim |torch |torchvision|nvidia-curobo|lerobot|numpy |msgpack|websockets|pytest)'`.
- Whether a Hugging Face login exists (`test -s ~/.cache/huggingface/token && echo yes`; never print the token), and `uv` /
  a Python 3.10 (`command -v uv python3.10`).

Then propose and get the user's OK for: the worktree path (default `/vision/u/minyeong/projects/ispatialgym-lab-vec-eval`), the
env name (`ispatialgym-lab-vec-eval`), env option A (clone the production env) or B (fresh build), the cache dir, the GPU
allocation, where the bundle is. Also ask which cluster branch will be merged later (prompt 2 needs it).

## Step 1 — worktree, simulator checkout, dataset links

```bash
ISG=<iSpatialGym clone>           # whose origin is kmy17518/iSpatialGym
WT=<confirmed worktree path>
B1K=<BEHAVIOR-1K clone>           # its origin kmy17518/BEHAVIOR-1K
D=<production dataset root>       # holds behavior-1k-assets, omnigibson-robot-assets, ...

git -C "$ISG" fetch origin lab-vec-eval
git -C "$ISG" worktree add -b lab-vec-eval "$WT" origin/lab-vec-eval      # a local lab-vec-eval already there: use another local name
git -C "$WT" log -1 --format='%H %s'                                       # the commit that added lab_tools/cluster/

git -C "$B1K" fetch origin v2.0-vectorized
git -C "$B1K" worktree add --detach "$WT/Behavior-1K" bc7180afce217f2af1dcd992c3c7a257e9f87720
# no usable clone: git clone git@github.com:kmy17518/BEHAVIOR-1K.git "$WT/Behavior-1K" && git -C "$WT/Behavior-1K" checkout --detach bc7180af...

mkdir -p "$WT/Behavior-1K/datasets"
for x in behavior-1k-assets omnigibson-robot-assets 2026-challenge-task-instances omnigibson.key; do ln -sfn "$D/$x" "$WT/Behavior-1K/datasets/$x"; done
```

`Behavior-1K/` is gitignored by iSpatialGym, so the outer worktree stays clean. The custom asset set `ispatialgym-assets` comes
from this worktree, never from `$D`: link it after step 2 (option B copies it there itself):
`ln -sfn "$WT/ispatialgym/assets/datasets/ispatialgym-assets" "$WT/Behavior-1K/datasets/ispatialgym-assets"`.

## Step 2 — conda env

Option A, when the production env matches the pins (Python 3.11, isaacsim 5.1.0.0, torch 2.7.0+cu128, nvidia-curobo 0.8.0):

```bash
source <conda base>/etc/profile.d/conda.sh
conda create -y -n ispatialgym-lab-vec-eval --clone <production env>
conda activate ispatialgym-lab-vec-eval
pip install --no-deps --no-build-isolation -e "$WT/Behavior-1K/bddl3" -e "$WT/Behavior-1K/OmniGibson" -e "$WT"
pip install msgpack==1.2.3 websockets==16.1.1 pytest            # the policy client and the unit tests, when missing
```

Option B, a fresh build (no `--dev`, no `--dataset`; `OMNIGIBSON_DATA_PATH` must be unset, or the script copies the custom assets
into that directory; run it before linking `ispatialgym-assets`):

```bash
cd "$WT" && env -u OMNIGIBSON_DATA_PATH ./setup.sh --new-env ispatialgym-lab-vec-eval --omnigibson --bddl --primitives --accept-conda-tos --accept-nvidia-eula
conda activate ispatialgym-lab-vec-eval && pip install msgpack==1.2.3 websockets==16.1.1 pytest
```

Then, for both options:

```bash
cp "$WT/lab_tools/cluster/env_snapshot/activate.d-ispatialgym_lab.sh" "$CONDA_PREFIX/etc/conda/activate.d/ispatialgym_lab.sh"
echo "export OMNIGIBSON_DATA_PATH=$WT/Behavior-1K/datasets" >> "$CONDA_PREFIX/etc/conda/activate.d/ispatialgym_lab.sh"
conda deactivate && conda activate ispatialgym-lab-vec-eval      # the hook: private caches, headless, Kit appdata on node-local scratch
mkdir -p "$ISPATIALGYM_REACH_BANDS" && cp -p "$WT"/runlogs/reach_bands_20260915/bands/* "$ISPATIALGYM_REACH_BANDS/"
cp -rp "$WT/runlogs/reach_volume_20260915" "$ISPATIALGYM_REACH_VOLUME"
```

Preflight (every shell, and before every Isaac Sim launch):

```bash
cd "$WT" && conda activate ispatialgym-lab-vec-eval
python -c "import ispatialgym, omnigibson, bddl; print(ispatialgym.REPO_ROOT); print(omnigibson.__file__); print(bddl.__file__)"   # all three under $WT
echo "$OMNIGIBSON_DATA_PATH $OMNIGIBSON_APPDATA_PATH $ISPATIALGYM_CACHE_DIR"
git branch --show-current; git -C Behavior-1K rev-parse HEAD                                                                      # lab-vec-eval, bc7180af...
python -m pip list 2>/dev/null | grep -iE '^(isaacsim |torch |nvidia-curobo|numpy |lerobot)'
```

Compare the env with `lab_tools/cluster/env_snapshot/pip-freeze-ispatialgym-v2.0-vectorized.txt` (ignore the `@ file:///tmp/...` wheel
paths and the editable paths) and list every version that differs in the report.

## Step 3 — ACT checkout and venv

```bash
git clone https://github.com/kmy17518/act.git "$WT/act-isg" && git -C "$WT/act-isg" checkout --detach 978ae9bdb759fbbfed2f8f9394235784b7156dbb
cd "$WT/act-isg" && PYTHON=3.10 LOCK=0 TORCH_INDEX=https://download.pytorch.org/whl/cu128 scripts/b1k/setup_venv.sh
```

`setup_venv.sh` needs `uv`; if it is missing, `pip install uv` inside `ispatialgym-lab-vec-eval` (do not run installers that edit
`~/.bashrc`). `LOCK=1` (the default) pins CUDA 13 wheels, which need a newer driver than cu128: keep `LOCK=0` unless the
driver is >= 580. Compare `.venv` with `lab_tools/cluster/env_snapshot/act-venv-packages.txt`.

## Step 4 — data

```bash
cd "$WT"
sha256sum -c <bundle dir>/isg-lab-eval-bundle-20261010.tar.gz.sha256      # run inside the bundle's directory
tar -xzf <bundle dir>/isg-lab-eval-bundle-20261010.tar.gz -C "$WT"        # -> datasets/{eval_1003_regen,eval_1003_cluster,train_regen,isg-reloc-train-instances,goals}, lab/
python - <<'EOF'                                                          # every file against the manifest, every TRO the maps name
import hashlib, json, os
m = json.load(open("datasets/lab-eval-bundle.manifest.json"))
bad = [e["path"] for e in m["files"] if hashlib.sha256(open(e["path"], "rb").read()).hexdigest() != e["sha256"]]
tros = [e["tro"] for n in ("lab/instances.json", "lab/instances-train.json") for e in json.load(open(n))["episodes"].values() if e.get("tro")]
print(len(m["files"]), "files,", len(bad), "bad;", len(tros), "TROs,", sum(not os.path.isfile(t) for t in tros), "missing")
EOF
install -m 600 /dev/null .env && printf 'HF_TOKEN=%s\n' "$(cat ~/.cache/huggingface/token)" > .env    # or the token the user gives you
act-isg/.venv/bin/python lab_tools/pull_datasets.py          # isg-init-eval whole + 2 files of isg-init at the pinned revisions, verified against the Hub
act-isg/.venv/bin/python - <<'EOF'                           # the rest of isg-init the lab uses (relocalization training episodes)
import os
for line in open(".env"):
    k, _, v = line.strip().partition("=")
    if k == "HF_TOKEN": os.environ["HF_TOKEN"] = v
from huggingface_hub import snapshot_download
snapshot_download("kmy17518/isg-init", repo_type="dataset", revision="3d1e0fd661bc9f03c80057404f4cfbc642fc9092", local_dir="datasets/isg-init",
                  allow_patterns=["README.md", "meta/*", "data/*", "isg_meta/*.json", "isg_meta/*.csv", "isg_meta/goals/index.json",
                                  "isg_meta/goals/camera_relocalization-standard/*", "videos/observation.rgb.zed_link_camera_0/chunk-000/*",
                                  "videos/observation.depth_linear.zed_link_camera_0/chunk-000/*",
                                  "videos/observation.rgb.left_realsense_link_camera_0/chunk-000/*",
                                  "videos/observation.rgb.right_realsense_link_camera_0/chunk-000/*"], max_workers=8)
EOF
act-isg/.venv/bin/python - <<'EOF'                           # the checkpoint of the closed-loop smoke test in prompt 2
import os
for line in open(".env"):
    k, _, v = line.strip().partition("=")
    if k == "HF_TOKEN": os.environ["HF_TOKEN"] = v
from huggingface_hub import hf_hub_download
print(hf_hub_download("kmy17518/act-isg-r1-early-copy05-st0-s0", "eval/step-00100000.pt", local_dir="act-isg/ckpts/r1-early-copy05-st0-s0"))
EOF
echo "629210205d12fbe984a43415c2b705de8d87a402be711a82628ca74bf9be950f  act-isg/ckpts/r1-early-copy05-st0-s0/eval/step-00100000.pt" | sha256sum -c
```

`lab/`, `datasets/`, `act-isg/`, `sanity/` and `.env` are gitignored on this branch; `git status` must stay clean apart from them.

## Step 5 — static checks (no GPU)

```bash
python lab_tools/cluster/check_eval_surface.py --base origin/lab-vec-eval      # expect: patch targets OK, lab-owned files identical, 0 changed, PASS
python -m pytest -q -p no:cacheprovider ispatialgym/tests/test_vec_stream.py ispatialgym/tests/test_problem_parsing.py ispatialgym/tests/test_geometry_and_targets.py
#   expect 631 passed (the lab machine's count)
DRY_RUN=1 lab_tools/cluster/sanity_battery.sh dry && rm -rf sanity/dry      # prints the battery's commands
```

## Step 6 — GPU smoke test (an allocated `ada` node)

```bash
srun -p viscam -A viscam --gres=gpu:1 --constraint=ada -c 8 --mem=48G -t 2:00:00 --pty bash     # adapt account / features to what step 0 found
# on the node: preflight, then
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
python lab_tools/replay_episodes.py --split eval50 --dataset datasets/isg-init-eval --instances lab/instances.json --task camera_relocalization-standard \
    --episodes 2 --horizon 1500 --no-planner --tag setup-smoke --lab-out sanity/setup/records/replay-eval50/camera_relocalization-standard \
    --raw-out sanity/setup/raw/replay-eval50/camera_relocalization-standard
python lab_tools/vec_eval.py --mode replay --split eval50 --dataset datasets/isg-init-eval --instances lab/instances.json --task camera_relocalization-standard \
    --episodes 2 --num-envs 1 --tag setup-smoke --lab-out sanity/setup/records/vec/reloc-n1 --raw-out sanity/setup/raw/vec/reloc-n1
python lab_tools/vec_gate_3b.py --root sanity/setup/records/vec --runs reloc-n1 --ref sanity/setup/records/replay-eval50/camera_relocalization-standard \
    --out sanity/setup/gate3b.json                                                               # expect PASS: N=1 bit-identical to the single-env replay
python lab_tools/cluster/compare_replays.py --a sanity/setup/records --expected lab_tools/cluster/expected_replays.json   # expect ep 2 ok
```

Expected for episode 2 (lab machine): success, first success at step 718, 863 steps in total, final position error ~4.8 cm.
Another machine may differ by a few steps (`compare_replays.py` allows 3). The first launch on a node builds Kit's shader
cache (~3-5 min); the world build takes ~2 min.

## Step 7 — report to the user

- Paths: worktree, BEHAVIOR-1K checkout (and the clone it belongs to), dataset root, conda env, cache dir, appdata dir, bundle.
- `git -C $WT log -1`, `git -C $WT/Behavior-1K rev-parse HEAD`, `git -C $WT/act-isg rev-parse HEAD`.
- Every package version that differs from the pins, and the asset versions.
- Node, GPU and driver of the smoke test; its counters next to the expected ones; gate 3b; time per episode.
- Anything you did differently from this prompt, and why. Then wait for prompt 2.
