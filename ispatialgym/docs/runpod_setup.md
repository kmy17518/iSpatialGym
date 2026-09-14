# Running iSpatialGym on a RunPod pod

Validated 2026-09-14 on a RunPod **RTX 4090 (24 GB)** pod, RunPod PyTorch template
(Ubuntu 24.04, CUDA 12.8.1 image, driver 580.126.20), 16 vCPU / 61 GB RAM allotment,
data centre EU-RO-1. Everything below was run exactly as written; timings are from that run.

Summary: a fresh pod goes from empty to "sampler writes instances" in **~20 minutes**
(13 min of which is `setup.sh`). Nothing about the install is RunPod specific beyond the
SSH and storage notes below - the same steps work on any Linux box with an RTX GPU.

## 1. Pod / template settings

| Setting | Recommendation | Why |
| --- | --- | --- |
| GPU | RTX 4090 24 GB (or any RTX card, driver >= 535) | one sampler process peaks at 8-9.6 GB of GPU memory, so a 4090 runs **2 concurrent samplers**; a 48 GB card (L40S / A6000) runs 4 |
| Image | RunPod PyTorch template, Ubuntu 24.04 / CUDA 12.8.x | has the NVIDIA Vulkan + EGL ICDs Isaac Sim needs; only 3 small apt libs are missing (section 3) |
| Container disk | **120-150 GB** (500 GB is wasted; it is billed per GB) | env 20 GB + datasets 36 GB + pip/HF caches ~3 GB + repos 1 GB = ~60 GB; Isaac Sim keeps its caches here too |
| Network volume (`/workspace`) | optional; **do not** put the conda env or the live datasets on it | it is MooseFS: 1.1 GB/s sequential but ~9 ms per small file (800 asset files read in 7.2 s vs 0.04 s on local NVMe); a 136k-file dataset copies at ~60 MB/s and `du` is inflated ~2x by chunk granularity |
| RAM | the default 61 GB allotment is fine | one sampler process peaks at 8-9 GB RSS; `/dev/shm` 28 GB |
| Ports | expose **TCP 22** | the proxy SSH cannot scp or forward an agent (section 2) |
| Account SSH key | register your public key under RunPod *Settings -> SSH Public Keys* **before** creating the pod | otherwise the pod's `/root/.ssh/authorized_keys` literally contains `null` and direct SSH is refused |
| Template env vars | `OMNIGIBSON_HEADLESS=1`, `OMNI_KIT_ACCEPT_EULA=YES`, `HF_TOKEN=<hf token>` | headless Isaac Sim, no EULA prompt, authenticated (rate-limit-free) BEHAVIOR asset download from the HF Hub |

The container disk is wiped when the pod is stopped or terminated. Re-running `setup.sh`
costs 13 min, so the simplest workflow is *one pod = one install*. If you spin pods often,
bake the finished env into a custom Docker image/template (~25 GB) instead. A single
`tar` of `Behavior-1K/datasets` on the network volume restores in 1-2 min (sequential
reads are fast) versus ~5 min for the HF download - a marginal gain, only worth it if HF
is slow from the data centre.

## 2. Connecting

RunPod offers two SSH paths:

```bash
# (a) proxy - what the console shows; interactive only
ssh <pod-id>-<hash>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The proxy **requires a PTY** (`ssh host 'cmd'` fails with *"Your SSH client doesn't support
PTY"*; use `ssh -tt`), and does not support scp/sftp, port forwarding or agent forwarding.

```bash
# (b) direct TCP - full SSH; find IP/port in the console "Connect" panel or, inside the pod,
#     echo $RUNPOD_PUBLIC_IP $RUNPOD_TCP_PORT_22
ssh -A -p <RUNPOD_TCP_PORT_22> -i ~/.ssh/id_ed25519 root@<RUNPOD_PUBLIC_IP>
```

Use (b) for everything. `-A` forwards your local ssh-agent so the pod can clone the
**private** `kmy17518/iSpatialGym` repo over SSH without copying keys around:

```bash
# on your machine, once
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519
```

If (b) says `Permission denied (publickey)`, the account key was not propagated: connect
once through the proxy and install your public key by hand:

```bash
# in a proxy session on the pod
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA...your key...' > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

## 3. One-time pod preparation (~1 min)

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libegl1 libglu1-mesa libxt6 time   # Isaac Sim GL libs + /usr/bin/time

# Miniforge (conda-forge only, no Anaconda TOS) on the LOCAL disk
curl -fsSL -o /root/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /root/miniforge.sh -b -p /root/miniforge3
source /root/miniforge3/etc/profile.d/conda.sh

# HF token for the dataset download (also put it in the template env vars)
echo 'export HF_TOKEN=<hf token>' >> /root/.bashrc && chmod 600 /root/.bashrc && source /root/.bashrc
```

## 4. Clone + install (~13 min)

```bash
cd /root
git clone git@github.com:kmy17518/iSpatialGym.git ispatialgym     # needs the forwarded agent (private repo)
cd ispatialgym

# Long-running: run it detached so a dropped SSH session does not kill it.
nohup ./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --dev --primitives \
      --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos > /root/isg_install.log 2>&1 &
tail -f /root/isg_install.log        # Ctrl-C detaches from the log only
```

What `setup.sh` does on the pod, with the measured phases:

1. Clones `Behavior-1K/` (fork `kmy17518/BEHAVIOR-1K`, branch `ispatialgym`) - over SSH if
   an agent is forwarded, otherwise it falls back to HTTPS (the fork is public). ~30 s.
2. Upstream BEHAVIOR-1K installer: conda env `ispatialgym` (Python 3.11), torch 2.7.0+cu128,
   bddl, OmniGibson 3.9.x, Isaac Sim 5.1 wheels. ~8 min.
3. Core-dependency remediation, `pip install -e .` of this repo, `ispatialgym-assets`
   (Kohs block) into `Behavior-1K/datasets/`. ~30 s.
4. Datasets: robot assets 2.8 GB, BEHAVIOR-1K assets 33 GB, 2026 challenge instances 0.6 GB
   (36 GB total). ~4-5 min from EU-RO-1 (measured unauthenticated; `HF_TOKEN` removes the
   HF rate-limit warnings and protects against throttling).
5. cuRoboV2 0.8.0 with pinned build deps (see troubleshooting). ~1 min.

Disk after install: env 20 GB (`/root/miniforge3`), repo + datasets 48 GB (`/root/ispatialgym`).

## 5. Verify (~5 min)

```bash
source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym
cd /root/ispatialgym
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES

# no-simulator unit tests (79 tests, < 2 s)
pytest -q ispatialgym/tests/test_problem_parsing.py ispatialgym/tests/test_geometry_and_targets.py

# cuRobo v2 + its data files, grasp library
python -c "import curobo, os; print(curobo.__version__, os.path.exists(os.path.join(os.path.dirname(curobo.__file__), 'content/configs/task/ik/lbfgs_ik.yml')))"
python -c "from ispatialgym.sampling import grasp_library as GL; print(len(GL.load_grasps('pack_of_kielbasa__dgodnn__t')), 'grasps')"

# one instance each (launches Isaac Sim headless); ~1:40 wall each after the first launch
python -m ispatialgym.sampling.sample_instances --problem camera_relocalization-0 --n-instances 1 --max-templates 1 --seed 0 --out /root/isg-smoke --overwrite
python -m ispatialgym.sampling.sample_instances --problem configuration_matching-0  --n-instances 1 --max-templates 1 --seed 0 --out /root/isg-smoke --overwrite
```

Expected: `SAMPLING_REPORT ... "n_written": 1` for both; scene load ~48 s, first cuRobo
embodiment build 13-21 s (JIT), peak RSS 8-9 GB, peak GPU memory 8-9.6 GB. The very first
Isaac Sim launch on a pod spends ~3 extra minutes on shader/extension caches.

## 6. Running sampling jobs

```bash
# detach every long job; one Isaac Sim process per problem, at most 2 in parallel on a 24 GB card
nohup python -m ispatialgym.sampling.sample_instances --problem <activity>-<def_id> --n-instances N --seed S \
      --out /root/isg-out > /root/isg-<name>.log 2>&1 &
```

- Outputs: `<out>/<problem>/<template>/{template.json,instances/,activity_definitions/}`,
  `<out>/sampling_report_<epoch>.json`, `<out>/scene_variants/`.
- Copy results back over the TCP port: `scp -P <port> -r root@<ip>:/root/isg-out .`
  (or `huggingface-cli upload` with `HF_TOKEN`).
- CPU: a sampler uses ~3.5 cores; the 16-vCPU allotment is not the bottleneck.

## 7. Troubleshooting (all hit during the validation run)

| Symptom | Cause / fix |
| --- | --- |
| `Your SSH client doesn't support PTY` | proxy SSH; use `ssh -tt`, or the direct TCP port |
| direct SSH `Permission denied (publickey)` | `authorized_keys` is `null`; register the key in RunPod settings or install it via the proxy (section 2) |
| `FileNotFoundError: .../curobo/content/configs/task/ik/lbfgs_ik.yml` when the planner starts | cuRobo wheel built with setuptools_scm >= 10.3 has no `content/` data; `setup.sh` now builds with `--no-build-isolation` and `setuptools_scm<10.3`, verifies, and copies the files from a shallow clone if still missing. For an existing env: `pip install "setuptools_scm>=8,<10.3" && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.8.0 pip install --no-build-isolation --force-reinstall --no-deps "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo@v0.8.0"` (15 s), then re-run the `curobo.__version__` check of section 5 |
| `/usr/bin/time: No such file or directory` | `apt-get install time` (the PyTorch image lacks it) |
| HF warning `unauthenticated requests ... rate limits` | export `HF_TOKEN` |
| slow scene loads / installs | env or datasets on `/workspace` (network volume); keep them on the container disk |
| background job dies when the SSH session drops | `nohup ... &`; the forwarded agent also dies with the session, so start SSH-dependent clones in the foreground |
| `pip` warnings `isaacsim-kernel requires websockets==12.0 / packaging==23.0 ...` | expected; the upstream installer deliberately overrides these pins |
