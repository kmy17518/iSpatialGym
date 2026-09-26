# RunPod operations for iSpatialGym (pool + band fleet)

How we create, use, schedule, drain and shut down GPU pods. Written 2026-09-17 after the first-demo pass; every command is what
was actually used. Scripts: `scripts/runpod/` (`rp.py`, `remote_run.py`, `dispatch_samp.py`, `collect.py`, `samp_summary.py`),
`scripts/reach_bands/` (`fleet_remaining.py`, `collect_bands.py`, `pull_loop.sh`, `fleet_run.py`), `scripts/isg/` (`first_demo.py`,
`pod_scheduler.py`). All run from the repo root in the `ispatialgym` conda env; the RunPod API key is read from the env file
(`rp.load_env()`), never passed on a command line.

## 1. Create, bootstrap, verify

```bash
python scripts/runpod/rp.py create --gpu "NVIDIA L40S" --cloud SECURE --name smoke2-l40s --count 5 --ttl-h 12 --purpose smoke
python scripts/runpod/rp.py create --gpu "NVIDIA A40"  --cloud SECURE --name pilot2-a40  --count 4 --ttl-h 12 --purpose samp
python scripts/runpod/rp.py bootstrap <pod id> --ref survey --ttl-h 12 --timeout 1500      # in parallel per pod
python scripts/runpod/rp.py run <pod id> "nvidia-smi; python -c 'import torch; print(torch.cuda.is_available())'"
```

- GPU / cloud: SECURE cloud with A40 or L40S allocates reliably; 4090s are scarce and some hosts (driver 595/610, RTX PRO 6000 Blackwell)
  segfault Isaac Sim at start → `create` keeps an `allowedCudaVersions` filter (12.8 / 12.9 / 13.0). `--cuda` restricts further.
- `--ttl-h` is enforced by the guard (`watchdog`, §2); `--purpose` (`smoke` = pool, `samp` = sampling/bands) is recorded in
  `runlogs/runpod/fleet.json` with name, cost/h, created_at.
- `bootstrap` installs the env, datasets and the grasp library from the repo ref, then writes `/root/BOOTSTRAP_DONE` (`/root/BOOTSTRAP_FAILED`
  on error). Pitfall: one pod had no usable CUDA device after a successful bootstrap — always verify with `nvidia-smi` + the torch probe
  before adding a pod to any pod file.
- Verify the thread cap too (§9): after `conda activate ispatialgym` on the pod, `taskset -cp $$` must list `0-<floor(quota)-1>` and
  `$ISG_THREAD_CAP` must be set: `rp.py run <pod> "bash -c 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && echo \$ISG_THREAD_CAP && taskset -cp \$\$'"`.
  `bootstrap.sh` does not install the hook yet (pending as of 2026-09-25): install it by hand. If the first Isaac launch segfaults, run it
  once under gdb (§9) before judging the host.
- Pods do **not** have the local `~/.cache/ispatialgym` caches or anything under `/tmp`; inputs a run needs go under
  `runlogs/smoke_inputs/` (git-ignored, synced by `remote_run.py`) and are referenced repo-relative.
- Cost: A40 $0.49/h, L40S $1.09/h (SECURE, Sep 2026). 27 pods ran at ≈ $21/h; the owner recharges the RunPod balance in small amounts —
  `rp.py balance` prints balance, fleet spend/h, spent_total vs the cap and the recharge count; each call also appends to
  `runlogs/runpod/billing.log`.

## 2. list / balance / the guard

```bash
python scripts/runpod/rp.py list
python scripts/runpod/rp.py balance
nohup python scripts/runpod/rp.py --cap 1000 watchdog --interval 300 --idle-min 240 --idle-grace-min 60 >> runlogs/runpod/watchdog.log 2>&1 &
```

The guard enforces the TTL of every pod we created and logs the balance every interval. **It never terminates for budget**: the cap
(`DEFAULT_CAP_USD`, $1000 since 2026-09-16) is a `create` guard (projected spend > cap → refuse) and a log line; `--enforce-cap` exists
but stays off — on 2026-09-15 a $200 default cap killed a whole sampling fleet mid-run (owner decision 21:27). Idle termination
(`--idle-min`) fires only for pods that report `/root/JOB_DONE` with no sampler / bench / bootstrap process alive; with pods reused across
roles we run it at 240 min so the scheduler, not the guard, decides. Stop it with `kill $(pgrep -f 'rp.py --cap 1000 watchdo[g]')` — and
never put the word `watchdog` after `rp.py` in the same command line as that `pgrep` (self-match, §8).

## 3. Roles — never both on one pod

- **Pool** (`runlogs/runpod/smoke/pods.txt`): one job at a time per pod, enforced by a `flock` on `runlogs/runpod/smoke/locks/<pod>.lock`
  held by `remote_run.py` for the whole run. Used by `first_demo.py`, agent probes, A/B experiments.
- **Band fleet** (`runlogs/runpod/rb/pods.txt` + the role file `runlogs/runpod/pods_bands.txt`): a pod-side runner (`/root/samp/run_jobs.sh`,
  written by `dispatch_samp.py`) consumes `/root/samp/queue/*.job` sequentially (reach bands, coverage sampling chunks) and writes
  `/root/samp/<job id>/`, `done.txt`, `current.txt`.
- A pod is in exactly one file at a time. Role changes move the pod between the files (the scheduler does this, §5) and are logged in
  `TODO.md` with the pod id and the jobs moved; `runlogs/pilot_20260915/paused_for_resume.md` records paused chunks.

## 4. One job on a free pool pod — `remote_run.py`

```bash
python scripts/runpod/remote_run.py --timeout 5400 --pull '$SMOKE_OUT' runlogs/<experiment>/<run>/ -- \
  'python -m ispatialgym.sampling.sample_isg --alias <A> --templates 1 --instances-per-template 1 --max-attempts 8 --seed 0 --no-write-back --out $SMOKE_OUT/inst; \
   tro=$(ls $SMOKE_OUT/inst/*/*/instances/*-tro_state.json | head -1); python -m ispatialgym.oracle.run_isg --instance "$tro" --mode oracle --out $SMOKE_OUT/demo --video --video-size 540'
```

What it does: picks the first pool pod whose lock is free and is `BOOTSTRAP_DONE` (waits up to `--wait` 3600 s), holds the lock,
rsyncs the **working tree** (`ispatialgym/`, `scripts/`, `runlogs/smoke_inputs/`, the reach-volume / bands / inventory runlogs;
uncommitted edits included — a half-edited file breaks the run on the pod, commit small), runs the command on the pod with
`OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES SMOKE_OUT=/root/smoke/<run id>` under `timeout`, pulls `--pull REMOTE LOCAL` back
(`$SMOKE_OUT` is substituted), then holds the lock while any process of the run is alive (≤ 15 min, then kills them) so a backgrounded
command cannot leave two simulators on one pod. `remote.log` in the pulled dir starts with `remote_run: pod <id> (<ip>:<port>), run id <id>`.
**SIGTERM / SIGINT handler** (added 2026-09-16 17:58): kills the pod-side tree (`pkill -KILL -f '/root/smoke/<run id>'`) and the local
ssh before releasing the lock — so cancelling a run means signalling `remote_run.py`, not its parent. A plain `kill` of the parent
(`first_demo.py`) leaves an orphan on the pod. To override the customs for an experiment without touching the shared file:
`export ISG_CUSTOMS=runlogs/smoke_inputs/<exp>/customs.json` inside the remote command (`sample_isg --customs` also exists).

## 5. `first_demo.py` and `pod_scheduler.py`

`first_demo.py run --alias A [--scene S] [--max-attempts 8] [--timeout 7200] [--local] [--note ..]` = sample one instance + oracle demo on a
pool pod (or `--local` on the 4090: one at a time, ≥ 12 GB free), ledger row in `runlogs/pilot_20260915/first_demo/ledger.jsonl`, board
regenerated; `demo-only --tro <tro>`; `fix --commit <sha> --subtasks .. --what .. --owner .. [--kind fix|facts|note|icl]`; `board`.
Validity = success + every *targeted* fix (< 6 subtasks) affecting the subtask in the run's commit (D-005; blanket entries → `valid (pre-<sha>)`,
final sweep at the end).

`pod_scheduler.py loop --interval 300` (one instance; `tick [--dry-run]` for one pass) keeps pods busy: reads the pool locks and the band
pods' state (`/root/samp/running/*.job` = in progress), maintains `runlogs/pilot_20260915/ready_queue.json` (a subtask enters when every fix
affecting it is in HEAD and its last run lacks the newest targeted one; `add --alias .. --priority N [--scene S] [--local] [--kind demo-only --tro ..]`
adds explicitly; `hold --subtask S --reason ..` / `release` for owner decisions — a targeted fix logged after a hold lifts it), launches one
`first_demo.py` per free pool pod (one per subtask per tick; in-flight runs of any agent are detected by `pgrep`), refills idle band pods from
`runlogs/reach_bands_20260915/active_manifests.txt` (one job per pod, via `fleet_remaining.py` + `dispatch_samp.py`, never `--rebalance`),
moves roles (band → pool when nothing is unassigned; pool → band only while nothing waits for a pod, ≤ 6 per tick), **preempts** band pods
for first-demo work (idle first, then the most recently started job: runner + `reach_bands` killed, job files parked in
`/root/samp/queue_preempted/`, `current.txt` cleared, pod → pool; the job stays in the remaining set), writes `idle_pods.md` (pod, role,
idle since, why nothing was schedulable; band pods "preemptible") and logs everything in `pod_scheduler.log` + `TODO.md`.
`cancel --subtask S|--entry ID|--pid P --reason ..` SIGTERMs the remote_run child first, waits ≤ 90 s, then the parent. Stop the loop with
`pkill -f 'pod_scheduler.py loo[p]'` **in its own command** (§8), then relaunch in a separate command.

## 6. Band jobs — dispatch, drain, collect

```bash
python scripts/reach_bands/fleet_remaining.py --manifest jobs_a.json jobs_b.json --pods-file runlogs/runpod/rb/pods.txt \
    --pulled runlogs/runpod/rb --out runlogs/reach_bands_20260915/jobs_remaining.json     # done = a .npz exists; running = running/*.job
python scripts/runpod/dispatch_samp.py --manifest jobs_remaining.json --pods-file runlogs/runpod/rb/pods.txt --only <pod ids> --wave 40
python scripts/runpod/dispatch_samp.py --status --pods-file runlogs/runpod/rb/pods.txt
nohup scripts/reach_bands/pull_loop.sh runlogs/runpod/rb/pods.txt 600 rb_only > runlogs/runpod/rb/pull_loop.log 2>&1 &
python scripts/reach_bands/collect_bands.py --manifest <all manifests> --pulled runlogs/runpod/rb --out runlogs/reach_bands_20260915
```

`dispatch_samp.py` exports the committed HEAD (`git archive`) to the pods (`--code-ref`), writes deterministic queue files
`w<wave>_<k>_<job id>.job` (weight order), starts the runner if it is not alive (`--restart-runner [--force]` restarts it; `--close` lets
runners exit when their queue is empty → `/root/JOB_DONE` → `collect.py --purpose rb --remote-dir /root/samp` pulls and terminates).
**Never `--rebalance`** on a shared fleet (it drops every queued job on the ready pods). To move queued jobs: remove the specific
`queue/*.job` files on the source pod, dispatch the same ids to the target pods. `collect_bands.py` copies the band files into
`bands/` and regenerates `results.json` / `results.csv` / the README table; the `--pulled` root must have the layout `<root>/<pod name>/<job id>/`
(symlink other layouts into a temp root).

## 7. What is a pod doing, and is it idle?

- Pool: `runlogs/runpod/smoke/locks/<pod>.lock` — try `flock` non-blocking (free ⇒ idle), or `lsof -t <lock>` for the holder pid
  (its parent is the `first_demo.py` / agent shell that owns the run). `pgrep -af remote_run.py` lists local runs (parent pid = owner).
- Band: `/root/samp/running/*.job` exists ⇔ a job is in progress. **`current.txt` is stale between jobs** (written at start, cleared only
  at `CLOSE`) and `done.txt` keeps earlier attempts of a re-queued job — never read idleness from them.
- On the pod: `ps -eo pid,etimes,args | grep -E 'sample_is[g]|run_is[g]|reach_band[s]'`, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
- An orphan = a pod-side simulator with no live local `remote_run` holding that pod's lock; kill it with `pkill -KILL -f '<run id or alias>[x]'`
  using the bracket trick, after pulling its `/root/smoke/<run id>`.

## 8. Terminate safely

1. Stop the schedulers / loops (`pod_scheduler.py loop`, `pull_loop.sh`, the Slack status loop) — each `pkill` in its own command.
2. Let first-demo runs finish (they pull their outputs); for other live `remote_run`s send SIGTERM **after** pulling their run dirs.
3. Collect everything: `rsync -az --exclude rb_cache/ root@<ip>:/root/smoke/ runlogs/runpod/collected_<date>/<name>/smoke/` (and `/root/samp/`),
   then verify per pod: a pod-side `find -type f -printf '%s\t%p\n'` manifest vs the local files and `rsync --dry-run --checksum` showing
   nothing left (the shutdown of 2026-09-17 used `/tmp/collect_all.py`, kept in `runlogs/runpod/collected_20260917/collection_results.json`
   form). Re-run `collect_bands.py` over all manifests + the collected dirs; check that every ledger row's `out/` exists locally.
4. Terminate only pods whose table shows 0 missing: `python scripts/runpod/rp.py terminate <id> --reason "..."` (records `terminated_at` and a
   `TERMINATE` billing line); kill the guard; `rp.py list` must show no pods; record the final balance / spent_total in `TODO.md`.
5. Empty `smoke/pods.txt`, `rb/pods.txt`, `pods_bands.txt`; delete `smoke/locks/*.lock`. The local GPU is untouched.

## 9. Pitfalls (all measured)

- **`pgrep -f` / `pkill -f` self-match**: the pattern (and any literal text matching it, e.g. the relaunch command) is part of the calling
  shell's command line. Use `loo[p]`-style brackets, keep the kill and the relaunch in separate commands, never echo a matching word in the
  same command. This killed the coordinator's own shell three times on 2026-09-16/17 and once the guard was killed together with it.
- `current.txt` staleness / `done.txt` history (§7): 12 of 15 idle band pods once read as "running".
- Killing a `first_demo.py` parent leaves the pod-side simulator running (the 17:29 orphan); signal `remote_run.py` instead (`cancel`).
- Working-tree sync: a file half-edited at sync time breaks the run on the pod (two runs lost); the code the pod runs is whatever the tree
  holds at launch — record the commit (`first_demo.py` does) and commit small.
- Two roles on one pod = two simulators (25 GB) and a wrong idle picture; the role files are the truth, the scheduler rewrites them — do
  not edit them by hand while the loop runs (a concurrent tick re-wrote a manual edit once).
- Bootstrapped pods cannot pull the private repo (token shredded): code reaches them by rsync only.
- A run dir pulled to `/tmp` is lost with the shell; pull to `runlogs/` (git-ignored subdirs are fine).
- 4090 / driver 610 hosts segfault Isaac Sim; EUR pods pull GitHub at 120 KB/s; vCPU allotment varies 9–36 per pod; one process per pod is
  the throughput optimum (the sampler is CPU-bound).
- The B4 inventory's `front_free_m` for cabinets came from the no-object traversability map (bamfsz_0: 3.0 m claimed, 0.0 m real).
- **CPU quota vs visible CPUs** (2026-09-25, large_articulation_integration_20260925): a pod container runs under a CFS CPU quota
  (`/sys/fs/cgroup/cpu/cpu.cfs_quota_us` / `cpu.cfs_period_us`: 13.6 CPUs on most L40S / 4090 hosts, 10.2-27.2 seen) but sees every host CPU
  (96-256), so torch / numba / OpenBLAS / Kit size their pools to the host and the container was throttled in 57-100 % of its 100-ms windows
  (`nr_throttled` / `nr_periods` in `cpu.stat`). Pinning to floor(quota) CPUs plus library caps: 0 % throttling, instance load 570 -> 331 s,
  sim step 0.59 -> 0.25 s/step (`runlogs/runpod/artic_integration/threads_ab/README.md`); library caps alone leave Kit's `carb.tasking` /
  `tbb.worker` pools at full size. Mechanism: the conda activation hook `/root/miniforge3/envs/ispatialgym/etc/conda/activate.d/isg_threads.sh`
  (thread caps floor(quota) - 1, `taskset -cp 0-<floor(quota)-1> $$`, `ISG_THREAD_CAP`), sourced by every `remote_run` command; a capped demo
  records `ISG_THREAD_CAP` in `meta.json` `rush_env`.
- **The first Isaac launch on a fresh pod can segfault** (2026-09-25): right after the bootstrap, launches segfaulted in
  `SimulationApp._start_app` within ~10 s, before any Kit log (two L40S hosts, drivers 580.159 / 570.124), for plain `isaacsim.SimulationApp`
  and OmniGibson alike. One launch of each under gdb (`gdb -batch -ex run --args python <smoke script>`) warmed its caches, and every later launch
  worked. Not a bad host: do not terminate the pod for it.
