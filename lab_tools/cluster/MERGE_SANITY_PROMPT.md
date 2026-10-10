# Prompt 2 of 2 — merge a cluster branch with `lab-vec-eval` and verify the evaluation code

You are the coding agent on the cluster that ran prompt 1 (`lab_tools/cluster/SETUP_PROMPT.md`): the worktree `$WT` on branch
`lab-vec-eval` exists, its env `ispatialgym-lab-vec-eval` is set up, the bundle and the datasets are in place and the smoke test
passed. The hard rules of prompt 1 still apply (production is read-only; no `git add -A`, no push; no `setup.sh --dev`;
never run `tierb_loop.py`, `status_loop.sh`, `lab_sync.py upload|sync`, `lab_tools/*.sh`, `scripts/runpod/*`, the production
sweep scripts).

Inputs from the user: **`CLUSTER_BRANCH`**, the branch to merge with `lab-vec-eval` (ask if it was not given; also ask whether it
has uncommitted work in its own checkout - that is not part of a merge, and you never commit there).

## What can break, and how this test finds it

The lab evaluates with the single-env evaluator (`run_isg --mode policy`, driven by `lab_tools/eval_episodes.py` /
`replay_episodes.py`) plus `lab_tools/evaluator_patches.py`, and has a vectorized evaluator (`ispatialgym/oracle/vec_loader.py`,
`vec_stream.py`, `ispatialgym/tasks/vec_task.py`, `lab_tools/vec_eval.py`) validated against it. A merge can break it in
three ways, none of which necessarily raises an error:

1. **Patch targets**: the patches replace `qa.Monitors.check`, wrap `Executor.epilogue`, set three `ExecutorConfig` fields, swap
   `run_isg.R1ProCuRobo` for a stub, subclass `run_isg.InstanceLoader`, call `execute._patch_ag_joint_type` (the
   `fixed_handle_ag` grasp rule every stream path uses). A rename or a moved import makes a patch a silent no-op.
2. **Evaluation semantics**: the other branch changes the single-env stream path itself (a new verdict in `Executor.epilogue`,
   a release / contact check in `Executor.step`, a new restore in `InstanceLoader`, success logic in `ISpatialGymTask`). The
   evaluator then judges policies differently from the lab's records - and the vectorized classes, which re-implement or
   override exactly these methods, no longer agree with the single-env path.
3. **Simulator**: the other branch expects another BEHAVIOR-1K commit than `bc7180af`.

The test: (A) the full battery on the unmerged `lab-vec-eval`, (B) the merge, (C) a static check of the evaluator surface,
(D) the same battery on the merge on the same GPU type, (E) comparisons. Open-loop replays are bit-deterministic on one machine,
so base vs merged must be identical record by record unless the merge changed the evaluator path - and then the static check
names the change. Closed-loop policy outcomes are not reproducible even on one machine (RTX real-time path tracing is
stochastic across processes; the lab measured a 0.14-0.29 per-episode flip rate between two runs of the same code), so they are
a plumbing check only.

Tools (all in `lab_tools/cluster/`, docstrings have the details): `sanity_battery.sh <label>` (static checks, single-env replays of
every task's demos, vectorized replays + gates 3b / 3d, closed loop through both paths; ~2-3 h on an RTX 4090-class GPU),
`check_eval_surface.py` (patch targets, lab-owned files, drift of every function the lab code uses), `compare_replays.py`
(record-by-record comparison; `--expected expected_replays.json` = the lab machine's counters).

Run the batteries as batch jobs on the same node type (ideally the same node), e.g.

```bash
sbatch -p viscam -A viscam --gres=gpu:1 --constraint=ada -c 8 --mem=48G -t 6:00:00 -J isg-sanity-base -o "$WT/sanity/%x-%j.out" \
  --wrap "source <conda base>/etc/profile.d/conda.sh && conda activate ispatialgym-lab-vec-eval && cd $WT && lab_tools/cluster/sanity_battery.sh base"
```

## Phase A — baseline on the unmerged `lab-vec-eval`

```bash
cd "$WT" && git switch lab-vec-eval && git status --short --untracked-files=no    # must be empty
git -C Behavior-1K rev-parse HEAD                                                   # bc7180afce217f2af1dcd992c3c7a257e9f87720
lab_tools/cluster/sanity_battery.sh base                                            # as a batch job (above)
python lab_tools/cluster/compare_replays.py --a sanity/base/records --expected lab_tools/cluster/expected_replays.json --out sanity/compare_base_vs_lab.json
```

The base must be healthy before you merge: every step of `sanity/base/steps.jsonl` exit 0, the gates PASS, every replay a
success (train episode 98 may end `tolerance`: it ends 0.07 mm from the 5 cm boundary), counters within 3 steps of the lab's.
If the base fails, stop: that is an environment problem (prompt 1), not the merge. Exception, and say so in the report:
`vec-banana-n1` / `vec-can-n1` / their gate 3b are first-time checks - the vectorized path has never replayed the in-hand
tasks (a grasp restored per slot) on the lab machine; a failure there is a known gap, not a regression.

## Phase B — preview, then merge

```bash
git fetch origin
CB=<CLUSTER_BRANCH ref>      # origin/<name> when pushed, or the local branch of the clone this worktree belongs to (refs are shared)
git log --oneline lab-vec-eval.."$CB" | head -50; git diff --stat lab-vec-eval..."$CB" | tail -40
python lab_tools/cluster/check_eval_surface.py --base lab-vec-eval --target "$CB" --diff --out sanity/preview_surface.json
git merge-base lab-vec-eval "$CB"; git diff --name-only 221c6e03 lab-vec-eval      # the files the lab side changed
git switch -c "merge/${CB##*/}+lab-vec-eval" "$CB"
git merge --no-ff lab-vec-eval -m "Merge lab-vec-eval ($(git rev-parse --short lab-vec-eval)) into ${CB##*/} ($(git rev-parse --short "$CB"))"
```

The preview runs before anything changes: it lists what the merge will change in the evaluator surface (symbols in files the
lab side also edits are marked: those are decided by the real merge). In `git merge`, "ours" is `CLUSTER_BRANCH`, "theirs" is
`lab-vec-eval`. Conflict rules:

- Lab-owned files take `lab-vec-eval`'s version (`git checkout --theirs -- <path>`): `ispatialgym/oracle/vec_loader.py`,
  `vec_stream.py`, `ispatialgym/tasks/vec_task.py`, `ispatialgym/eval/clone_frames.py`, `ispatialgym/tests/test_vec_stream.py`,
  `lab_tools/**`.
- `ispatialgym/sampling/predicates.py` (the attachment context keyed by scene index) and `ispatialgym/tasks/__init__.py` (the
  `VecISpatialGymTask` export): keep both sides' changes.
- The shared evaluator files (`ispatialgym/oracle/{run_isg,executor,execute,qa,recorder,behavior_lerobot,motions,programs,actions}.py`,
  `ispatialgym/tasks/ispatialgym_task.py`, `ispatialgym/eval/tro.py`): the lab side did not change them after `221c6e03` (check
  with `git diff 221c6e03 lab-vec-eval -- <file>`), so a conflict there comes from the history between `221c6e03` and the
  other branch: resolve it with that branch's intent and write down what you chose.
- Binary production data under `runlogs/` (e.g. reach-band `.npz` added on both sides): `CLUSTER_BRANCH`'s version
  (`git checkout --ours -- <path>`); the evaluator does not read them.
- `.gitignore`: the union.
- Stage only resolved paths (`git add <path>`), never `git add -A` / `git add .` / `git commit -a`; before committing,
  `git diff --cached --stat` and `git status` must show no `lab/`, `datasets/`, `act-isg/`, `sanity/`, `.env`.

Simulator: find the BEHAVIOR-1K commit `CLUSTER_BRANCH` expects (the `PROD_JOB ... b1k_commit=` lines of its production job logs,
its docs, the `Behavior-1K/` of its checkout or snapshot). Keep `$WT/Behavior-1K` at `bc7180af` for Phase D; if the branch
expects another commit, Phase D2 tests it.

## Phase C — static check of the merge

```bash
python -m compileall -q ispatialgym lab_tools
python lab_tools/cluster/check_eval_surface.py --base lab-vec-eval --diff --out sanity/merged_surface.json
python -m pytest -q -p no:cacheprovider ispatialgym/tests/test_vec_stream.py ispatialgym/tests/test_problem_parsing.py ispatialgym/tests/test_geometry_and_targets.py
```

Patch targets must be OK and the lab-owned files identical (else fix the merge). For every CHANGED symbol read the diff and
classify it:

- (a) not on the stream path (oracle planning, sampling, recording of demos): note it;
- (b) on the single-env stream path: it changes how the evaluator judges policies and replays. List each with the tasks it can
  affect;
- (c) marked "mirrored by" / "overridden by": the vectorized counterpart must get the same change, or single-env and vectorized
  verdicts diverge.

Known case: `origin/v2.0-vectorized` (5 commits after `221c6e03`, 2026-10-03) adds a `release_not_commanded` verdict to
`Executor.epilogue` and release tracking to `Executor.step`: an assisted-grasp release that does not happen at a gripper
command of exactly +1 fails the episode. If `CLUSTER_BRANCH` contains it, single-env policy and replay streams get this verdict
(policies output continuous gripper commands: the in-hand tasks mental_rotation-banana / object_scaling-canned_food and the
blender lid are exposed), while `VecStreamExecutor` and the lab's records do not have it. Whether evaluation keeps that check
(then mirror it in `vec_stream.py`) or drops it for streams (then disable it in `lab_tools/evaluator_patches.py` next to the
other stream-QA patches) is the owner's decision: report it with the diff and Phase E's evidence; do not decide it. Do the same
for any other (b) / (c) change. Code changes you make on the merge branch (only after the user agrees) go in separate commits
on top of the merge commit, followed by Phases C-E again.

## Phase D — the battery on the merge

```bash
lab_tools/cluster/sanity_battery.sh merged       # same node type as base, same BEHAVIOR-1K (bc7180af)
```

D2, only when `CLUSTER_BRANCH` expects another BEHAVIOR-1K commit: `git -C "$WT/Behavior-1K" fetch origin && git -C "$WT/Behavior-1K"
checkout --detach <that commit>` (this checkout is the worktree's own), preflight, `lab_tools/cluster/sanity_battery.sh merged-b1k
static replay vec`, then compare `merged` with `merged-b1k` like Phase E: what differs comes from the simulator (e.g.
`a0e81fb68` wakes the object an assisted grasp releases: in-hand tasks). Put `bc7180af` back afterwards unless the user decides
otherwise.

## Phase E — compare and decide

```bash
python lab_tools/cluster/compare_replays.py --a sanity/base/records --b sanity/merged/records --strict --out sanity/compare_base_vs_merged.json
python lab_tools/cluster/compare_replays.py --a sanity/merged/records --expected lab_tools/cluster/expected_replays.json --out sanity/compare_merged_vs_lab.json
python lab_tools/cluster/compare_replays.py --summarize sanity/base/closedloop sanity/merged/closedloop
```

| Check | Pass |
| --- | --- |
| Static (Phase C) | patch targets OK; lab-owned files identical; compileall clean; no test fails (631 pass on the lab machine; the merge may add tests) |
| Replays on the merge | every episode succeeds and reached the goal (train 98: either verdict); within 3 steps of the lab's counters |
| Base vs merge, same machine (`--strict`) | every record identical: counters and per-step arrays. A difference passes only when a (b) change from Phase C explains it (tasks, `first_diff_step`); an unexplained difference fails |
| Vectorized gates on the merge | `gate3b_reloc` PASS (N=1 bit-identical to the single-env replay; N=4 counters identical, trajectories within 5 mm / 5 mrad); `gate3b_align` / `_dish` / `_blend` PASS (N=1 bit-identical); `gate3d_reloc` PASS (first frames within 20 of the dataset's, typically ~6); banana / can: report |
| Closed loop | both paths run end to end, 0 errors, every record complete. Successes are reported, not gated: this checkpoint on training episodes 101 / 105 / 111 / 113 / 121 with default execution got 0-2 of 5 on the lab machine (~0 on test episodes) |

## Phase F — report to the user

- The merge: branch name, merge commit, both parents, conflicts and how each was resolved, the BEHAVIOR-1K commit(s) tested.
- Phase C: every (b) and (c) change with its diff, the tasks it can affect, and a proposed resolution (owner's decision).
- The table above with numbers: base vs lab, merge vs lab, base vs merge (identical / which records differ and why), the
  gates, closed-loop counts; the node / GPU / driver of each battery.
- Verdict: the evaluation code is OK on the merge, OK with listed protocol decisions pending, or broken (what and where).
- Leave everything in place: do not push, do not move `CLUSTER_BRANCH` to the merge, do not delete the worktree or `sanity/`.
  Pushing the merge branch or fast-forwarding `CLUSTER_BRANCH` is the owner's call.
