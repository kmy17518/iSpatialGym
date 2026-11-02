#!/usr/bin/env python3
"""Regenerate the chosen ``eval_1003`` instances of one variant on the lab machine.

The test set ``kmy17518/isg-init-eval`` was collected on the cluster (sweep ``isg_prod/eval_1003``: ``sample_isg`` with
``--seed 1003``, instance indices from 50000). Its selection manifest lists the chosen instance indices per task but the
instance files (TRO + template) stayed on the cluster, so they are regenerated here with the same sampler, seed and
indices: ``sample_isg``'s per-instance RNG is seeded from ``(seed, index, attempt)`` only, so an index can be drawn on its
own. Everything else (world, template choice = the first draw of ``default_rng(seed)``) is ``sample_isg.main`` unchanged;
only ``run_template`` is replaced by a loop over the chosen indices, and a resumed run keeps the template of the first
root (``_existing_templates`` is disabled so the same model combination is drawn again under a new ``inst_r<k>`` root).

    python lab_tools/sample_chosen.py --variant <variant> --manifest <isg-init-eval.manifest.json> --out <job dir> [--seed 1003]

Writes ``<job dir>/inst*/<variant>/<template_id>/instances/<iid>-tro_state.json`` (+ sidecar, summary) like a production job;
``<job dir>/chosen_progress.jsonl`` gets one line per index (written / failed, attempts, seconds).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def chosen_indices(manifest_path: str, variant: str):
    with open(manifest_path) as f:
        man = json.load(f)
    task = man["tasks"][variant]
    return sorted(int(c["index"]) for c in task["chosen"]), [c["instance"] for c in task["chosen"]]


_INDEX_RE = re.compile(r"__(\d{4,})-tro_state\.json$")


def written_indices(job_dir: str):
    out = {}
    for p in glob.glob(os.path.join(job_dir, "inst*", "*", "*", "instances", "*-tro_state.json")):
        m = _INDEX_RE.search(p)
        if m:
            out[int(m.group(1))] = p
    return out


def next_root(job_dir: str) -> str:
    k = 0
    while os.path.exists(os.path.join(job_dir, "inst" if k == 0 else f"inst_r{k}")):
        k += 1
    return os.path.join(job_dir, "inst" if k == 0 else f"inst_r{k}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--manifest", required=True, help="isg_meta/isg-init-eval.manifest.json of the test set")
    ap.add_argument("--out", required=True, help="job directory (inst*/ roots, chosen_progress.jsonl)")
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="restrict to these indices (debug)")
    ap.add_argument("--lenient-candidates", action="store_true",
                    help="recovery for indices the strict pass could not reproduce (the cluster accepted an attempt whose IK-based checks "
                         "fail here): every attempt 0..max-attempts-1 is drawn with the IK-based checks (stance collision, carry legs, program "
                         "check) skipped and each attempt that passes the physical checks is written as a candidate under "
                         "<job dir>/candidates/<index>/a<attempt>/ (the first frame decides which, if any, matches the test episode)")
    ap.add_argument("--fast-candidates", action="store_true",
                    help="with --lenient-candidates: per index, draw every attempt's draft (sample + apply, no settle), render it and keep only the "
                         "attempt whose first view matches one of the task's test episodes (depth mismatch); that attempt alone goes through the "
                         "physical checks and is written (candidates/<index>/a<k>/). Needs --dataset.")
    ap.add_argument("--dataset", default=os.path.join(os.path.dirname(HERE), "datasets", "isg-init-eval"))
    ap.add_argument("--lenient-all", action="store_true",
                    help="with --lenient-candidates: every chosen index, not only the ones the strict pass missed (the strict pass may have "
                         "accepted an attempt the cluster rejected on an IK check, i.e. written a different instance under the right index); "
                         "indices whose candidates/<index>/ exists are skipped")
    args = ap.parse_args(argv)

    indices, names = chosen_indices(args.manifest, args.variant)
    if args.only:
        indices = [i for i in indices if i in set(args.only)]
    job_dir = os.path.abspath(args.out)
    os.makedirs(job_dir, exist_ok=True)
    done = written_indices(job_dir)
    if args.lenient_candidates and args.lenient_all:
        have = {int(d) for d in os.listdir(os.path.join(job_dir, "candidates"))} if os.path.isdir(os.path.join(job_dir, "candidates")) else set()
        todo = [i for i in indices if i not in have]
    else:
        todo = [i for i in indices if i not in done]
    progress_path = os.path.join(job_dir, "chosen_progress.jsonl")
    print(f"SAMPLE_CHOSEN variant={args.variant} chosen={len(indices)} written={len(done)} todo={len(todo)} seed={args.seed} lenient={args.lenient_candidates}", flush=True)
    if not todo:
        return 0
    if args.lenient_candidates:
        return lenient_candidates(args, todo, job_dir)
    root = next_root(job_dir)

    import ispatialgym.sampling.sample_isg as S
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.core import ContextV2, get_sampler

    def run_template(problem, choice, world, kit, out_root, n_instances, seed, max_attempts, variant=None, customs_sha256=None, start_index=None):
        out_dir = os.path.join(out_root, problem.variant, choice.template_id)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        ctx = ContextV2(problem, choice, world, kit, out_dir, base_seed=seed, customs_sha256=customs_sha256, variant=variant)
        sampler = get_sampler(problem.subtask)(ctx)
        written = []
        try:
            sampler.setup()
            print(f"SAMPLE_CHOSEN template={choice.template_id} models={ {r: f'{c.category}/{c.model}' for r, c in choice.models.items()} }", flush=True)
            for idx in todo:
                t1 = time.time()
                rej0 = dict(sampler.rejections)
                paths = sampler.run_instances(1, max_attempts=max_attempts, start_index=idx)
                written += paths
                rec = {"index": idx, "written": bool(paths), "path": paths[0] if paths else None, "wall_s": round(time.time() - t1, 1),
                       "rejections_delta": {k: v - rej0.get(k, 0) for k, v in sampler.rejections.items() if v - rej0.get(k, 0)},
                       "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                with open(progress_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"SAMPLE_CHOSEN index={idx} written={bool(paths)} wall={rec['wall_s']}s done={len(written)}/{len(todo)}", flush=True)
            entry = sampler.template_summary(len(written), time.time() - t0)
            entry["written"] = [os.path.relpath(p, out_root) for p in written]
            return entry
        except Reject as e:
            sampler.rejection_log.add(e.reason, str(e), phase="setup")
            entry = sampler.template_summary(len(written), time.time() - t0)
            entry["error"] = f"setup rejected: {e}"
            entry["reject_reason"] = e.reason
            return entry
        except Exception as e:  # noqa: BLE001
            S.log.error("traceback", exc_info=True)
            entry = sampler.template_summary(len(written), time.time() - t0)
            entry["error"] = repr(e)
            return entry
        finally:
            try:
                sampler.write_template_meta() if sampler.template_paths is None else None
            except Exception:  # noqa: BLE001
                pass

    S.run_template = run_template
    S._existing_templates = lambda *a, **k: []  # a resumed run draws the same first template again under the new root
    argv_s = ["--variant", args.variant, "--templates", "1", "--instances-per-template", "1", "--start-index", str(todo[0]),
              "--seed", str(args.seed), "--max-attempts", str(args.max_attempts), "--no-write-back", "--out", root, "--log-level", "INFO"]
    print("SAMPLE_CHOSEN sample_isg " + " ".join(argv_s), flush=True)
    try:
        S.main(argv_s)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def lenient_candidates(args, todo, job_dir: str) -> int:
    """Per failed index and attempt: the ``run_instances`` attempt body with the IK-based checks skipped; every attempt that
    passes the physical checks (sample, apply, settle + BEHAVIOR validation, goal feasibility) is written as a candidate
    template dir ``<job dir>/candidates/<index>/a<attempt>/`` (template files symlinked from the strict run's template)."""
    import shutil

    import numpy as np

    import ispatialgym.sampling.sample_isg as S
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.core import ContextV2, get_sampler

    progress_path = os.path.join(job_dir, "candidates_progress.jsonl")
    cand_root = os.path.join(job_dir, "candidates")
    os.makedirs(cand_root, exist_ok=True)
    strict_templates = glob.glob(os.path.join(job_dir, "inst*", args.variant, "*", "template_meta.json"))
    if not strict_templates:
        print("SAMPLE_CHOSEN lenient: no strict template to copy from", flush=True)
        return 2
    strict_template_dir = os.path.dirname(strict_templates[0])

    def run_template(problem, choice, world, kit, out_root, n_instances, seed, max_attempts, variant=None, customs_sha256=None, start_index=None):
        out_dir = os.path.join(out_root, problem.variant, choice.template_id)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        ctx = ContextV2(problem, choice, world, kit, out_dir, base_seed=seed, customs_sha256=customs_sha256, variant=variant)
        sampler = get_sampler(problem.subtask)(ctx)
        sampler.setup()
        # IK-based checks off; the physical ones stay
        sampler.check_stances_with_objects = lambda draft: None
        sampler.verify_carry_legs = lambda draft: None
        sampler.program_check = lambda draft: {"mode": "skipped_lenient"}
        if sampler.template_paths is None:
            sampler.save_template()
        # --fast-candidates: the test episodes' first frames (RGB + depth) of this task, and a depth modality on the sampler's camera
        ep_frames = {}
        if args.fast_candidates:
            import cv2

            sys.path.insert(0, os.path.dirname(HERE))
            from lab_tools.match_instances import dataset_episodes, first_depth, first_frame, quantized_like_dataset

            eps_all = dataset_episodes(args.dataset)
            for e, d in eps_all.items():
                if d["task"] == args.variant:
                    ep_frames[e] = (first_frame(args.dataset, d), first_depth(args.dataset, d))
            try:
                sampler.world.cam.add_modality("depth_linear")
                sampler.world.render(2)
                have_depth = True
            except Exception as ex:  # noqa: BLE001
                print(f"SAMPLE_CHOSEN fast: no depth modality ({ex!r}); RGB only", flush=True)
                have_depth = False
            print(f"SAMPLE_CHOSEN fast: {len(ep_frames)} test episodes loaded, depth={have_depth}", flush=True)

            def view():
                sampler.world.render(4)
                obs, _ = sampler.world.robot.get_obs()
                rgb = obs[sampler.world.cam_key]["rgb"][..., :3].cpu().numpy()
                rgb = cv2.resize(rgb, (720, 720), interpolation=cv2.INTER_AREA) if rgb.shape[0] != 720 else rgb
                depth = None
                if have_depth and "depth_linear" in obs[sampler.world.cam_key]:
                    dep = obs[sampler.world.cam_key]["depth_linear"].cpu().numpy().astype(np.float32)
                    dep = dep.reshape(dep.shape[0], dep.shape[1])
                    if dep.shape[0] != 720:
                        dep = cv2.resize(dep, (720, 720), interpolation=cv2.INTER_NEAREST)
                    depth = quantized_like_dataset(np.clip(dep, 0.0, 20.0))
                return rgb, depth

            def best_episode(rgb, depth):
                best = (None, 9.0)
                for e, (fr, fd) in ep_frames.items():
                    if depth is not None and fd is not None:
                        c = float(np.mean(np.abs(np.clip(fd, 0.01, 10) - np.clip(depth, 0.01, 10)) > 0.05))
                    else:
                        c = float(np.mean(np.abs(fr.astype(np.float32) - rgb.astype(np.float32)))) / 100.0
                    if c < best[1]:
                        best = (e, c)
                return best

        n_written = 0
        for idx in todo:
            attempts_todo = list(range(max_attempts))
            if args.fast_candidates:
                scores = []
                for attempt in range(max_attempts):
                    aseed = int(sampler.ctx.base_seed * 1000003 + idx * 1009 + attempt)
                    rng = np.random.default_rng(aseed)
                    t1 = time.time()
                    try:
                        sampler.world.reset_scene_objects([o.name for o in sampler.objects.values()])
                        sampler.park_robot()
                        sampler.ikc.solutions.clear()
                        sampler.ikc.attempt = attempt
                        draft = sampler.sample(rng)
                        sampler.apply_draft(draft)
                        rgb, depth = view()
                        e, c = best_episode(rgb, depth)
                        scores.append((c, attempt, e))
                        print(f"SAMPLE_CHOSEN fast index={idx} attempt={attempt} best episode={e} cost={c:.4f} ({time.time() - t1:.1f}s)", flush=True)
                    except Reject as ex:
                        print(f"SAMPLE_CHOSEN fast index={idx} attempt={attempt} rejected in draft: {ex.reason}", flush=True)
                    except Exception as ex:  # noqa: BLE001
                        S.log.error("fast draft failed", exc_info=True)
                        print(f"SAMPLE_CHOSEN fast index={idx} attempt={attempt} error {ex!r}", flush=True)
                scores.sort()
                with open(progress_path, "a") as f:
                    f.write(json.dumps({"index": idx, "fast_scores": scores}) + "\n")
                attempts_todo = [a for c, a, e in scores[:1]] if scores else []
                if scores and scores[0][0] > 0.05:
                    print(f"SAMPLE_CHOSEN fast index={idx}: best cost {scores[0][0]:.4f} is poor; writing the two best attempts", flush=True)
                    attempts_todo = [a for c, a, e in scores[:2]]
            for attempt in attempts_todo:
                aseed = int(sampler.ctx.base_seed * 1000003 + idx * 1009 + attempt)
                rng = np.random.default_rng(aseed)
                t1 = time.time()
                phase = "reset"
                rec = {"index": idx, "attempt": attempt, "seed": aseed, "written": False}
                try:
                    sampler.world.reset_scene_objects([o.name for o in sampler.objects.values()])
                    sampler.park_robot()
                    sampler.ikc.solutions.clear()
                    sampler.ikc.attempt = attempt
                    phase = "sample"
                    draft = sampler.sample(rng)
                    phase = "apply"
                    sampler.apply_draft(draft)
                    phase = "settle_validate"
                    sampler.settle_and_validate(draft)
                    phase = "goal_check"
                    sampler.check_goal_feasible(draft)
                    sampler.after_goal_check(draft)
                    phase = "write"
                    draft.summary["wall_time_s"] = time.time() - t1
                    draft.summary["attempt"] = attempt
                    draft.summary["lenient"] = "ik_checks_skipped"
                    try:
                        path = sampler.write_instance(draft, idx, aseed)
                    except Exception as e:  # noqa: BLE001  the TRO is written first; a sidecar failure (no program) is fine here
                        path = os.path.join(out_dir, "instances", f"{sampler.instance_id(idx)}-tro_state.json")
                        rec["sidecar_error"] = repr(e)
                        if not os.path.exists(path):
                            raise
                    # move the instance files into the candidate template dir
                    cdir = os.path.join(cand_root, str(idx), f"a{attempt}")
                    os.makedirs(os.path.join(cdir, "instances"), exist_ok=True)
                    iid = sampler.instance_id(idx)
                    for f in glob.glob(os.path.join(out_dir, "instances", iid + "*")):
                        shutil.move(f, os.path.join(cdir, "instances", os.path.basename(f)))
                    for name in os.listdir(strict_template_dir):
                        if name == "instances":
                            continue
                        dst = os.path.join(cdir, name)
                        if not os.path.exists(dst):
                            os.symlink(os.path.join(strict_template_dir, name), dst)
                    rec.update({"written": True, "path": os.path.join(cdir, "instances", iid + "-tro_state.json"), "wall_s": round(time.time() - t1, 1)})
                    n_written += 1
                    print(f"SAMPLE_CHOSEN lenient index={idx} attempt={attempt} candidate written ({rec['wall_s']}s)", flush=True)
                except Reject as e:
                    rec.update({"reject": e.reason, "phase": phase, "message": str(e)[:300], "wall_s": round(time.time() - t1, 1)})
                    print(f"SAMPLE_CHOSEN lenient index={idx} attempt={attempt} rejected in {phase}: {e.reason}", flush=True)
                except Exception as e:  # noqa: BLE001
                    rec.update({"error": repr(e)[:300], "phase": phase, "wall_s": round(time.time() - t1, 1)})
                    S.log.error("lenient attempt failed", exc_info=True)
                with open(progress_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        entry = sampler.template_summary(n_written, time.time() - t0)
        return entry

    S.run_template = run_template
    S._existing_templates = lambda *a, **k: []
    root = os.path.join(job_dir, "lenient_scratch")
    shutil.rmtree(root, ignore_errors=True)
    argv_s = ["--variant", args.variant, "--templates", "1", "--instances-per-template", "1", "--start-index", str(todo[0]),
              "--seed", str(args.seed), "--max-attempts", str(args.max_attempts), "--no-write-back", "--out", root, "--log-level", "INFO"]
    print("SAMPLE_CHOSEN lenient sample_isg " + " ".join(argv_s), flush=True)
    try:
        S.main(argv_s)
    except SystemExit as e:
        return 0 if int(e.code or 0) in (0, 2) else int(e.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
