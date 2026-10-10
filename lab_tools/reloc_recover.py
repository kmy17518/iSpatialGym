#!/usr/bin/env python3
"""Recover the camera-relocalization instances of the test set from the sampler's RNG stream.

A relocalization instance is an init pose (base + joints) and a goal camera pose. ``sample_isg`` draws both from an RNG
seeded by ``(seed, index, attempt)``; which attempt was *accepted* on the cluster depended on render-based visibility
checks that do not reproduce here, so the strict regeneration picked other attempts for 47 of 50 indices. The init
pose of an attempt, however, is fixed by the RNG and a collision check, and the test episode's first head frame shows
it. So, per chosen index:

1. for every attempt 0..7 the init pose is drawn exactly as ``CameraRelocalizationSampler.sample`` draws it, the robot
   is placed there and the head camera rendered; the attempt whose render matches the episode's first frame (lowest
   mean abs diff, must be <= --threshold) is the cluster's attempt;
2. for that attempt the goal candidates are enumerated as ``sample`` does (same RNG draws, collision + geodesic
   filters, the visibility filter ignored), each candidate's camera view rendered and compared with the episode's
   goal PNG (the demo's last frame, within the goal tolerance of the goal pose); the best one is the goal;
3. the instance is written (init, goal, tolerances) as a candidate template dir ``<job>/candidates/<index>/a<k>/``.

    python lab_tools/reloc_recover.py --manifest <manifest.json> --dataset datasets/isg-init-eval --out <job dir> [--seed 1003]

Runs inside the sampler's world (``sample_isg.main`` with ``run_template`` replaced), conda env ispatialgym-v2.0-vectorized.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
sys.path.insert(0, WT)
ALIAS = "camera_relocalization-standard"


def downsample(rgb: np.ndarray, size: int = 720) -> np.ndarray:
    import cv2

    rgb = np.asarray(rgb)[..., :3]
    if rgb.shape[0] == size:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def mad(a, b) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None, help="test set: isg_meta/isg-init-eval.manifest.json (the chosen instance indices)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="instead of --manifest: dataset episode indices (training set kmy17518/isg-init); their instance indices come from the CSV")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1003, help="the sweep's sample_isg --seed (test set 1003; the training sweep's seed is machine 1's / the user's to give)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="candidate sweep seeds tried in turn per index (the CSV distance match identifies the right one); default: just --seed")
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    from lab_tools.match_instances import dataset_episodes, first_frame

    # episode of each instance index (the dataset's CSV; the training CSV also lists the obstructed task - alias rows only)
    ep_of_index = {}
    bucket_of_episode = {}
    csv_rows = {}
    with open(os.path.join(args.dataset, "isg_meta", "camera_relocalization_distances.csv")) as f:
        for r in csv.DictReader(f):
            if r.get("task") and r["task"] != ALIAS:
                continue
            ep_of_index[int(r["instance"].rsplit("__", 1)[1])] = int(r["episode_index"])
            bucket_of_episode[int(r["episode_index"])] = (float(r["bucket_lo_m"]), float(r["bucket_hi_m"]))
            csv_rows[int(r["episode_index"])] = r
    if args.manifest:
        man = json.load(open(args.manifest))["tasks"][ALIAS]
        chosen = sorted(int(c["index"]) for c in man["chosen"])
    elif args.episodes is not None:
        index_of_ep = {e: i for i, e in ep_of_index.items()}
        missing = [e for e in args.episodes if e not in index_of_ep]
        if missing:
            raise SystemExit(f"episodes {missing} are not {ALIAS} episodes of {args.dataset}")
        chosen = sorted(index_of_ep[e] for e in args.episodes)
    else:
        raise SystemExit("one of --manifest / --episodes")
    if args.only:
        chosen = [i for i in chosen if i in set(args.only)]
    seeds = list(args.seeds) if args.seeds else [args.seed]
    goals = {g["episode_index"]: g for g in json.load(open(os.path.join(args.dataset, "isg_meta", "goals", "index.json")))["episodes"]}
    eps = dataset_episodes(args.dataset, chunks=[0])  # chunk 0 = camera_relocalization-standard in both datasets
    job_dir = os.path.abspath(args.out)
    cand_root = os.path.join(job_dir, "candidates")
    os.makedirs(cand_root, exist_ok=True)
    with open(os.path.join(job_dir, "recover.pid"), "w") as f:
        f.write(str(os.getpid()))
    progress_path = os.path.join(job_dir, "recover_progress.jsonl")
    done = set()
    if os.path.exists(progress_path):
        for line in open(progress_path):
            try:
                r = json.loads(line)
                if r.get("written"):
                    done.add(int(r["index"]))
            except Exception:  # noqa: BLE001
                pass
    todo = [i for i in chosen if i not in done and i in ep_of_index]
    print(f"RELOC_RECOVER chosen={len(chosen)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return 0
    strict_templates = glob.glob(os.path.join(job_dir, "inst*", ALIAS, "*", "template_meta.json"))
    strict_template_dir = os.path.dirname(strict_templates[0]) if strict_templates else None

    from PIL import Image

    import ispatialgym.sampling.sample_isg as S
    import torch as th
    from ispatialgym.activity.problem import AGENT_INSTANCE
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.tasks.base import Draft
    from ispatialgym.sampling.tro_writer import pose_target
    from ispatialgym.sampling.v2.core import ContextV2, get_sampler

    def run_template(problem, choice, world, kit, out_root, n_instances, seed, max_attempts, variant=None, customs_sha256=None, start_index=None):
        from ispatialgym.robot import conventions as C
        from ispatialgym.sampling.v2.stances import Stance
        from ispatialgym.sampling.v2.tasks import camera_relocalization as CR

        out_dir = os.path.join(out_root, problem.alias, choice.template_id)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        ctx = ContextV2(problem, choice, world, kit, out_dir, base_seed=seed, customs_sha256=customs_sha256, variant=variant)
        sampler = get_sampler(problem.subtask)(ctx)
        sampler.setup()
        if sampler.template_paths is None:
            sampler.save_template()
        # the candidates link to a template dir that must outlive this run: the strict job's template when there is one, else a
        # persistent copy of the scratch template (template.json, -partial_rooms, template_meta.json, activity_definitions)
        persist = os.path.join(job_dir, "template")
        if strict_template_dir is None and not os.path.exists(os.path.join(persist, "template_meta.json")):
            try:
                sampler.write_template_meta()
            except Exception as ex:  # noqa: BLE001
                S.log.warning("template_meta not written: %r", ex)
            os.makedirs(persist, exist_ok=True)
            for name in os.listdir(out_dir):
                if name == "instances":
                    continue
                src, dst = os.path.join(out_dir, name), os.path.join(persist, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        tdir = strict_template_dir or persist
        buckets = sampler.buckets()
        bounds = CR.bucket_bounds(buckets)
        max_h = math.radians(float(sampler.g("max_heading_deg")))
        tr_lo, tr_hi = sampler.g("goal_translate_range")
        tr_hi = min(float(tr_hi), CR.MAX_TRANSLATE)
        ti_lo, ti_hi = sampler.g("goal_tilt_range")
        margin = float(sampler.g("collision_margin_m"))
        tr0, ti0 = sampler.init_trunk()

        def draw_init(rng, consume_bucket_draw):
            """``sample_instance`` up to the init pose. ``choose_bucket`` draws one integer when more than one bucket has a
            deficit (``rng.integers(n)`` consumes one 64-bit value for n >= 2 and nothing for n == 1)."""
            if consume_bucket_draw:
                rng.integers(2)
            pts = sampler.kit.random_free_points(rng, CR.N_FREE_POINTS, room_filter=sampler.in_room)
            if len(pts) < 2:
                return None, pts
            for xy in pts:
                base = (float(xy[0]), float(xy[1]), float(rng.uniform(-math.pi, math.pi)))
                q = sampler.kit.q_at_base_pose(sampler.q_reset_rel, *base)
                if bool(sampler.kit.collision_free(q, margin=margin)[0]):
                    return (base, q), pts
            return None, pts

        def render():
            return downsample(sampler.world.rgb(3).numpy())

        n_written = 0
        for idx in todo:
            e = ep_of_index[idx]
            t1 = time.time()
            rec = {"index": idx, "episode_index": e, "written": False}
            try:
                ep_frame = first_frame(args.dataset, eps[e])
                goal_png = np.asarray(Image.open(os.path.join(args.dataset, goals[e]["png"])).convert("RGB"), dtype=np.uint8)
                lo_csv, hi_csv = bucket_of_episode[e]
                bi = min(range(len(bounds)), key=lambda i: abs(bounds[i][1] - hi_csv))
                lo, hi = bounds[bi]
                # 1. + 2. the (attempt, RNG variant, goal candidate) whose init -> goal Euclidean and geodesic distances equal the
                #    dataset's CSV values (to 2 mm; the sampler recorded them from the accepted candidate): no rendering needed,
                #    exact. Candidates are enumerated as sample_instance does (same RNG draws, the episode's bucket).
                csv_e, csv_geo = float(csv_rows[e]["euclidean_m"]), float(csv_rows[e]["geodesic_m"])
                matches = []
                variants = []
                for base_seed, attempt in ((s_, a_) for s_ in seeds for a_ in range(max_attempts)):
                    # every seed is enumerated: a short path (e.g. the 0.4 m minimum) matches by chance under a wrong seed, and the
                    # init-view / goal-view renders below decide between the distance matches
                    aseed = int(base_seed * 1000003 + idx * 1009 + attempt)
                    for consume in (False, True):
                        sampler.world.reset_scene_objects([o.name for o in sampler.objects.values()])
                        rng = np.random.default_rng(aseed)
                        init, pts = draw_init(rng, consume)
                        if init is None:
                            continue
                        base0, q0 = init
                        var = {"attempt": attempt, "consume": consume, "seed": aseed, "base_seed": base_seed, "base0": base0, "q0": q0, "pts": pts}
                        variants.append(var)
                        cam0 = sampler.kit.camera_pose(q0)
                        axis0 = sampler.world.camera_model.optical_axis(cam0)
                        cands = []
                        for xy in pts:
                            dist = math.hypot(float(xy[0]) - base0[0], float(xy[1]) - base0[1])
                            if dist < CR.MIN_EUCLIDEAN_M or dist > hi:
                                continue
                            for _ in range(CR.N_YAW_TRIES):
                                yaw1 = C.wrap_angle(base0[2] + float(rng.uniform(-max_h, max_h)))
                                tr1 = float(rng.uniform(float(tr_lo), tr_hi))
                                ti1 = float(rng.uniform(float(ti_lo), float(ti_hi)))
                                base1 = (float(xy[0]), float(xy[1]), yaw1)
                                q1 = sampler.config(base1, tr1, ti1)
                                arms = "same"
                                if sampler.self_collides(q1):
                                    q1 = sampler.config(base1, tr1, ti1, arms="untucked")
                                    arms = "untucked"
                                cam1 = sampler.kit.camera_pose(q1)
                                heading = CR.heading_between(axis0, sampler.world.camera_model.optical_axis(cam1))
                                if heading > max_h:
                                    continue
                                cands.append((dist, heading, base1, tr1, ti1, q1, cam1, arms))
                        if not cands:
                            continue
                        order = list(rng.permutation(len(cands)))
                        order.sort(key=lambda i: 0 if cands[int(i)][0] > lo else 1)
                        for rank, i in enumerate(order[:CR.N_GOAL_TRIES]):
                            dist, heading, base1, tr1, ti1, q1, cam1, arms = cands[int(i)]
                            if abs(dist - csv_e) > 2e-3:
                                continue
                            path, geo = sampler.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
                            if geo is None or path is None or len(path) < 2 or abs(float(geo) - csv_geo) > 2e-3:
                                continue
                            matches.append({"var": var, "rank": int(rank), "cand": (dist, heading, base1, tr1, ti1, q1, cam1, arms, geo, path),
                                            "flags": {"attempt": attempt, "consume": consume, "base_seed": base_seed, "collision_free": bool(sampler.kit.collision_free(q1, margin=margin)[0]),
                                                      "bucket_ok": bool(CR.bucket_index(float(geo), buckets) == bi), "geodesic_m": round(float(geo), 4), "euclidean_m": round(dist, 4)}})
                rec["n_variants"] = len(variants)
                rec["n_distance_matches"] = len(matches)
                if not matches:
                    raise Reject("no_distance_match", f"no (attempt, variant, candidate) reproduces euclidean {csv_e} / geodesic {csv_geo}")

                def lagfree_render():
                    sampler.world.render(4)  # the head-camera read lags a teleport by one to two renders
                    return downsample(sampler.world.rgb(3).numpy())

                # verification renders: the init view of every matching variant against the episode's first frame, the goal view of
                # every matching candidate against the episode's goal PNG (ties: same point, other yaw / trunk)
                seen_vars = {}
                for m in matches:
                    key = (m["var"]["attempt"], m["var"]["consume"])
                    if key not in seen_vars:
                        sampler.kit.apply(m["var"]["q0"], m["var"]["base0"])
                        img = lagfree_render()
                        seen_vars[key] = (mad(img, ep_frame), img)
                    m["init_diff"] = seen_vars[key][0]
                    m["init_img"] = seen_vars[key][1]
                    dist, heading, base1, tr1, ti1, q1, cam1, arms, geo, path = m["cand"]
                    sampler.kit.apply(q1, base1)
                    m["img"] = lagfree_render()
                    m["goal_diff"] = mad(m["img"], goal_png)
                gbest = min(matches, key=lambda m: (m["init_diff"] > args.threshold, m["goal_diff"] + m["init_diff"]))
                best = gbest["var"]
                base0 = tuple(float(v) for v in best["base0"])
                q0 = best["q0"]
                others = sorted(m["goal_diff"] for m in matches if m is not gbest)
                rec.update({"attempt": best["attempt"], "base_seed": best["base_seed"], "bucket_draw_consumed": best["consume"], "init_diff": round(gbest["init_diff"], 3),
                            "init_second_best": None, "bucket": [lo, hi], "goal_rank": gbest["rank"], "goal_diff": round(gbest["goal_diff"], 3),
                            "goal_second_best": round(others[0], 3) if others else None, "goal_flags": gbest["flags"],
                            "goal_candidates": [{"rank": m["rank"], "goal_diff": round(m["goal_diff"], 3), "init_diff": round(m["init_diff"], 3), **m["flags"]} for m in matches]})
                if gbest["init_diff"] > args.threshold:
                    raise Reject("init_mismatch", f"distance-matched variant's init view differs from the first frame by {gbest['init_diff']:.1f} > {args.threshold}")
                best["img"] = gbest["init_img"]
                dist, heading, base1, tr1, ti1, q1, cam1, arms, geo, path = gbest["cand"]
                diag = os.path.join(cand_root, str(idx), "diag")
                os.makedirs(diag, exist_ok=True)
                Image.fromarray(np.concatenate([ep_frame, best["img"]], axis=1)).save(os.path.join(diag, f"init_ep{e}_vs_a{best['attempt']}.png"))
                Image.fromarray(np.concatenate([goal_png, gbest["img"]], axis=1)).save(os.path.join(diag, f"goal_ep{e}_vs_rank{gbest['rank']}.png"))
                if geo is None or path is None or len(path) < 2:
                    raise Reject("goal_sampling", "the matching goal candidate has no path on the eroded map")
                prec = sampler.build_path_record(path, base0, base1, (tr0, ti0))
                if prec is None:
                    rec["path_record"] = "collides at a stride here (kept: the cluster accepted it)"
                    waypoints, sched = sampler.path_schedule(path, base0, base1)
                    from ispatialgym.sampling.v2 import navigation as NAV
                    from ispatialgym.sampling.v2.stances import PathRecord, path_length
                    from ispatialgym.robot import base_path as BP
                    prec = PathRecord("init", "goal", "global", waypoints, path_length(waypoints), posture=CR.RESET_ARMS, carried=None, trunk=(tr0, ti0),
                                      stride_m=NAV.STRIDE_M, n_checked=0, style=BP.FACE_TRAVEL)
                # 3. the instance, as sample_instance builds it (scene objects back to the stable scene first)
                sampler.world.reset_scene_objects([o.name for o in sampler.objects.values()])
                sampler.park_robot()
                sampler.kit.apply(q0, base0)
                d = Draft()
                d.robot_q = q0
                d.base_pose2d = base0
                d.settle_free = []
                d.extras.update({"goal_robot_q": q1, "goal_base_pose2d": base1, "goal_cam": cam1, "goal_trunk": (tr1, ti1), "goal_arms": arms,
                                 "bucket": {"index": bi, "lo_m": lo, "hi_m": hi}})
                d.summary.update({"euclidean_m": dist, "geodesic_m": float(geo), "path_length_m": prec.length_m, "heading_deg": math.degrees(heading),
                                  "goal_base_pose2d": list(base1), "goal_trunk": [tr1, ti1], "goal_arms": arms, "path_bucket_m": hi, "bucket": {"lo_m": lo, "hi_m": hi},
                                  "attempt": best["attempt"], "lenient": "reloc_recover(visibility filter replaced by the goal-image match)",
                                  "recovered": {"attempt": best["attempt"], "base_seed": best["base_seed"], "bucket_draw_consumed": best["consume"], "init_diff": rec["init_diff"],
                                                "goal_rank": gbest["rank"], "goal_diff": rec["goal_diff"], "goal_second_best": rec["goal_second_best"],
                                                "distance_matches": rec["n_distance_matches"], "method": "csv euclidean+geodesic match of the RNG candidates"}})
                sampler.sidecar.reset_instance()
                st_init = Stance("init", base0, tr0, ti0, arms=[])
                st_init.q = q0
                st_init.extras["posture"] = "JoyLo reset (arms zero)"
                st_goal = Stance("goal", base1, tr1, ti1, arms=[] if arms == "same" else "untucked")
                st_goal.q = q1
                sampler.sidecar.add_stance(st_init)
                sampler.sidecar.add_stance(st_goal)
                sampler.sidecar.set_goal(base1, tr1, ti1, arms=arms)
                sampler.sidecar.add_path(prec)
                sampler.sidecar.path_bucket_m = hi
                sampler.goal(d)
                sampler.apply_draft(d)
                if hasattr(sampler, "settle_and_validate"):
                    sampler.settle_and_validate(d)
                for check in ("check_goal_feasible", "after_goal_check"):
                    fn = getattr(sampler, check, None)
                    if fn is None:
                        continue
                    try:
                        fn(d)
                    except Reject as ex:
                        rec["goal_check"] = f"{check}: {ex.reason}: {ex}"[:200]
                sampler.kit.apply(q0, base0)
                try:
                    path_w = sampler.write_instance(d, idx, best["seed"])
                except Exception as ex:  # noqa: BLE001
                    path_w = os.path.join(out_dir, "instances", f"{sampler.instance_id(idx)}-tro_state.json")
                    rec["sidecar_error"] = repr(ex)[:200]
                    if not os.path.exists(path_w):
                        raise
                cdir = os.path.join(cand_root, str(idx), f"a{best['attempt']}")
                os.makedirs(os.path.join(cdir, "instances"), exist_ok=True)
                iid = sampler.instance_id(idx)
                for f in glob.glob(os.path.join(out_dir, "instances", iid + "*")):
                    shutil.move(f, os.path.join(cdir, "instances", os.path.basename(f)))
                for name in os.listdir(tdir):
                    if name == "instances":
                        continue
                    dst = os.path.join(cdir, name)
                    if not os.path.exists(dst):
                        os.symlink(os.path.join(tdir, name), dst)
                rec.update({"written": True, "path": os.path.join(cdir, "instances", iid + "-tro_state.json"), "wall_s": round(time.time() - t1, 1)})
                n_written += 1
                print(f"RELOC_RECOVER index={idx} ep={e} seed={best['base_seed']} attempt={best['attempt']} consume={best['consume']} init_diff={rec['init_diff']} "
                      f"goal rank={gbest['rank']} diff={rec['goal_diff']} (2nd {rec['goal_second_best']}, {rec['n_distance_matches']} distance matches) {rec['wall_s']}s", flush=True)
            except Reject as ex:
                rec.update({"reject": ex.reason, "message": str(ex)[:300], "wall_s": round(time.time() - t1, 1)})
                print(f"RELOC_RECOVER index={idx} ep={e} REJECT {ex.reason}: {str(ex)[:120]}", flush=True)
            except Exception as ex:  # noqa: BLE001
                rec.update({"error": repr(ex)[:300], "wall_s": round(time.time() - t1, 1)})
                S.log.error("recover failed", exc_info=True)
            with open(progress_path, "a") as f:
                f.write(json.dumps({k: v for k, v in rec.items() if k not in ("q0", "pts", "rng_state")}, default=str) + "\n")
        return sampler.template_summary(n_written, time.time() - t0)

    S.run_template = run_template
    S._existing_templates = lambda *a, **k: []
    root = os.path.join(job_dir, "recover_scratch")
    shutil.rmtree(root, ignore_errors=True)
    argv_s = ["--alias", ALIAS, "--templates", "1", "--instances-per-template", "1", "--start-index", str(todo[0]), "--seed", str(args.seed),
              "--max-attempts", str(args.max_attempts), "--no-write-back", "--out", root, "--log-level", "INFO"]
    print("RELOC_RECOVER sample_isg " + " ".join(argv_s), flush=True)
    try:
        S.main(argv_s)
    except SystemExit as ex:
        return 0 if int(ex.code or 0) in (0, 2) else int(ex.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
