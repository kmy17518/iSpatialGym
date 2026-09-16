#!/usr/bin/env python3
"""First-valid-demo loop (owner decisions 2026-09-16 06:50): fix -> sample one alias (1 template x 1 instance) ->
``run_isg --mode oracle --video`` -> read meta.json -> board + ledgers.

    python scripts/isg/first_demo.py run --alias <alias> [--max-attempts 8] [--timeout 5400] [--note "..."]
    python scripts/isg/first_demo.py demo-only --tro <local tro path> [--note "..."]      # existing instance, re-run only
    python scripts/isg/first_demo.py board                                                # regenerate the board files
    python scripts/isg/first_demo.py fix --commit <sha> --subtasks A B ... --what "..."    # append to the fix log

One pool pod per run (``remote_run.py``: working tree synced, one job per pod). Everything the run produces goes to
``runlogs/pilot_20260915/first_demo/<subtask>/<run id>/`` (``inst/`` = sample_isg output, ``demo/`` = run_isg output,
``remote.log``). Every run appends a row to ``runlogs/pilot_20260915/first_demo/ledger.jsonl`` with the code commit
(``git rev-parse HEAD`` of the working tree at launch; ``dirty`` flag), the sidecar's ``sampler_version``, the sampling
outcome (n_written, rejections, wall) and the demo verdict (status, success, reason, failed action + QA category,
n_steps, wall, meta.json path).

Validity rule: a demo is **valid** when its code commit contains every fix-log entry that affects its subtask (fix
log = ``runlogs/pilot_20260915/fix_log.jsonl``); earlier demos are marked ``pre-fix``. The board
(``runlogs/pilot_20260915/first_demo_board.md``, one row per subtask: state ∈ {no instance, instance, demo run,
valid demo}, blocking item, owner, fixing commits, artefact, last change) is regenerated from the ledger + fix log
after every event; ``board`` also rewrites the mirror table between the ``<!-- first-demo-board -->`` markers in
``TODO.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.join(REPO, "runlogs", "pilot_20260915")
FD = os.path.join(ROOT, "first_demo")
LEDGER = os.path.join(FD, "ledger.jsonl")
FIXLOG = os.path.join(ROOT, "fix_log.jsonl")
BOARD = os.path.join(ROOT, "first_demo_board.md")
BOARD_STATE = os.path.join(ROOT, "first_demo_board.json")  # manual fields: blocking item, owner per subtask
REMOTE_RUN = os.path.join(REPO, "scripts", "runpod", "remote_run.py")
CUSTOMS = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json")
SUBTASKS = ["camera_relocalization-standard", "camera_relocalization-obstructed", "object_scaling", "mental_rotation", "disembedding",
            "configuration_matching-block_design", "configuration_matching-attachment", "configuration_matching-articulation_open_small_scale",
            "configuration_matching-articulation_close_small_scale", "configuration_matching-articulation_open_large_scale",
            "configuration_matching-articulation_close_large_scale", "alignment-world_vertical", "alignment-world_horizontal", "alignment-axial",
            "inter_object_relation_satisfaction-ontop_small_scale", "inter_object_relation_satisfaction-under_small_scale",
            "inter_object_relation_satisfaction-inside", "inter_object_relation_satisfaction-nextto", "inter_object_relation_satisfaction-touching",
            "inter_object_relation_satisfaction-ontop_large_scale", "inter_object_relation_satisfaction-under_large_scale"]


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M PDT")


def git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True).stdout.strip()


def subtask_of(alias, customs):
    e = customs.get(alias) or {}
    if e.get("subtask"):
        return e["subtask"]
    iddl = e.get("iddl", "")
    fam, fn = iddl.split("/", 1) if "/" in iddl else ("", iddl)
    fn = os.path.splitext(fn)[0]
    return fam if fam in ("object_scaling", "mental_rotation", "disembedding") and fn == fam else (f"{fam}-{fn}" if fam else fn)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_jsonl(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def commit_contains(commit, fix_commit):
    """True when ``fix_commit`` is an ancestor of (or equal to) ``commit``."""
    if not commit or not fix_commit:
        return False
    r = subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", fix_commit, commit], capture_output=True)
    return r.returncode == 0


def parse_meta(demo_dir):
    for m in glob.glob(os.path.join(demo_dir, "*", "meta.json")) + glob.glob(os.path.join(demo_dir, "meta.json")):
        try:
            d = json.load(open(m))
        except Exception:
            continue
        failed = None
        for a in d.get("actions", []):
            v = a.get("verdict")
            if v:
                failed = {"action": a.get("name"), "index": a.get("index"), "category": v.get("category"), "message": (v.get("message") or "")[:200]}
                break
        qa = ((d.get("qa") or {}).get("stats") or {}).get("verdicts") or []
        if failed is None and qa:
            v = qa[-1]
            failed = {"action": v.get("action"), "category": v.get("category"), "message": (v.get("message") or "")[:200]}
        return {"meta": m, "status": d.get("status"), "success": d.get("success"), "reason": d.get("reason"), "n_steps": d.get("n_steps"),
                "wall_s": d.get("wall_time_s"), "failed": failed}
    return None


def parse_sampling(inst_dir):
    out = {"n_written": 0, "rejections": {}, "error": None, "time_s": None, "sampler_version": None, "tro": None, "sidecar": None}
    sp = os.path.join(inst_dir, "summary.json")
    if os.path.exists(sp):
        try:
            d = json.load(open(sp))
            out["sampler_version"] = d.get("sampler_version")
            for prob in (d.get("problems") or {}).values():
                for t in prob.values():
                    if isinstance(t, dict):
                        out["n_written"] += int(t.get("n_written", 0))
                        out["time_s"] = (out["time_s"] or 0) + float(t.get("time_s") or 0)
                        for k, v in (t.get("rejections") or {}).items():
                            out["rejections"][k] = out["rejections"].get(k, 0) + int(v)
                        out["error"] = out["error"] or t.get("error")
            if not out["error"] and d.get("errors"):
                out["error"] = str(d["errors"][0])[:200]
        except Exception as e:  # noqa: BLE001
            out["error"] = f"summary unreadable: {e}"
    tros = sorted(glob.glob(os.path.join(inst_dir, "*", "*", "instances", "*-tro_state.json")))
    if tros:
        out["tro"] = tros[0]
        sc = tros[0].replace("-tro_state.json", ".oracle.json")
        if os.path.exists(sc):
            out["sidecar"] = sc
            try:
                out["sampler_version"] = json.load(open(sc)).get("sampler_version") or out["sampler_version"]
            except Exception:
                pass
    return out


def run_remote(cmd, out_dir, timeout):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "remote.log"), "w") as log:
        r = subprocess.run([sys.executable, REMOTE_RUN, "--timeout", str(timeout), "--pull", "$SMOKE_OUT", out_dir, "--", cmd],
                           stdout=log, stderr=subprocess.STDOUT, text=True, cwd=REPO, timeout=timeout + 3600)
    return r.returncode


def cmd_run(args):
    customs = json.load(open(CUSTOMS))
    st = subtask_of(args.alias, customs)
    run_id = time.strftime("%m%d-%H%M%S")
    out_dir = os.path.join(FD, st, run_id)
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", "ispatialgym", "scripts"))
    t0 = time.time()
    scene = f" --scene {args.scene}" if args.scene else ""
    cmd = (f"python -m ispatialgym.sampling.sample_isg --alias {args.alias}{scene} --templates 1 --instances-per-template 1 --max-attempts {args.max_attempts} "
           f"--seed {args.seed} --no-write-back --out $SMOKE_OUT/inst --log-level INFO; echo SAMPLE_EXIT $?; "
           f"tro=$(ls $SMOKE_OUT/inst/*/*/instances/*-tro_state.json 2>/dev/null | head -1); "
           f"if [ -n \"$tro\" ]; then echo TRO=$tro; python -m ispatialgym.oracle.run_isg --instance \"$tro\" --mode oracle --out $SMOKE_OUT/demo --video --video-size 540; echo DEMO_EXIT $?; fi")
    rc = run_remote(cmd, out_dir, args.timeout)
    samp = parse_sampling(os.path.join(out_dir, "inst"))
    demo = parse_meta(os.path.join(out_dir, "demo"))
    rec = {"time": now(), "kind": "sample+demo", "subtask": st, "alias": args.alias, "run_id": run_id, "out": os.path.relpath(out_dir, REPO), "rc": rc,
           "wall_s": round(time.time() - t0), "code_commit": commit, "dirty": dirty, "sampler_version": samp["sampler_version"],
           "sampling": {k: samp[k] for k in ("n_written", "rejections", "error", "time_s")}, "tro": samp["tro"] and os.path.relpath(samp["tro"], REPO),
           "demo": demo and {**demo, "meta": os.path.relpath(demo["meta"], REPO)}, "note": args.note}
    append_jsonl(LEDGER, rec)
    print(json.dumps(rec, indent=1))
    build_board()


def cmd_demo_only(args):
    customs = json.load(open(CUSTOMS))
    tro = os.path.abspath(args.tro)
    alias = os.path.basename(tro).split("__")[0]
    st = subtask_of(alias, customs)
    run_id = time.strftime("%m%d-%H%M%S")
    out_dir = os.path.join(FD, st, run_id)
    # stage the template dir under the synced inputs
    tdir = os.path.dirname(os.path.dirname(tro))
    stage = os.path.join(REPO, "runlogs", "smoke_inputs", "first_demo", alias, os.path.basename(tdir))
    if not os.path.exists(os.path.join(stage, "template.json")):
        import shutil

        shutil.copytree(tdir, stage, ignore=shutil.ignore_patterns("renders", "validation", "*.png"), dirs_exist_ok=True)
    rel_tro = os.path.relpath(os.path.join(stage, "instances", os.path.basename(tro)), REPO)
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--", "ispatialgym", "scripts"))
    t0 = time.time()
    cmd = f"python -m ispatialgym.oracle.run_isg --instance {rel_tro} --mode oracle --out $SMOKE_OUT/demo --video --video-size 540; echo DEMO_EXIT $?"
    rc = run_remote(cmd, out_dir, args.timeout)
    demo = parse_meta(os.path.join(out_dir, "demo"))
    sv = None
    sc = tro.replace("-tro_state.json", ".oracle.json")
    if os.path.exists(sc):
        try:
            sv = json.load(open(sc)).get("sampler_version")
        except Exception:
            pass
    rec = {"time": now(), "kind": "demo-only", "subtask": st, "alias": alias, "run_id": run_id, "out": os.path.relpath(out_dir, REPO), "rc": rc,
           "wall_s": round(time.time() - t0), "code_commit": commit, "dirty": dirty, "sampler_version": sv, "sampling": None,
           "tro": os.path.relpath(tro, REPO), "demo": demo and {**demo, "meta": os.path.relpath(demo["meta"], REPO)}, "note": args.note}
    append_jsonl(LEDGER, rec)
    print(json.dumps(rec, indent=1))
    build_board()


def cmd_fix(args):
    rec = {"time": now(), "commit": args.commit, "subtasks": args.subtasts if hasattr(args, "subtasts") else args.subtasks, "what": args.what,
           "owner": args.owner, "kind": args.kind}
    append_jsonl(FIXLOG, rec)
    build_board()
    print("fix logged:", rec)


def fixes_for(st, fixes):
    return [f for f in fixes if st in (f.get("subtasks") or []) or "*" in (f.get("subtasks") or [])]


def build_board():
    ledger, fixes = read_jsonl(LEDGER), read_jsonl(FIXLOG)
    manual = json.load(open(BOARD_STATE)) if os.path.exists(BOARD_STATE) else {}
    rows, states = [], {}
    for st in SUBTASKS:
        recs = [r for r in ledger if r["subtask"] == st]
        req = fixes_for(st, fixes)
        best, best_rank = None, -1
        for r in recs:
            d = r.get("demo") or {}
            valid_code = all(commit_contains(r.get("code_commit"), f["commit"]) for f in req) and not (r.get("dirty") and req)
            has_inst = bool(r.get("tro")) or (r.get("sampling") or {}).get("n_written", 0) > 0
            if d.get("success") and valid_code:
                rank, state = 4, "valid demo"
            elif d.get("success"):
                rank, state = 3, "demo run (pre-fix success)"
            elif d.get("status"):
                rank, state = 2, "demo run"
            elif has_inst:
                rank, state = 1, "instance"
            else:
                rank, state = 0, "no instance"
            r["_state"], r["_valid_code"] = state, valid_code
            if rank > best_rank or (rank == best_rank and best is not None and r["time"] > best["time"]):
                best, best_rank = r, rank
        m = manual.get(st, {})
        state = best["_state"] if best else m.get("state", "no instance")
        # a pre-existing pilot demo counts as "demo run (pre-fix)" when the manual board says so
        if best is None and m.get("pre_fix_demo"):
            state = "demo run (pre-fix success)"
        states[st] = state
        art = (best.get("demo") or {}).get("meta") if best else m.get("artefact", "")
        last = best["time"] if best else m.get("last_change", "")
        fixc = ", ".join(sorted({f["commit"][:7] for f in req})) or "-"
        verdict = ""
        if best and best.get("demo"):
            d = best["demo"]
            verdict = f"{d.get('status')}" + (f" / {d.get('reason')}" if d.get("reason") else "") + (f" @ {d['failed']['action']}:{d['failed']['category']}" if d.get("failed") else "")
        elif best and best.get("sampling"):
            s = best["sampling"]
            verdict = f"sampled n={s.get('n_written')} " + (f"err={str(s.get('error'))[:60]}" if s.get("error") else ", ".join(f"{k} {v}" for k, v in list((s.get("rejections") or {}).items())[:3]))
        rows.append((st, state, m.get("blocking", ""), m.get("owner", ""), fixc, verdict, art or "", last))
    lines = [f"# First-demo board ({now()})", "", "state ∈ {no instance, instance, demo run, valid demo}; a demo is valid only when its code commit contains every "
             "fix-log entry affecting the subtask (earlier successes are `pre-fix`).", "",
             "| subtask | state | blocking item | owner | fixing commits | last verdict | artefact (meta.json) | last change |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    n_valid = sum(1 for s in states.values() if s == "valid demo")
    lines.insert(1, f"\n**{n_valid} / {len(SUBTASKS)} subtasks with a valid demo.**")
    lines += ["", "## Fix log", "", "| time | commit | subtasks | what | owner |", "|---|---|---|---|---|"]
    for f in fixes:
        lines.append(f"| {f['time']} | `{f['commit'][:7]}` | {', '.join(f.get('subtasks') or [])} | {f['what']} | {f.get('owner','')} |")
    with open(BOARD, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # TODO.md mirror (short table) between markers
    todo = os.path.join(REPO, "TODO.md")
    if os.path.exists(todo):
        s = open(todo).read()
        short = [f"<!-- first-demo-board --> _{now()} — {n_valid}/{len(SUBTASKS)} valid; full board: `runlogs/pilot_20260915/first_demo_board.md`_", "",
                 "| subtask | state | blocking | owner |", "|---|---|---|---|"]
        for r in rows:
            short.append(f"| {r[0].replace('inter_object_relation_satisfaction-', 'IORS ').replace('configuration_matching-', 'CM ')} | {r[1]} | {r[2]} | {r[3]} |")
        short.append("<!-- /first-demo-board -->")
        block = "\n".join(short)
        if "<!-- first-demo-board -->" in s and "<!-- /first-demo-board -->" in s:
            s = re.sub(r"<!-- first-demo-board -->.*?<!-- /first-demo-board -->", lambda m: block, s, flags=re.S)
        else:
            s = s.replace("\n## Decisions register", "\n## First-demo board (mirror)\n\n" + block + "\n\n## Decisions register", 1)
        open(todo, "w").write(s)
    print(f"board: {n_valid}/{len(SUBTASKS)} valid -> {BOARD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--alias", required=True)
    s.add_argument("--max-attempts", type=int, default=8)
    s.add_argument("--scene", default=None, help="restrict sample_isg to one candidate scene")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=int, default=5400)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("demo-only")
    s.add_argument("--tro", required=True)
    s.add_argument("--timeout", type=int, default=3600)
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_demo_only)
    s = sub.add_parser("fix")
    s.add_argument("--commit", required=True)
    s.add_argument("--subtasks", nargs="+", required=True, help="subtask ids or * for all")
    s.add_argument("--what", required=True)
    s.add_argument("--owner", default="")
    s.add_argument("--kind", default="fix")
    s.set_defaults(fn=cmd_fix)
    s = sub.add_parser("board")
    s.set_defaults(fn=lambda a: build_board())
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
