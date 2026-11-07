"""Grasp ladder step 3: agent proposals for the library entries still at 0 physics-valid grasps after round 2 (owner decision
2026-11-07; docs/GRASP_BANDS.md "a grasp ladder for every v2 entry"; the contract: scripts/grasp_ladder/CONTRACT.md).

    # cluster (svl11), CPU: pick every round-2 zero entry, write its inputs + strips (prepare --regen re-runs the round's own candidate
    # generation with the gate dump when the round did not keep one: CPU, ~1 min per entry -> run it as a CPU slurm job, job_prepare.sh)
    agent_step.py select  --manifest M.json [M2.json ...] --roots <glib root> [...] --batch B [--exclude ID ...]
    agent_step.py prepare --batch B [--regen]
    # lambda (codex is installed and logged in there): one non-interactive proposer per entry, answers checked against the contract
    agent_step.py propose --batch B --runner codex [--only ID ...]        # or --runner manual (prints what is missing)
    # cluster: the proposals as given poses through the same gates + physics (one glib job), then the per-step numbers
    agent_step.py submit  --batch B --code <frozen snapshot> --out <new root> [--run]
    agent_step.py report  --batch B --out <root>

Unattended: ``run_agent_step.sh`` chains the five (rsync of the batch's inputs / strips / proposals between the cluster and lambda -
JSON and PNG only, no meshes). The proposer is ``codex exec --approve-for-me`` (the Stanford workspace's managed baseline overrides
``--yolo``; ``--approve-for-me`` is its most autonomous allowed mode) with ``--output-schema contract.schema.json`` and ``-o
proposals/<id>.json``; an answer that fails :func:`ladder.check_proposals` is kept as ``proposals/<id>.invalid.json`` and asked once
more with the errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from ispatialgym.grasping import ladder as LD  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402

GLIB_ENV = ('ISG_GLIB_CAND_OVERRIDES=\'{"n_candidates":48000,"n_orientations":48,"max_representatives":30,"topk":36}\' '
            'ISG_GLIB_VAL_OVERRIDES=\'{"max_trials":30,"max_valid":30}\'')


def _entries(batch):
    return S.load_json(os.path.join(batch, "entries.json")) or []


def cmd_select(a):
    jobs = []
    for m in a.manifest:
        jobs += json.load(open(m))
    seen, uniq = set(), []
    for j in jobs:
        if j["id"] not in seen:
            seen.add(j["id"])
            uniq.append(j)
    sel = LD.select_zero(uniq, a.roots, exclude=a.exclude or ())
    os.makedirs(a.batch, exist_ok=True)
    S.save_json(os.path.join(a.batch, "entries.json"), sel)
    S.save_json(os.path.join(a.batch, "select.json"), {"manifests": a.manifest, "roots": a.roots, "exclude": a.exclude or [], "created": S.now(),
                                                       "n_jobs": len(uniq), "n_selected": len(sel)})
    for e in sel:
        print("ZERO", e["job"]["id"], e["job"].get("alias_key"), {os.path.basename(os.path.dirname(r)) or r: v for r, v in e["status"].items()})
    print(f"AGENT_SELECT {len(sel)} of {len(uniq)} entries at 0 physics-valid after the given rounds -> {a.batch}/entries.json")


def _regen(e, batch):
    """The round's own candidate generation again, with the gate dump (into the batch; the geometry is linked, not copied)."""
    from dataclasses import fields

    from ispatialgym.grasping import candidates as CA

    src = e["src_dir"]
    cand = S.load_json(os.path.join(src, "candidates.json")) or {}
    known = {f.name for f in fields(CA.CandidateConfig)}
    cfg = CA.CandidateConfig(**{**{k: v for k, v in (cand.get("config") or {}).items() if k in known}, "gate_dump": True})
    d = os.path.join(batch, "regen", e["job"]["id"])
    os.makedirs(d, exist_ok=True)
    g = os.path.join(d, "geometry.json")
    if not os.path.lexists(g):
        os.symlink(os.path.join(src, "geometry.json"), g)
    CA.generate(e["job"], d, cfg=cfg, densify=())
    return S.load_json(os.path.join(d, "gate_dump.json")), S.load_json(os.path.join(d, "candidates.json"))


def cmd_prepare(a):
    from ispatialgym import REPO_ROOT
    from ispatialgym.grasping import candidates as CA
    from ispatialgym.grasping import object_geometry as OG

    customs = S.load_json(os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")) or {}
    for name in ("CONTRACT.md", "contract.schema.json"):  # copyfile: a frozen snapshot's files are read-only, the batch's copy is not
        dst = os.path.join(a.batch, name)
        if os.path.exists(dst):
            os.chmod(dst, 0o644)
        shutil.copyfile(os.path.join(HERE, name), dst)
    for e in _entries(a.batch):
        j, src = e["job"], e["src_dir"]
        if a.only and j["id"] not in a.only:
            continue
        out = os.path.join(a.batch, "inputs", j["id"] + ".json")
        if os.path.exists(out) and not a.force:
            continue
        geom = OG.load(src)
        R = CA.rest_rotation(j.get("generation_rest_wxyz") or j.get("rest_hint_wxyz"))
        dump = S.load_json(os.path.join(src, "gate_dump.json"))
        cand = S.load_json(os.path.join(src, "candidates.json"))
        if dump is None and a.regen:
            dump, cand = _regen(e, a.batch)
        markers = LD.markers_from_dump(dump, cand)
        gs = LD.geometry_summary(geom, R)
        hist = LD.history(src, e["status"])
        strip = os.path.join(a.batch, "strips", j["id"] + ".png")
        hdr = (f"{j['id']}   alias {j.get('alias_key')}   constraint {j.get('constraint')}   extents {gs['extents_mm']} mm   mass {gs['mass_kg']} kg   "
               f"rounds {list(e['status'].values())} valid   markers {len(markers)} (near misses of the rounds' gates)")
        LD.render_strip(geom.rotated(R), markers, hdr, strip)
        doc = {"contract": LD.CONTRACT, "id": j["id"], "category": j["category"], "model": j["model"], "scale": j.get("scale"),
               "rest_hint_wxyz": j.get("rest_hint_wxyz"), "geometry": gs, "task": LD.task_role(j, customs), "gripper": LD.gripper_facts(),
               "history": hist, "markers": markers, "gate_dump": "recorded" if os.path.exists(os.path.join(src, "gate_dump.json")) else
               ("regenerated (batch regen/)" if dump is not None else "none"), "strip": os.path.relpath(strip, a.batch),
               "output": f"proposals/{j['id']}.json", "created": S.now()}
        S.save_json(out, doc)
        print("AGENT_INPUT", j["id"], "markers", len(markers), "strip", strip, flush=True)
    print("AGENT_PREPARE_DONE", flush=True)


PROMPT = ("You are a grasp proposer. Read CONTRACT.md in this directory, then inputs/{id}.json; the attached image is strips/{id}.png "
          "(open it). Propose pinch grasps for entry {id} exactly as the contract says. Your final message must be the JSON document "
          "only (it is written to proposals/{id}.json).{retry}")


def cmd_propose(a):
    os.makedirs(os.path.join(a.batch, "proposals"), exist_ok=True)
    todo = []
    for e in _entries(a.batch):
        i = e["job"]["id"]
        if a.only and i not in a.only:
            continue
        p = os.path.join(a.batch, "proposals", i + ".json")
        if os.path.exists(p) and not LD.check_proposals(S.load_json(p), i) and not a.force:
            continue
        if not os.path.exists(os.path.join(a.batch, "inputs", i + ".json")):
            print("AGENT_NO_INPUT", i, "(run prepare first)")
            continue
        todo.append(i)
    if a.runner == "manual":
        for i in todo:
            print(f"AGENT_MISSING {i}: write proposals/{i}.json per CONTRACT.md from inputs/{i}.json and strips/{i}.png")
        return
    B = os.path.abspath(a.batch)
    for i in todo:
        retry = ""
        for attempt in range(2):
            out = os.path.join(B, "proposals", i + ".json")
            cmd = [a.codex, "exec", "--approve-for-me", "--skip-git-repo-check", "--ephemeral", "-C", B, "-i", os.path.join(B, "strips", i + ".png"),
                   "--output-schema", os.path.join(B, "contract.schema.json"), "-o", out, PROMPT.format(id=i, retry=retry)]
            print("AGENT_PROPOSE", i, "attempt", attempt, flush=True)
            with open(os.path.join(B, "proposals", i + f".codex{attempt}.log"), "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
            try:
                doc = json.load(open(out))
            except Exception as ex:  # noqa: BLE001
                doc, errs = None, [f"no JSON ({ex!r}, rc {rc})"]
            else:
                errs = LD.check_proposals(doc, i)
            if not errs:
                print("AGENT_PROPOSED", i, "feasible", doc["feasible"], "n", len(doc["proposals"]), flush=True)
                break
            if os.path.exists(out):
                os.replace(out, os.path.join(B, "proposals", i + ".invalid.json"))
            retry = " Your previous answer broke the contract: " + "; ".join(errs[:6])
            print("AGENT_INVALID", i, errs[:6], flush=True)


def cmd_submit(a):
    jobs = []
    rec = []
    for e in _entries(a.batch):
        j = e["job"]
        p = os.path.join(a.batch, "proposals", j["id"] + ".json")
        doc = S.load_json(p)
        if doc is None:
            rec.append((j["id"], "no proposals"))
            continue
        errs = LD.check_proposals(doc, j["id"])
        if errs:
            rec.append((j["id"], f"invalid: {errs[:3]}"))
            continue
        if not doc["feasible"] or not doc["proposals"]:
            rec.append((j["id"], "infeasible (proposer)"))
            continue
        jobs.append(LD.agent_job(j, doc, os.path.abspath(p)))
        d = os.path.join(a.out, "glib", "models", j["id"])
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "geometry.json")):  # the pipeline's own export, reused (no re-export)
            shutil.copy2(os.path.join(e["src_dir"], "geometry.json"), os.path.join(d, "geometry.json"))
        rec.append((j["id"], f"{len(doc['proposals'])} proposals"))
    for i, r in rec:
        print("AGENT_SUBMIT", i, r)
    if not jobs:
        print("AGENT_SUBMIT nothing to build")
        return
    os.makedirs(a.out, exist_ok=True)
    m = os.path.join(a.out, "manifest.json")
    S.save_json(m, jobs)
    tag = os.path.basename(os.path.normpath(a.out))
    line = (f"GBH={a.gbh} ISG_GB_ROOT={a.code} GB_OUT={a.out} GB_GLIB={a.out}/glib GB_TAG={tag} GB_MANIFEST={m} GB_NSHARD={len(jobs)} GB_STAGES=glib "
            f"{GLIB_ENV} sbatch --parsable -A vision -p svl --exclude=svl9 --gres=gpu:titanrtx:1 --array=0-{len(jobs) - 1} -J isgof_agent "
            f"-o {a.gbh}/slurm/isgof_agent_%A_%a.out {a.code}/scripts/grasp_bands/job_glib.sh")
    print("AGENT_SBATCH", line)
    if a.run:
        jid = subprocess.check_output(line, shell=True, text=True).strip()
        print("AGENT_JOB", jid)
        with open(os.path.join(a.out, "jobid"), "w") as f:
            f.write(jid + "\n")


def cmd_report(a):
    rows = []
    for e in _entries(a.batch):
        j = e["job"]
        doc = S.load_json(os.path.join(a.batch, "proposals", j["id"] + ".json"))
        d = os.path.join(a.out, "glib", "models", j["id"])
        r = {"id": j["id"], "alias_key": j.get("alias_key"), "legacy_key": j.get("legacy_key"),
             "proposals": None if doc is None else len(doc.get("proposals") or []), "feasible": None if doc is None else doc.get("feasible")}
        if os.path.exists(os.path.join(d, "candidates.json")):
            rep = LD.entry_report(d)
            r.update(rep)
            print(LD.summarize(rep))
        else:
            print(f"{j['id']}: not built ({'no proposals' if doc is None else 'proposals ' + str(r['proposals'])})")
        rows.append(r)
    S.save_json(os.path.join(a.batch, "report.json"), {"out": a.out, "rows": rows, "created": S.now()})
    print("AGENT_REPORT", len(rows), "entries ->", os.path.join(a.batch, "report.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("select")
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--roots", nargs="+", required=True, help="glib roots of rounds 1-2 (each holds models/<id>/library.json)")
    p.add_argument("--batch", required=True)
    p.add_argument("--exclude", nargs="*")
    p.set_defaults(fn=cmd_select)
    p = sp.add_parser("prepare")
    p.add_argument("--batch", required=True)
    p.add_argument("--regen", action="store_true", help="re-run the round's generation with the gate dump when it kept none")
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_prepare)
    p = sp.add_parser("propose")
    p.add_argument("--batch", required=True)
    p.add_argument("--runner", choices=["codex", "manual"], default="manual")
    p.add_argument("--codex", default="codex")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_propose)
    p = sp.add_parser("submit")
    p.add_argument("--batch", required=True)
    p.add_argument("--code", required=True, help="the frozen snapshot the glib job runs from")
    p.add_argument("--out", required=True, help="a NEW build root")
    p.add_argument("--gbh", default="/vision/u/minyeong/projects/isg_home/oracle_fix")
    p.add_argument("--run", action="store_true")
    p.set_defaults(fn=cmd_submit)
    p = sp.add_parser("report")
    p.add_argument("--batch", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
