#!/usr/bin/env python3
"""merge_vecnext: a collector-only rerun of a finished fast-flow job on the instances it sampled (job_recollect.sh).

    prep_recollect.py <source job dir> <new job dir>

Copies the source job's job.env and inst tree into the new job dir and puts every TRO back to the state the sampler wrote: the
source collector's goal override (run_isg write_tro_updates: pose_targets position / orientation, articulation_targets joint
positions) is reverted from the demo's meta.goal_override (``*_was``, the values the source run loaded), trying the JSON spellings
of integral values (``0`` / ``-0.0`` / ``0.0``) until the file hashes to meta.tro_updates.tro_sha256_before ("byte_identical"); when
none does, the restore is kept only if re-applying the source override reproduces the source collector's TRO exactly ("semantic":
nothing but the override's fields differ, and those hold what the source run loaded). The sidecar's tro_sha256 follows the restored
file. An instance restored neither way gets a skip claim (the rerun passes it over). Writes <new job dir>/recollect.json; nothing in
the source job dir is written.
"""
import glob
import hashlib
import itertools
import json
import os
import shutil
import sys

TRO = "-tro_state.json"
KEY = "ispatialgym"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def spellings(raw, override):
    """The reverted file's candidate texts: every reverted value that is integral as a float, an int or a negative zero too."""
    slots = []  # (container, key) of every reverted integral value
    meta = raw[KEY]
    for inst, rec in (override.get("pose_targets") or {}).items():
        if rec.get("applied"):
            for f in ("position", "orientation"):
                if f in rec:
                    vec = meta["pose_targets"][inst][f]
                    slots += [(vec, i) for i, v in enumerate(vec) if float(v).is_integer()]
    for inst, rec in (override.get("articulation_targets") or {}).items():
        if rec.get("applied"):
            for jn in rec["joints"]:
                j = meta["articulation_targets"][inst]["joints"][jn]
                if float(j["position"]).is_integer():
                    slots.append((j, "position"))
    slots = slots[:12]  # at most 3**12 texts
    base = [c[k] for c, k in slots]
    for combo in itertools.product(*[(float(v), int(v), -0.0 if v == 0 else float(v)) for v in base]):
        for (c, k), v in zip(slots, combo):
            c[k] = v
        yield json.dumps(raw, indent=2)


def revert(raw, override):
    meta = raw[KEY]
    for inst, rec in (override.get("pose_targets") or {}).items():
        if rec.get("applied"):
            tgt = meta["pose_targets"][inst]
            if "position" in rec:
                tgt["position"] = [float(v) for v in rec["position_was"]]
            if "orientation" in rec:
                tgt["orientation"] = [float(v) for v in rec["orientation_was"]]
    for inst, rec in (override.get("articulation_targets") or {}).items():
        if rec.get("applied"):
            for jn, j in rec["joints"].items():
                meta["articulation_targets"][inst]["joints"][jn]["position"] = float(j["was"])
    if any(r.get("applied") for r in (override.get("groups") or {}).values()):
        raise ValueError("a group override keeps no previous slot poses")


def apply_override(raw, override):
    """executor.apply_goal_override's writes (pose and articulation targets), for the semantic check."""
    meta = raw[KEY]
    for inst, rec in (override.get("pose_targets") or {}).items():
        if rec.get("applied"):
            for f in ("position", "orientation"):
                if f in rec:
                    meta["pose_targets"][inst][f] = [float(v) for v in rec[f]]
    for inst, rec in (override.get("articulation_targets") or {}).items():
        if rec.get("applied"):
            for jn, jr in rec["joints"].items():
                meta["articulation_targets"][inst]["joints"][jn]["position"] = float(jr["position"])


def main(src, dst):
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    if os.path.exists(os.path.join(dst, "inst")):
        raise SystemExit(f"{dst}/inst exists")
    os.makedirs(os.path.join(dst, "claims"), exist_ok=True)
    with open(os.path.join(src, "job.env")) as f:
        env = f.read()
    with open(os.path.join(dst, "job.env"), "w") as f:
        f.write(env.rstrip("\n") + f"\n# recollect of {src} (prep_recollect.py)\n")
    shutil.copytree(os.path.join(src, "inst"), os.path.join(dst, "inst"), symlinks=True)
    report = {"source": src, "instances": {}}
    for m in sorted(glob.glob(os.path.join(src, "demo", "*", "meta.json"))):
        j = json.load(open(m))
        iid, upd = j["instance_id"], j.get("tro_updates") or {}
        rec = {"source_status": j.get("status"), "source_reason": j.get("reason"), "changed": upd.get("changed") or []}
        if rec["changed"]:
            tro = j["tro_path"].replace(src + "/", dst + "/")
            assert tro.startswith(dst + "/") and tro.endswith(TRO), tro
            current = json.load(open(tro))  # the source collector's TRO (overridden)
            raw = json.load(open(tro))
            override = j.get("goal_override") or {}
            rec["byte_identical"] = rec["semantic"] = False
            try:
                revert(raw, override)
                text = None
                for cand in spellings(raw, override):
                    if hashlib.sha256(cand.encode()).hexdigest() == upd["tro_sha256_before"]:
                        text, rec["byte_identical"] = cand, True
                        break
                if text is None:
                    raw = json.load(open(tro))
                    revert(raw, override)
                    again = json.loads(json.dumps(raw))
                    apply_override(again, override)
                    rec["semantic"] = again == current
                    text = json.dumps(raw, indent=2)
                if rec["byte_identical"] or rec["semantic"]:
                    with open(tro, "w") as f:
                        f.write(text)
                    rec["restored_sha256"] = sha256(tro)
            except (KeyError, ValueError) as e:
                rec["error"] = repr(e)
            side = upd.get("sidecar_updated")
            if (rec["byte_identical"] or rec["semantic"]) and side:
                side = side.replace(src + "/", dst + "/")
                text = open(side).read()
                assert text.count(upd["tro_sha256_after"]) == 1, side
                with open(side, "w") as f:
                    f.write(text.replace(upd["tro_sha256_after"], rec["restored_sha256"]))
            if not (rec["byte_identical"] or rec["semantic"]):
                with open(os.path.join(dst, "claims", iid), "w") as f:
                    f.write("skip prep_recollect: the sampled TRO could not be restored byte for byte\n")
        report["instances"][iid] = rec
    report["n"] = len(report["instances"])
    report["n_restored"] = sum(1 for r in report["instances"].values() if r["changed"] and r.get("byte_identical"))
    report["n_semantic"] = sum(1 for r in report["instances"].values() if r["changed"] and not r.get("byte_identical") and r.get("semantic"))
    report["n_skipped"] = sum(1 for r in report["instances"].values() if r["changed"] and not (r.get("byte_identical") or r.get("semantic")))
    with open(os.path.join(dst, "recollect.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"PREP {dst}: {report['n']} instances, TROs restored {report['n_restored']} byte for byte + {report['n_semantic']} semantic, "
          f"{report['n_skipped']} skipped")
    return 0 if report["n_skipped"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
