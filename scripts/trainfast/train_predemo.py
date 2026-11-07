#!/usr/bin/env python3
"""Before the demos of a train job (job_train.sh): decide which sampled instances the queue demos.

    train_predemo.py <job dir>

An instance on disk is demoed only if the sampler logged it as written (``<alias>: instance <g> written``, logged after the
eval-repeat filter and the goal-occlusion check; a sampler stopped by its time limit can leave an unchecked instance behind).
Camera relocalization only: an instance whose start (robot start base x, y, yaw) and goal (goal camera pose) are both within
0.25 m / 15 deg of ONE earlier train instance of the alias (lower train index: this job's instances kept so far, and other jobs'
instances that are not known to be dropped) is a duplicate (``instance_separation.find_repeat`` with the earlier train instances
as the comparison set). Instances not to demo get a claim file (``skip <why>``) so ``run_isg --queue`` passes them over.
Writes <job dir>/predemo.tsv.
"""
import glob
import importlib.util
import os
import re
import shlex
import sys

TRAIN = os.environ.get("ISGF_PREDEMO_ROOT") or "/vision/u/minyeong/projects/isg_home/trainfast"  # shiyuc mode: the account's own root
TRO = "-tro_state.json"
RX_WRITTEN = re.compile(r"\] (\S+): instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")


def read_env(job):
    env = {}
    for line in open(os.path.join(job, "job.env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = shlex.split(v)[0] if v else ""
    return env


def load_sep(snap):
    spec = importlib.util.spec_from_file_location("isg_instance_separation", os.path.join(snap, "ispatialgym/sampling/instance_separation.py"))
    sep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sep
    spec.loader.exec_module(sep)
    return sep


def index_of(iid):
    tail = iid.rsplit("__", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def instances(job, alias):
    """[(g, iid, tro)] with a summary, in index order."""
    out = []
    for tro in glob.glob(os.path.join(job, "inst", alias, "*", "instances", "*" + TRO)):
        iid = os.path.basename(tro)[: -len(TRO)]
        if os.path.exists(tro[: -len(TRO)] + ".summary.json"):
            out.append((index_of(iid), iid, tro))
    return sorted(out)


def dropped_in(job):
    """Instance ids a job will not keep: final rows of instances.tsv other than ok, else its skip claims."""
    path = os.path.join(job, "instances.tsv")
    if os.path.exists(path):
        rows = [l.rstrip("\n").split("\t") for l in open(path)]
        hdr = rows[0]
        return {dict(zip(hdr, r))["iid"] for r in rows[1:] if dict(zip(hdr, r)).get("status") != "ok"}, True
    out = set()
    for c in glob.glob(os.path.join(job, "claims", "*")):
        try:
            if open(c).read().startswith("skip"):
                out.add(os.path.basename(c))
        except OSError:
            pass
    return out, False


def claim(job, iid, why):
    path = os.path.join(job, "claims", iid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"skip {why}\n".encode())
    os.close(fd)
    return True


def main():
    job = os.path.abspath(sys.argv[1])
    env = read_env(job)
    alias, key, family = env["ALIAS"], env["ALIAS_KEY"], env["FAMILY"]
    confirmed = {}
    for line in open(os.path.join(job, "sample.log"), errors="replace"):
        m = RX_WRITTEN.search(line)
        if m and m.group(1) == alias:
            confirmed[int(m.group(2))] = int(m.group(3))
    mine = instances(job, alias)
    rows = []
    sep = load_sep(os.environ.get("ISGE_CODE") or env["ISGE_CODE"]) if family == "camera_relocalization" else None
    earlier = []  # (g, iid, record) the duplicate rule compares against
    if sep is not None:
        for other in sorted(glob.glob(os.path.join(TRAIN, "out", key, "*"))):
            if os.path.abspath(other) == job or not os.path.isdir(other):
                continue
            drop, _ = dropped_in(other)
            for g, iid, tro in instances(other, alias):
                if iid in drop:
                    continue
                try:
                    earlier.append((g, iid, sep.read_instance(tro)))
                except Exception as e:  # noqa: BLE001
                    print(f"PREDEMO unreadable {tro}: {e!r}")
    for g, iid, tro in mine:
        status, dup_of, gaps = "queued", "", ""
        if g not in confirmed:
            status = "unconfirmed"
        elif sep is not None:
            rec = sep.read_instance(tro)
            hit = sep.find_repeat(rec, [r for (h, _, r) in earlier if h < g])
            if hit is not None:
                e, gp = hit
                status, dup_of = "duplicate", e.instance_id or os.path.basename(e.path)[: -len(TRO)]
                gaps = ", ".join(f"{k} {d:.2f} m / {a:.0f} deg" for k, (d, a) in gp.items())
            else:
                earlier.append((g, iid, rec))
        if status != "queued":
            claim(job, iid, f"{status}" + (f" of={dup_of}" if dup_of else ""))
        rows.append((iid, g, status, dup_of, gaps))
        print(f"PREDEMO {iid} g={g} {status}" + (f" of={dup_of} ({gaps})" if dup_of else ""))
    with open(os.path.join(job, "predemo.tsv"), "w") as f:
        f.write("iid\tindex\tstatus\tdup_of\tgaps\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    n = {s: sum(1 for r in rows if r[2] == s) for s in ("queued", "unconfirmed", "duplicate")}
    print(f"PREDEMO_SUMMARY on_disk={len(rows)} confirmed={len(confirmed)} " + " ".join(f"{k}={v}" for k, v in n.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
