#!/usr/bin/env python3
"""Batched-replay run finisher (minyeong, visionlab Slurm CPU job; source $RV/tool/creds.sh): every --every seconds

1. collect the cluster lanes' finished units into $RV/runs/<run>/out/ (+ R2 done/) and pull the Vast units (sync.py pull);
2. release the "cluster:hold" claims (the Vast queue's train units, held so Vast takes eval units first) once every eval unit of the
   Vast queue is claimed or done;
3. promote the shiyuc lane's cache deltas into the shared stores, missing files only (Kit textures: data files first, .desc last;
   Kit app-data templates per key, .complete when the key is new);
4. once every eval unit is pulled: export the eval layout (export_eval.py: ispatialgym-eval-instances + ispatialgym-eval-demos) and the
   eval LeRobot v3 dataset (merge_train.py --purpose eval), push both to private Hugging Face dataset repos, mark it done.

    finisher.py --run evalprod2 --lane minyeong=<dir> --lane shiyuc=<dir> --code <code dir> --hf-eval-lerobot <repo>
        --hf-eval-instances <repo> [--every 300]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))
CPY = "/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python"
MAIN_TEX = Path("/vision/u/minyeong/projects/isg_home/campaign_20261110/kit_textures")
MAIN_TPL = Path("/vision/u/minyeong/projects/isg_home/campaign_20261110/kit_templates")
SHIYUC_DELTA = Path("/vision/u/shiyuc/minyeong/isgrend/cache_delta")
TOOL = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        log(f"FAILED {' '.join(map(str, cmd))[:200]}: {r.stderr[-800:]}")
    return r


def s5(*args, inp=None):
    return subprocess.run(os.environ["S5"].split() + list(args), capture_output=True, text=True, input=inp)


def names(base, sub):
    return {line.split()[-1][:-5] for line in s5("ls", f"{base}/{sub}/*").stdout.splitlines() if line.strip()}


def copy_new(src: Path, dst: Path) -> int:
    if dst.exists() or dst.is_symlink():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.finisher.tmp"
    shutil.copyfile(src, tmp)
    os.chmod(tmp, 0o444)
    try:
        os.link(tmp, dst)
        n = 1
    except FileExistsError:
        n = 0
    os.unlink(tmp)
    return n


def promote_deltas():
    n_tex = n_tpl = 0
    tex = SHIYUC_DELTA / "kit_textures"
    if tex.is_dir():
        have = set(os.listdir(MAIN_TEX))
        parts = {}
        for name in os.listdir(tex):
            stem, sep, _ = name.partition(".ovtex.")
            if sep and not name.startswith("."):
                parts.setdefault(stem, []).append(name)
        for stem, ns in parts.items():
            desc = stem + ".ovtex.desc"
            if desc not in ns or desc in have:
                continue
            for name in sorted(n for n in ns if n != desc) + [desc]:
                copy_new(tex / name, MAIN_TEX / name)
            n_tex += 1
    tpl = SHIYUC_DELTA / "kit_templates"
    for keydir in (sorted(tpl.iterdir()) if tpl.is_dir() else []):
        if not (keydir / ".complete").exists():
            continue
        dst = MAIN_TPL / keydir.name
        new = not (dst / ".complete").exists()
        for root, _, files in os.walk(keydir):
            for f in files:
                if f == ".complete" or f.startswith("."):
                    continue
                src = Path(root) / f
                n_tpl += copy_new(src, dst / os.path.relpath(src, keydir))
        if new:
            (dst / ".complete").touch()
    return n_tex, n_tpl


def hf_push(src: Path, repo: str):
    env = dict(os.environ, SRC=str(src), REPO=repo)
    r = subprocess.run(["bash", str(TOOL / "upload_hf.sh")], env=env, capture_output=True, text=True)
    log(f"HF push {repo}: rc={r.returncode} {r.stdout[-600:]} {r.stderr[-600:] if r.returncode else ''}")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lane", action="append", default=[])
    ap.add_argument("--code", required=True)
    ap.add_argument("--hf-eval-lerobot", required=True)
    ap.add_argument("--hf-eval-instances", required=True)
    ap.add_argument("--every", type=int, default=300)
    a = ap.parse_args()
    base = f"{os.environ['JOB_S3']}/runs/{a.run}"
    spec = json.load(open(RV / "runs" / a.run / "units.json"))
    units = {u["unit"]: u for u in spec["units"]}
    eval_units = {u for u, x in units.items() if x["purpose"] == "eval"}
    lane_units = set()
    for lf in (RV / "cluster_lanes" / a.run).glob("*.units.txt"):
        lane_units |= set(lf.read_text().split())
    hold_file = RV / "cluster_lanes" / a.run / "vast_hold.txt"
    merged = RV / "merged" / a.run
    marker = merged / "EVAL_PUSHED"
    lane_args = sum((["--lane", x] for x in a.lane), [])
    while True:
        sh([os.environ["VPY"], str(TOOL / "cluster_assign.py"), "--run", a.run, "collect"] + lane_args)
        sh([CPY, str(TOOL / "sync.py"), "--run", a.run, "pull"])
        done = names(base, "done")
        claims = names(base, "claims")
        # 2. hold release
        if hold_file.exists():
            vast_eval = eval_units - lane_units
            if vast_eval <= (claims | done):
                held = hold_file.read_text().split()
                for uid in held:
                    cl = json.loads(s5("cat", f"{base}/claims/{uid}.json").stdout or "{}")
                    if cl.get("instance") == "cluster:hold":
                        s5("rm", f"{base}/claims/{uid}.json")
                hold_file.rename(hold_file.with_suffix(".released"))
                log(f"released {len(held)} held train units to the Vast queue")
        # 3. cache deltas
        try:
            t, p = promote_deltas()
            if t or p:
                log(f"cache deltas promoted: textures +{t}, app-data files +{p}")
        except OSError as e:
            log(f"cache promote failed: {e}")
        # 4. eval merge + push
        pulled = {u for u in eval_units if (RV / "runs" / a.run / "out" / u / ".synced").exists()}
        n_done = len(done & set(units))
        log(f"units done {n_done}/{len(units)}; eval pulled {len(pulled)}/{len(eval_units)}")
        if not marker.exists() and pulled == eval_units:
            log("all eval units in: eval export + LeRobot merge")
            shutil.rmtree(merged / "standard-eval", ignore_errors=True)
            shutil.rmtree(merged / "standard-eval-lerobot", ignore_errors=True)
            (merged / "standard-eval").mkdir(parents=True)
            env = dict(os.environ, PYTHONPATH=f"{a.code}:{a.code}/Behavior-1K/OmniGibson:{a.code}/Behavior-1K/bddl3",
                       OMNIGIBSON_DATA_PATH=f"{a.code}/Behavior-1K/datasets", HDF5_USE_FILE_LOCKING="FALSE",
                       HF_HOME=f"/tmp/isgrend_hf_{os.getpid()}", HF_HUB_OFFLINE="1")
            r1 = sh([CPY, str(TOOL / "export_eval.py"), "--runs", a.run, "--out", str(merged / "standard-eval/ispatialgym-eval-instances")], env=env)
            r2 = sh([CPY, str(TOOL / "merge_train.py"), "--runs", a.run, "--purpose", "eval", "--name", Path(a.hf_eval_lerobot).name,
                     "--out", str(merged / "standard-eval-lerobot"), "--load-test"], env=env)
            log(f"export rc={r1.returncode}: {r1.stdout[-400:]}")
            log(f"lerobot merge rc={r2.returncode}: {r2.stdout[-400:]}")
            if r1.returncode == 0 and r2.returncode == 0:
                ok1 = hf_push(merged / "standard-eval-lerobot", a.hf_eval_lerobot)
                ok2 = hf_push(merged / "standard-eval", a.hf_eval_instances)
                if ok1 and ok2:
                    marker.write_text(json.dumps({"time": time.time(), "lerobot": a.hf_eval_lerobot, "instances": a.hf_eval_instances}))
                    log("EVAL MERGED AND PUSHED")
            else:
                log("eval merge failed: not pushed; retrying next tick")
        if n_done >= len(units) and marker.exists():
            log("all units done")
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
