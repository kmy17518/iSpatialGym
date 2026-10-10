"""Machine 1's 2026-10-08T12:12Z dataset update (lab side): two files of kmy17518/isg-init at a pinned revision (never the whole
repo), the whole kmy17518/isg-init-eval at its pinned revision, every file verified against the Hub (LFS files by sha256, the
others by their git blob id) and the announced sha256 prefixes checked. Writes lab/diagnostics/dataset_update_20261008.json.

    act-isg/.venv/bin/python lab_tools/pull_datasets.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)

from lab_tools import lab_sync as LS  # noqa: E402

INIT_REPO, INIT_REV = "kmy17518/isg-init", "3d1e0fd661bc9f03c80057404f4cfbc642fc9092"
EVAL_REPO, EVAL_REV = "kmy17518/isg-init-eval", "c379f77dafa0054bcadc9e90d65260e00b06895b"
EXPECT = {("isg-init", "isg_meta/train_split.json"): "b6f222590683", ("isg-init", "README.md"): "c47132a066af",
          ("isg-init-eval", "isg_meta/eval_split.json"): "09784e470637", ("isg-init-eval", "isg_meta/settle_windows.json"): "cbb5587ae6d4",
          ("isg-init-eval", "isg_meta/goals/index.json"): "4d18bc7e2581"}
EXPECT_EVAL_FILES = 358


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    LS.load_env()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    out = {"at": LS.utc(), "isg-init": {"repo": INIT_REPO, "revision": INIT_REV, "files": {}}, "isg-init-eval": {"repo": EVAL_REPO, "revision": EVAL_REV}}
    # -- isg-init: two files only ---------------------------------------------------------------------------------------
    init_dir = os.path.join(WT, "datasets", "isg-init")
    for rel in ("isg_meta/train_split.json", "README.md"):
        p = hf_hub_download(INIT_REPO, rel, repo_type="dataset", revision=INIT_REV, local_dir=init_dir)
        h = sha256_of(p)
        exp = EXPECT[("isg-init", rel)]
        out["isg-init"]["files"][rel] = {"sha256": h, "expected_prefix": exp, "ok": h.startswith(exp), "path": os.path.relpath(p, WT)}
        print(f"isg-init {rel}: sha256 {h} expected {exp}… {'OK' if h.startswith(exp) else 'MISMATCH'}", flush=True)
    try:
        ts = json.load(open(os.path.join(init_dir, "isg_meta", "train_split.json")))
        n_train = sum(len(v.get("train", [])) for v in ts.get("tasks", ts).values() if isinstance(v, dict))
        excl = {k: v.get("excluded") for k, v in ts.get("tasks", ts).items() if isinstance(v, dict) and v.get("excluded")}
        out["isg-init"]["train_split"] = {"n_train_episodes": n_train, "excluded": excl}
        print(f"isg-init train_split.json: {n_train} train episodes; excluded: {excl}", flush=True)
    except Exception as e:  # noqa: BLE001
        out["isg-init"]["train_split"] = {"error": repr(e)}
    # -- isg-init-eval: whole snapshot at the revision ----------------------------------------------------------------
    eval_dir = os.path.join(WT, "datasets", "isg-init-eval")
    t0 = time.time()
    snapshot_download(EVAL_REPO, repo_type="dataset", revision=EVAL_REV, local_dir=eval_dir, max_workers=8)
    out["isg-init-eval"]["download_s"] = round(time.time() - t0, 1)
    tree = list(api.list_repo_tree(EVAL_REPO, repo_type="dataset", revision=EVAL_REV, recursive=True, expand=True))
    files = [t for t in tree if getattr(t, "rfilename", None) is not None and type(t).__name__ == "RepoFile"]
    verified, bad, missing = 0, [], []
    for t in files:
        p = os.path.join(eval_dir, t.rfilename)
        if not os.path.exists(p):
            missing.append(t.rfilename)
            continue
        if getattr(t, "lfs", None):
            ok = sha256_of(p) == t.lfs.sha256
        else:
            ok = git_blob_sha1(p) == t.blob_id
        if ok:
            verified += 1
        else:
            bad.append(t.rfilename)
    out["isg-init-eval"].update({"hub_file_count": len(files), "expected_file_count": EXPECT_EVAL_FILES, "verified": verified, "bad": bad, "missing": missing,
                                 "n_lfs": sum(1 for t in files if getattr(t, "lfs", None)), "ok": (len(files) == EXPECT_EVAL_FILES and verified == len(files) and not bad and not missing)})
    out["isg-init-eval"]["files"] = {}
    for rel in ("isg_meta/eval_split.json", "isg_meta/settle_windows.json", "isg_meta/goals/index.json"):
        p = os.path.join(eval_dir, rel)
        h = sha256_of(p) if os.path.exists(p) else None
        exp = EXPECT[("isg-init-eval", rel)]
        out["isg-init-eval"]["files"][rel] = {"sha256": h, "expected_prefix": exp, "ok": bool(h and h.startswith(exp))}
        print(f"isg-init-eval {rel}: sha256 {h} expected {exp}… {'OK' if h and h.startswith(exp) else 'MISMATCH'}", flush=True)
    try:
        es = json.load(open(os.path.join(eval_dir, "isg_meta", "eval_split.json")))
        tasks = es.get("tasks", es)
        out["isg-init-eval"]["eval_split"] = {k: {"n": len(v.get("episodes", v.get("eval50", [])) if isinstance(v, dict) else v), "train_task_index": (v.get("train_task_index") if isinstance(v, dict) else None)} for k, v in tasks.items()} if isinstance(tasks, dict) else {"raw_type": type(tasks).__name__}
        gi = json.load(open(os.path.join(eval_dir, "isg_meta", "goals", "index.json")))
        out["isg-init-eval"]["goals_index_entries"] = len(gi.get("episodes", []))
        sw = json.load(open(os.path.join(eval_dir, "isg_meta", "settle_windows.json")))
        out["isg-init-eval"]["settle_windows_entries"] = len(sw.get("episodes", sw)) if isinstance(sw, dict) else len(sw)
    except Exception as e:  # noqa: BLE001
        out["isg-init-eval"]["eval_split"] = {"error": repr(e)}
    print(f"isg-init-eval: {len(files)} files on the Hub (expected {EXPECT_EVAL_FILES}), {verified} verified, bad={bad}, missing={missing}", flush=True)
    os.makedirs(os.path.join(WT, "lab", "diagnostics"), exist_ok=True)
    with open(os.path.join(WT, "lab", "diagnostics", "dataset_update_20261008.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if (all(v["ok"] for v in out["isg-init"]["files"].values()) and out["isg-init-eval"]["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
