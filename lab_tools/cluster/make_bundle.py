"""Pack the lab machine's evaluation inputs that are neither in git nor on the Hub into one tarball for another machine (the
cluster worktree of lab_tools/cluster/SETUP_PROMPT.md).

Contents (paths inside the archive are worktree-relative: extract it at the worktree root):

* the instance files ``lab/instances.json`` / ``lab/instances-train.json`` point at: every TRO's template directory
  (template.json, template_meta.json, template-partial_rooms.json, activity_definitions/, instances/) from
  ``datasets/eval_1003_regen`` (regenerated here), ``datasets/eval_1003_cluster`` (copied from the cluster's eval_1003),
  ``datasets/train_regen`` (regenerated here) and ``datasets/isg-reloc-train-instances`` (the Hub bundle, whole, without its
  download cache). Symlinks are dereferenced: they are absolute paths of this worktree. Renders, demos and logs inside
  those directories are left out. The absolute ``scene_file`` paths inside the templates are provenance only (the loader
  passes the template's own template.json as the scene file);
* the wrist goal PNGs ``datasets/goals/`` (lab_tools/wrist_goals.py output);
* the lab's records: ``lab/instances*.json``, ``lab/env.json``, ``lab/notes.md`` and ``lab/diagnostics/`` (the reference
  replays of lab_tools/cluster/expected_replays.json and every gate record).

Writes ``<out>/isg-lab-eval-bundle-<date>.tar.gz``, its ``.sha256`` (``sha256sum -c`` format) and ``.manifest.json`` (every
file with size and sha256; also inside the archive as ``datasets/lab-eval-bundle.manifest.json``).

    python lab_tools/cluster/make_bundle.py --out /tmp/v20eval-transfer
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_EXT = (".png", ".jpg", ".mp4", ".hdf5", ".log", ".pid", ".pyc", ".incomplete", ".lock")
LAB_FILES = ("lab/instances.json", "lab/instances-train.json", "lab/instances_report.json", "lab/instances-train_report.json",
             "lab/env.json", "lab/notes.md")
WHOLE = ("datasets/isg-reloc-train-instances", "datasets/goals", "lab/diagnostics")


def walk(rel_dir: str, skip_ext=SKIP_EXT):
    """Worktree-relative paths of the files below ``rel_dir``, through symlinked directories (by their link path)."""
    out = []
    for root, dirs, files in os.walk(os.path.join(WT, rel_dir), followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in (".cache", "__pycache__", ".git"))
        for f in sorted(files):
            if not f.endswith(skip_ext):
                out.append(os.path.relpath(os.path.join(root, f), WT))
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=f"isg-lab-eval-bundle-{time.strftime('%Y%m%d')}")
    args = ap.parse_args(argv)

    files, tros = set(), []
    for name in ("lab/instances.json", "lab/instances-train.json"):
        for e in json.load(open(os.path.join(WT, name)))["episodes"].values():
            if e.get("tro"):
                tros.append(e["tro"])
                files.update(walk(os.path.dirname(os.path.dirname(e["tro"]))))
    for d in WHOLE:
        files.update(walk(d, skip_ext=(".log", ".pid", ".pyc", ".incomplete", ".lock")))
    files.update(p for p in LAB_FILES if os.path.isfile(os.path.join(WT, p)))
    missing = [t for t in tros if t not in files]
    if missing:
        print(f"{len(missing)} referenced TROs would be missing from the bundle, e.g. {missing[:3]}", file=sys.stderr)
        return 1

    commit = subprocess.run(["git", "-C", WT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    entries = []
    for p in sorted(files):
        full = os.path.join(WT, p)
        entries.append({"path": p, "size": os.path.getsize(full), "sha256": sha256(full)})
    manifest = {"format": "isg-lab-eval-bundle/v1", "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename,
                "worktree_commit": commit, "n_files": len(entries), "bytes": sum(e["size"] for e in entries), "n_tros": len(set(tros)),
                "extract": "tar -xzf <bundle> -C <worktree root>", "files": entries}

    os.makedirs(args.out, exist_ok=True)
    tgz = os.path.join(args.out, args.name + ".tar.gz")
    blob = json.dumps(manifest, indent=1).encode()
    with tarfile.open(tgz, "w:gz", compresslevel=6, dereference=True) as tar:
        for e in entries:
            tar.add(os.path.join(WT, e["path"]), arcname=e["path"], recursive=False)
        info = tarfile.TarInfo("datasets/lab-eval-bundle.manifest.json")
        info.size, info.mtime, info.mode = len(blob), int(time.time()), 0o644
        tar.addfile(info, io.BytesIO(blob))
    digest = sha256(tgz)
    with open(tgz + ".sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(tgz)}\n")
    with open(os.path.join(args.out, args.name + ".manifest.json"), "wb") as f:
        f.write(blob)
    print(f"{tgz}: {len(entries)} files, {manifest['bytes'] / 1e6:.1f} MB raw -> {os.path.getsize(tgz) / 1e6:.1f} MB, {len(set(tros))} TROs")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
