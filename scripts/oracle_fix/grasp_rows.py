"""oracle-fix 2026-11-06: a legacy grasp-library key made by ``scripts/oracle_fix/job_glib_scaled_key.sh`` into the repo (no simulator) - the
steps the spatula__hkvsep__t076 / lid__pbmsji__s065 runs did by hand (isg_home/eval85/scratch_spatula076: make_grasp_rows.py, insert_grasp_rows.py).

    # 1. the reviewed run's files into runlogs/grasp_library_20260905/{cands,results,review}/ (the review: <work>/glib/strips/<key>.png
    #    judged with runlogs/grasp_library_20260905/review_prompt.md, written as review/wave7_spatulat076.json is: one entry per AG-valid row)
    python scripts/oracle_fix/grasp_rows.py install <work>/glib <key> <review.json>
    # 2. require/grasps.csv rows by build_icl.Builder.build_grasps (the code that writes the sheet), inserted after <after_key>'s rows (CRLF)
    python scripts/oracle_fix/grasp_rows.py rows <key> <after_key>

Run from the repo root in the ``ispatialgym`` conda env (build_icl needs the knowledge base and the grasp library)."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB = os.path.join(REPO, "runlogs", "grasp_library_20260905")
GRASPS_CSV = os.path.join(REPO, "ispatialgym", "assets", "icl", "require", "grasps.csv")


def install(glib: str, key: str, review: str) -> None:
    res = json.load(open(os.path.join(glib, "results", f"{key}.json")))
    rev = json.load(open(review))
    entry = next((m for m in rev.get("models", []) if m.get("key") == key), None)
    if entry is None:
        sys.exit(f"{review} has no review entry for {key}")
    n_valid = sum(1 for t in res.get("tested", []) if t.get("ag_attached") and t.get("held_after_gravity"))
    if len(entry.get("grasps", [])) != n_valid:
        sys.exit(f"{review}: {len(entry.get('grasps', []))} reviewed rows for {n_valid} AG-valid grasps of {key}")
    rest = (res.get("candidates_meta") or {}).get("rest_pose")
    for src, sub in ((os.path.join(glib, "cands", f"{key}.yaml"), "cands"), (os.path.join(glib, "results", f"{key}.json"), "results"), (review, "review")):
        dst = os.path.join(LIB, sub, os.path.basename(src))
        if os.path.exists(dst):
            sys.exit(f"{dst} exists (a key is installed once)")
        shutil.copy2(src, dst)
        print(f"{src} -> {os.path.relpath(dst, REPO)}")
    print(f"{key}: {n_valid} AG-valid of {len(res.get('tested', []))} tested, rest {rest}, scale {res.get('scale')}")


def _builder():
    spec = importlib.util.spec_from_file_location("build_icl", os.path.join(REPO, "scripts", "icl", "build_icl.py"))
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    return B


def rows(key: str, after_key: str) -> None:
    B = _builder()
    b = B.Builder.__new__(B.Builder)  # only build_grasps is needed: the results index + the key
    b.results = B.grasp_results_index()
    b.keys_used = {key}
    if key not in b.results:
        sys.exit(f"no results/{key}.json in {B.LIB} (run install first)")
    new = [{c: r.get(c, "") for c in B.GRASP_COLUMNS} for r in b.build_grasps()]
    if not new or any(r["verdict"] == "none" for r in new):
        sys.exit(f"{key}: no loadable grasps ({[r.get('flags') for r in new]})")
    data = open(GRASPS_CSV, "rb").read()
    lines = data.split(b"\r\n")
    assert lines[-1] == b"", "grasps.csv must end with CRLF"
    lines = lines[:-1]
    assert all(b"\n" not in ln and b"\r" not in ln for ln in lines), "grasps.csv must use CRLF only"
    assert lines[0].decode() == ",".join(B.GRASP_COLUMNS), "grasps.csv header differs from build_icl.GRASP_COLUMNS"
    existing = [ln.split(b",", 1)[0].decode() for ln in lines[1:]]
    if key in existing:
        sys.exit(f"{key} already in {GRASPS_CSV}")
    last = max(i for i, k in enumerate(existing, start=1) if k == after_key)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=B.GRASP_COLUMNS, lineterminator="\r\n")
    w.writerows(new)
    ins = buf.getvalue().encode().split(b"\r\n")[:-1]
    open(GRASPS_CSV, "wb").write(b"\r\n".join(lines[:last + 1] + ins + lines[last + 1:]) + b"\r\n")
    print(f"{len(ins)} rows of {key} inserted after line {last + 1} ({after_key}) of {os.path.relpath(GRASPS_CSV, REPO)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("install")
    a.add_argument("glib")
    a.add_argument("key")
    a.add_argument("review")
    r = sub.add_parser("rows")
    r.add_argument("key")
    r.add_argument("after_key")
    args = ap.parse_args(argv)
    if args.cmd == "install":
        install(args.glib, args.key, args.review)
    else:
        rows(args.key, args.after_key)


if __name__ == "__main__":
    main()
