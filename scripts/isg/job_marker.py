#!/usr/bin/env python3
"""Job status markers for cross-process hand-offs (``runlogs/jobs/README.md``).

    python scripts/isg/job_marker.py <job_id> IMPLEMENTED --summary "..." [--files a.py b.py] [--extra key=value ...]
    python scripts/isg/job_marker.py <job_id> RUNNING --extra pid=1234 run_dir=runlogs/...
    python scripts/isg/job_marker.py <job_id> DONE --summary "..." --extra run_id=0924-... success=true
    python scripts/isg/job_marker.py <job_id> FAILED --summary "..." --extra error="..."
    python scripts/isg/job_marker.py <job_id> --wait DONE [--timeout 7200]     # block until the marker exists

A marker is one JSON file ``runlogs/jobs/<job_id>/<STATE>`` (states ``IMPLEMENTED`` -> ``RUNNING`` -> ``DONE`` |
``FAILED``), written atomically (``<STATE>.tmp`` + rename) and never rewritten: its presence *is* the state, its
content the details (time, git commit / dirty flag, summary, files, extras). Importable: :func:`write_marker`,
:func:`read_marker`, :func:`wait_marker`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
JOBS = os.path.join(REPO, "runlogs", "jobs")
STATES = ("IMPLEMENTED", "RUNNING", "DONE", "FAILED")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def job_dir(job_id: str) -> str:
    return os.path.join(JOBS, job_id)


def write_marker(job_id: str, state: str, summary: str = "", files: Iterable[str] = (), **extra: Any) -> str:
    """Write ``runlogs/jobs/<job_id>/<state>`` atomically; returns the path. Refuses to overwrite an existing marker."""
    if state not in STATES:
        raise ValueError(f"state {state!r} not in {STATES}")
    d = job_dir(job_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, state)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: markers are write-once (use a new job id or FAILED/DONE once)")
    rec: Dict[str, Any] = {
        "job": job_id, "state": state, "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": os.uname().nodename, "pid": os.getpid(), "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", "ispatialgym", "scripts")), "summary": summary, "files": list(files),
    }
    rec.update(extra)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_marker(job_id: str, state: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(job_dir(job_id), state)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def wait_marker(job_id: str, state: str, timeout_s: Optional[float] = None, poll_s: float = 5.0) -> Optional[Dict[str, Any]]:
    """Poll until ``<state>`` exists (or ``FAILED`` when waiting for ``DONE``); None on timeout."""
    t0 = time.time()
    while True:
        rec = read_marker(job_id, state)
        if rec is not None:
            return rec
        if state == "DONE":
            failed = read_marker(job_id, "FAILED")
            if failed is not None:
                return failed
        if timeout_s is not None and time.time() - t0 > timeout_s:
            return None
        time.sleep(poll_s)


def _parse_extra(items: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--extra expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id")
    ap.add_argument("state", nargs="?", choices=STATES)
    ap.add_argument("--summary", default="")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--wait", choices=STATES, default=None, help="block until this marker exists (DONE also returns on FAILED)")
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()
    if args.wait:
        rec = wait_marker(args.job_id, args.wait, args.timeout)
        if rec is None:
            print(f"timeout waiting for {args.job_id}/{args.wait}", file=sys.stderr)
            return 2
        print(json.dumps(rec, indent=1))
        return 0 if rec.get("state") == args.wait else 1
    if not args.state:
        ap.error("state required unless --wait is given")
    path = write_marker(args.job_id, args.state, summary=args.summary, files=args.files, **_parse_extra(args.extra))
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
