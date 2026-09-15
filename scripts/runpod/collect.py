#!/usr/bin/env python3
"""Background collector: pull results from our pods as soon as their job finishes, then terminate them.

    nohup python scripts/runpod/collect.py --purpose bench --interval 120 > runlogs/runpod/collect.log 2>&1 &

For every pod in the fleet registry with the given purpose and no ``terminated_at``: when ``/root/JOB_DONE`` exists,
rsync ``/root/bench`` (or ``--remote-dir``) into ``runlogs/runpod/<purpose>/<pod name>/``, mark ``collected_at`` in
the registry and terminate the pod (``--keep`` to leave it running). Exits when no such pod is left. Runs detached so
a paused interactive session cannot lose results to the TTL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", default="bench")
    ap.add_argument("--remote-dir", default="/root/bench")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--keep", action="store_true", help="do not terminate pods after collecting")
    args = ap.parse_args()
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    while True:
        fleet = rp.load_fleet()
        live = api.myself()["pods"]
        live_ids = {p["id"] for p in live}
        pending = [(pid, info) for pid, info in fleet["pods"].items()
                   if info.get("purpose") == args.purpose and not info.get("terminated_at") and not info.get("collected_at")]
        if not pending:
            print(f"{rp._now()} collector: nothing pending; exiting", flush=True)
            return
        for pid, info in pending:
            if pid not in live_ids:
                info["terminated_at"] = info.get("terminated_at") or rp._now()
                rp.save_fleet(fleet)
                print(f"{rp._now()} {info['name']}: pod gone before collection", flush=True)
                continue
            ep = rp.pod_ssh(api, pid)
            if ep is None:
                continue
            try:
                r = rp.ssh_run(*ep, "test -f /root/JOB_DONE && echo DONE; test -f /root/BOOTSTRAP_FAILED && echo FAILED; tail -1 /root/bench/jobs.txt 2>/dev/null",
                               capture=True, timeout=45)
            except SystemExit:
                continue
            out = (r.stdout or "").strip()
            if "DONE" not in out and "FAILED" not in out:
                print(f"{rp._now()} {info['name']}: running ({out.splitlines()[-1] if out else '-'})", flush=True)
                continue
            dst = os.path.join(rp.STATE_DIR, args.purpose, info["name"])
            os.makedirs(dst, exist_ok=True)
            try:
                rp.rsync(args.remote_dir + "/", dst + "/", *ep, to_pod=False)
                for f in ("/root/isg_bootstrap.log", "/root/isg_bench.log"):
                    try:
                        rp.rsync(f, dst + "/", *ep, to_pod=False)
                    except subprocess.CalledProcessError:
                        pass
                with open(os.path.join(dst, "pod.json"), "w") as fh:
                    json.dump({"pod": pid, **info}, fh, indent=2)
                info["collected_at"] = rp._now()
                rp.save_fleet(fleet)
                print(f"{rp._now()} {info['name']}: collected -> {dst} ({'FAILED' if 'FAILED' in out else 'ok'})", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"{rp._now()} {info['name']}: rsync failed: {e}", flush=True)
                continue
            if not args.keep:
                try:
                    api.terminate_pod(pid)
                    info["terminated_at"] = rp._now()
                    rp.save_fleet(fleet)
                    rp.log_billing(f"TERMINATE {pid} {info['name']} reason=collected")
                    print(f"{rp._now()} {info['name']}: terminated", flush=True)
                except RuntimeError as e:
                    print(f"{rp._now()} {info['name']}: terminate failed: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
