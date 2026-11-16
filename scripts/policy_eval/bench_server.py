#!/usr/bin/env python3
"""Latency / throughput of a policy server with the eval client's wire format (ispatialgym.eval.ws_policy.BatchedWSPolicy):
C concurrent clients (threads; each its own websocket session, as C run_isg processes on one GPU), batch B per request, N
control steps each. Observations are 224 x 224 uint8 frames (random) + a 61-D proprio near the robot's start state.

    bench_server.py --url ws://127.0.0.1:8000 --clients 1,2,4 --steps 200 [--batch 1] [--task-id 0] [--task-name A]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

import importlib.util  # noqa: E402

# the client module by path: the ispatialgym package's __init__ imports the simulator stack, which a policy venv lacks
_spec = importlib.util.spec_from_file_location("isg_ws_policy", Path(__file__).resolve().parents[2] / "ispatialgym/eval/ws_policy.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BatchedWSPolicy = _mod.BatchedWSPolicy


class Client(BatchedWSPolicy):
    task_names = None

    def _message(self, obs_by_slot):
        msg = super()._message(obs_by_slot)
        if self.task_names:
            msg["__task_name__"] = [self.task_names[s] for s in self.slots]
        return msg


def obs(rng):
    cams = {f"robot_r1:{c}:Camera:0": {"rgb": rng.integers(0, 255, (224, 224, 4), dtype=np.uint8)}
            for c in ("zed_link", "left_realsense_link", "right_realsense_link")}
    p = np.zeros(61, np.float32)
    p[53:57] = [1.025, -1.45, -0.47, 0.0]
    p[24:26] = p[49:51] = 0.05
    return {"robot_r1": {**cams, "proprio": p}}


def run(url, B, steps, task_id, task_name, out, k):
    rng = np.random.default_rng(k)
    goal = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    c = Client(url, slots=list(range(B)), task_ids={s: task_id for s in range(B)}, goals={s: {"zed_link": goal} for s in range(B)}, client_image_size=None)
    if task_name:
        c.task_names = {s: task_name for s in range(B)}
    c.connect()
    o = {s: obs(rng) for s in range(B)}
    t0 = time.time()
    for _ in range(steps):
        c.act(o)
    out[k] = {"wall_s": time.time() - t0, **c.stats()}
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000")
    ap.add_argument("--clients", default="1")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--task-name", default=None)
    a = ap.parse_args()
    run(a.url, a.batch, a.warmup, a.task_id, a.task_name, {}, 0)  # compile / warm up
    for C in [int(x) for x in a.clients.split(",")]:
        out = {}
        th = [threading.Thread(target=run, args=(a.url, a.batch, a.steps, a.task_id, a.task_name, out, k)) for k in range(C)]
        t0 = time.time()
        [t.start() for t in th]
        [t.join() for t in th]
        wall = time.time() - t0
        steps = C * a.batch * a.steps
        req = np.mean([v["request_s_mean"] for v in out.values()])
        print(json.dumps({"clients": C, "batch": a.batch, "env_steps_per_s": round(steps / wall, 1), "per_client_steps_per_s": round(a.steps * a.batch / wall, 1),
                          "request_ms_mean": round(1000 * req, 1), "requests_per_client": out[0]["n_requests"], "protocol": out[0]["protocol"]}), flush=True)


if __name__ == "__main__":
    main()
