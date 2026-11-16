#!/usr/bin/env python3
"""Websocket overhead of a policy request: the client's round trip minus the server's own inference time (openpi replies carry
``server_timing.infer_ms``), with the eval client's exact message (3 x 224 x 224 uint8 cameras, goal, 61-D proprio).

    ws_overhead.py --url ws://127.0.0.1:8102 [--n 50]
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("isg_ws_policy", Path(__file__).resolve().parents[2] / "ispatialgym/eval/ws_policy.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8102")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    c = W.BatchedWSPolicy(a.url, slots=[0], task_ids={0: 0}, goals={0: {"zed_link": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)}},
                          client_image_size=None, action_chunk=16)
    c.connect()
    obs = {"robot_r1": {**{f"robot_r1:{cam}:Camera:0": {"rgb": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)} for cam in W.CAMERAS},
                        "proprio": np.zeros(61, np.float32)}}
    msg = c._message({0: obs})
    msg[W.ACTION_CHUNK_REQUEST_KEY] = 16
    t_pack = time.time()
    for _ in range(100):
        W.packb(msg)
    pack_ms = (time.time() - t_pack) * 10
    rt, inf = [], []
    for k in range(a.n + 5):
        t0 = time.time()
        c.ws.send(W.packb(msg))
        reply = W.unpackb(c.ws.recv(timeout=300))
        dt = (time.time() - t0) * 1000
        if k >= 5:
            rt.append(dt)
            inf.append(float((reply.get("server_timing") or {}).get("infer_ms") or np.nan))
    rt, inf = np.array(rt), np.array(inf)
    print(f"message {len(W.packb(msg)) / 1e6:.2f} MB, pack {pack_ms:.2f} ms; round trip {rt.mean():.1f} ms, server inference {np.nanmean(inf):.1f} ms, "
          f"websocket + (un)pack overhead {np.nanmean(rt - inf):.2f} ms per request ({100 * np.nanmean(rt - inf) / rt.mean():.1f}%)")


if __name__ == "__main__":
    main()
