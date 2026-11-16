"""A websocket policy server as a ``run_isg --mode policy`` policy (``--policy ispatialgym.eval.ws_run_policy:WSPolicy``).

The single-env form of the vectorized evaluator's client (:class:`~ispatialgym.eval.ws_policy.BatchedWSPolicy` with one slot,
so the wire format is the same): every control step the robot's 61-D proprioception, the three cameras' uint8 frames at
their native size (224 x 224 with ``--image-size 224``, what the training data holds, so no resize happens anywhere), the
instance's goal image (the eval dataset's goal PNG, ``goal::<head camera key>``) and the task's training ``task_id``.
Servers: ACT ``b1k_server.py`` / DP ``serve_b1k.py`` (one action per request; they run their own chunking) and openpi
``serve_b1k.py`` (an ``action_chunk`` per request, executed open loop: ``action_chunk`` = its ``--action-horizon``).

Task identity: the instance's ``source.class_id`` (= the training dataset's ``task_name``) -> ``task_index`` of the training
dataset's ``meta/tasks.jsonl`` (``tasks``: that file). ``send_task_name``: also send the alias as ``__task_name__`` (openpi
servers patched by scripts/policy_eval/patches/openpi_task_prompts.py then prompt with that task's exact training prompt;
the text-only checkpoints need it).

Recorded in ``meta.json`` ``policy_client``: server metadata (protocol), alias / task id, goal sha, requests and their latency.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from ispatialgym.eval.ws_policy import BatchedWSPolicy, load_rgb


class _Client(BatchedWSPolicy):
    task_names: Optional[Dict[int, str]] = None
    send_goal: bool = True

    def _build_goal_msg(self) -> None:
        if self.send_goal:
            return super()._build_goal_msg()
        # text-only checkpoints: no goal image on the wire (their server reads only the goal views it trained with: none)
        self._goal_msg, self.goal_views = {}, []

    def _message(self, obs_by_slot):
        msg = super()._message(obs_by_slot)
        if self.task_names:
            msg["__task_name__"] = [self.task_names[s] for s in self.slots]
        return msg


def _tasks(path: str) -> Dict[str, int]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                out[t["task_name"]] = int(t["task_index"])
    return out


class WSPolicy:
    def __init__(self, url: str = "ws://127.0.0.1:8000", tasks: Optional[str] = None, protocol: str = "auto",
                 action_chunk: Optional[int] = None, send_task_name: bool = False, goal: Optional[str] = None, send_goal: bool = True,
                 timeout: float = 300.0, open_timeout: float = 600.0):
        self.url = os.environ.get("ISG_POLICY_URL", url)
        self.tasks = _tasks(tasks or os.environ["ISG_TRAIN_TASKS"])
        self.protocol = protocol
        self.action_chunk = action_chunk
        self.send_task_name = bool(send_task_name)
        self.send_goal = bool(send_goal)
        self.goal = goal
        self.timeout = float(timeout)
        self.open_timeout = float(open_timeout)
        self.client: Optional[_Client] = None
        self.meta: Optional[dict] = None
        self.t_connect = None

    def bind(self, executor=None, loader=None, meta=None, **_):
        from ispatialgym.oracle import run_isg as R

        rec = R._dataset_record(loader)
        alias = (rec.source or {}).get("class_id") if rec is not None else None
        if alias is None or alias not in self.tasks:
            raise RuntimeError(f"WSPolicy: no training task for instance {loader.tro_path} (class_id {alias!r})")
        # run_isg --cross-goal: the instance's cross-episode goal image (eval dataset files.cross_episode_goal_rgb: another episode of its
        # goal group) instead of its own; an instance without one (a singleton goal group) keeps its own
        cross = bool(((meta or {}).get("config") or {}).get("cross_goal"))
        cross_path = rec.path("cross_episode_goal_rgb") if rec.files.get("cross_episode_goal_rgb") else None
        if self.goal:
            goal_path, goal_kind = self.goal, "given"
        elif cross and cross_path and os.path.exists(cross_path):
            goal_path, goal_kind = cross_path, "cross_episode"
        elif cross:
            goal_path, goal_kind = loader.goal_image_path, "own (no cross-episode goal)"
        else:
            goal_path, goal_kind = loader.goal_image_path, "own"
        goal = load_rgb(goal_path)
        t0 = time.time()
        self.client = _Client(self.url, slots=[0], task_ids={0: self.tasks[alias]}, goals={0: {"zed_link": goal}}, protocol=self.protocol,
                              action_chunk=self.action_chunk, timeout=self.timeout, open_timeout=self.open_timeout, client_image_size=None)
        if self.send_task_name:
            self.client.task_names = {0: alias}
        self.client.send_goal = self.send_goal
        if not self.send_goal:
            goal_kind = "none (send_goal false)"
        server = self.client.connect()
        self.t_connect = time.time() - t0
        self.meta = meta
        if meta is not None:
            meta["policy_client"] = {"url": self.url, "alias": alias, "task_id": self.tasks[alias], "send_task_name": self.send_task_name,
                                     "goal_path": goal_path, "goal_kind": goal_kind,
                                     "cross_episode_goal": (rec.extra or {}).get("cross_episode_goal", {}).get("difficulty") if goal_kind == "cross_episode" else None,
                                     "goal_shape": list(goal.shape), "goal_sha256": self.client.goal_sha256.get(0),
                                     "server_metadata": {k: v for k, v in (server or {}).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
                                     "connect_s": round(self.t_connect, 3)}

    def __call__(self, obs, info):
        return self.client.act({0: obs})[0]

    def finalize(self, result) -> None:
        if self.client is None:
            return
        st = self.client.stats()
        merged = dict((self.meta or {}).get("policy_client") or {})
        merged.update(st)
        if self.meta is not None:
            self.meta["policy_client"] = merged
        # run_isg copies the episode's policy_client over meta's: keep the bind-time record (alias, task id, goal kind / sha) in it
        result["policy_client"] = merged
        self.client.close()
