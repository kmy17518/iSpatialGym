#!/usr/bin/env python3
"""Patch an openpi isg-train224 checkout (eval serving copy) so one server prompts each request with its own task's exact
training prompt: a request may carry ``__task_name__`` (a task alias, or one per batch element); the wrapper then uses the
checkpoint's recorded ``task_prompts[alias]`` (b1k_metadata.json, the prompts the run trained with) instead of the server's
start-time prompt. Without the key nothing changes. Needed by the text-only checkpoints (the prompt is the task's text);
the goal checkpoints' prompt is the fixed goal scaffold whatever the task.

    openpi_task_prompts.py <openpi checkout>      (idempotent)
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
W = root / "src/openpi/shared/eval_b1k_wrapper.py"
S = root / "scripts/b1k/serve_b1k.py"
MARK = "# isg policy-eval: per-request task prompts"


def sub(text, old, new, path):
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


w = W.read_text()
if MARK not in w:
    w = sub(w, "        fixed_goals: dict[str, np.ndarray] | None = None,\n    ) -> None:",
            "        fixed_goals: dict[str, np.ndarray] | None = None,\n        task_prompts: dict[str, str] | None = None,\n    ) -> None:", W)
    w = sub(w, "        self.text_prompt = text_prompt\n",
            f"        self.text_prompt = text_prompt\n        {MARK}: alias -> exact training prompt (b1k_metadata task_prompts)\n"
            "        self.task_prompts = dict(task_prompts or {})\n", W)
    w = sub(w, "        processed_input = [\n",
            f"        {MARK}: `__task_name__` (one alias, or one per batch element) selects the recorded training prompt\n"
            "        names = obs.get(\"__task_name__\")\n"
            "        if names is None:\n"
            "            prompts = [self.text_prompt] * batch_size\n"
            "        else:\n"
            "            names = [str(names)] * batch_size if isinstance(names, (str, bytes)) else [str(n) for n in names]\n"
            "            if len(names) != batch_size:\n"
            "                raise ValueError(\"__task_name__ batch size does not match proprioception\")\n"
            "            missing = sorted({n for n in names if n not in self.task_prompts})\n"
            "            if missing:\n"
            "                raise ValueError(f\"no recorded training prompt for task(s) {missing}\")\n"
            "            prompts = [self.task_prompts[n] for n in names]\n"
            "        processed_input = [\n", W)
    w = sub(w, '                "prompt": self.text_prompt,\n', '                "prompt": prompts[i],\n', W)
    W.write_text(w)
s = S.read_text()
if MARK not in s:
    s = sub(s, "        fixed_goals=fixed_goals,\n    )\n",
            "        fixed_goals=fixed_goals,\n"
            f"        {MARK}: only the checkpoint's own recorded prompts, and only when no override was given\n"
            "        task_prompts=(metadata or {}).get(\"task_prompts\") if args.text_prompt is None and args.prompt_source is None else None,\n"
            "    )\n", S)
    S.write_text(s)
print("patched" if MARK in W.read_text() and MARK in S.read_text() else "NOT patched")
