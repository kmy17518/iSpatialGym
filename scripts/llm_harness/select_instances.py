#!/usr/bin/env python3
"""Select the 100-instance LLM-harness eval set from the 1000 v2 eval instances: one instance per task (the dataset's
``source.class_id``; 100 tasks x 10 instances), and record it.

    select_instances.py --instances <ispatialgym-eval-instances root> [--out scripts/llm_harness]

Rule (deterministic, reproducible from the Hugging Face dataset): for each task, the candidates are its 10 instances, narrowed
in order (a filter that would leave none is skipped):
  1. those with a cross-episode goal image (every task except camera relocalization, whose goal groups are singletons), so the
     cross-episode-goal eval covers every task that has one;
  2. those whose instance file carries no collection-time lighting draw (``ispatialgym.campaign.perturbations.lighting``; the
     evaluator ignores it anyway, 74 of the 1000 have one);
then one is drawn with ``random.Random("isg-llm-harness-100-v1:" + task)``.

Writes ``instances_100.json`` / ``instances_100.csv`` (dataset index, instance id, task, task class, task family, subtask, scene,
files, step budget, cross-episode goal difficulty, iors flag, the text-only prompt for iors tasks) and the subset lists
``subsets/full_100.txt`` and ``subsets/iors_35.txt`` (instance ids, one per line).
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "policy_eval"))
from summarize import TASK_CLASSES  # noqa: E402  (the 23 task classes, display names)

IORS = "inter_object_relation_satisfaction"


def budget(length):
    """run_isg's policy-mode step budget of an eval instance (ispatialgym.eval.instances.step_budget: max(300, 1.5 x demo length))."""
    return max(300, int(round(1.5 * int(length))))


def has_lighting_draw(tro_path):
    with open(tro_path) as f:
        tro = json.load(f)
    return bool((((tro.get("ispatialgym") or {}).get("campaign") or {}).get("perturbations") or {}).get("lighting"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="ispatialgym-eval-instances root (kmy17518/isg-standard-eval-224-instances-v2)")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    root = Path(a.instances)
    recs = [json.loads(l) for l in open(root / "metadata" / "instances.jsonl") if l.strip()]
    prompts = json.load(open(HERE / "text_prompts_iors.json"))["prompts"]
    by_task = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by_task[r["source"]["class_id"]].append(i)
    rows = []
    for task in sorted(by_task):
        cands = by_task[task]
        for keep in (lambda i: bool(recs[i]["files"].get("cross_episode_goal_rgb")),
                     lambda i: not has_lighting_draw(root / recs[i]["files"]["tro_state"])):
            narrowed = [i for i in cands if keep(i)]
            cands = narrowed or cands
        i = random.Random(f"isg-llm-harness-100-v1:{task}").choice(sorted(cands))
        r = recs[i]
        iid = os.path.basename(r["files"]["tro_state"])[: -len("-tro_state.json")]
        ceg = r.get("cross_episode_goal") or {}
        rows.append({"index": i, "instance_id": iid, "task": task, "task_class": TASK_CLASSES.get(r["task"], r["task"]),
                     "task_family": r["task_family"], "dataset_task": r["task"], "subtask": r["subtask"], "scene": r["scene"],
                     "demo_length": r["length"], "step_budget": budget(r["length"]), "iors": r["task_family"] == IORS,
                     "cross_episode_goal": ceg.get("difficulty") if r["files"].get("cross_episode_goal_rgb") else None,
                     "text_prompt": prompts.get(task) if r["task_family"] == IORS else None,
                     "tro_state": r["files"]["tro_state"], "goal_rgb": r["files"]["goal_rgb"],
                     "cross_episode_goal_rgb": r["files"].get("cross_episode_goal_rgb")})
    assert len(rows) == 100, len(rows)
    iors = [x for x in rows if x["iors"]]
    assert len(iors) == 35 and all(x["text_prompt"] for x in iors), (len(iors), [x["task"] for x in iors if not x["text_prompt"]])
    out = Path(a.out)
    (out / "subsets").mkdir(parents=True, exist_ok=True)
    json.dump({"rule": "one instance per task (source.class_id); prefer a cross-episode goal, then no collection lighting draw; "
                       "random.Random('isg-llm-harness-100-v1:' + task)",
               "dataset": "kmy17518/isg-standard-eval-224-instances-v2 (revision cf94275c)", "instances": rows},
              open(out / "instances_100.json", "w"), indent=1)
    with open(out / "instances_100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "subsets" / "full_100.txt").write_text("".join(x["instance_id"] + "\n" for x in rows))
    (out / "subsets" / "iors_35.txt").write_text("".join(x["instance_id"] + "\n" for x in iors))
    fam = collections.Counter(x["task_family"] for x in rows)
    print(f"100 instances ({dict(fam)}); 35 iors; cross-episode goal: {collections.Counter(x['cross_episode_goal'] for x in rows)}; "
          f"step budget total {sum(x['step_budget'] for x in rows)} (iors {sum(x['step_budget'] for x in iors)})")


if __name__ == "__main__":
    main()
