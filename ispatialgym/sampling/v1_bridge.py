"""v1_bridge (formerly ground_isg v1 bridge): iddl + icl (+ the representative selection) -> one v1 task spec per grounded alias
+ a job manifest for ``sample_instances.py``.

The v2 sampler (`docs/instance_sampling_spec.md` §9) is not implemented yet; until it is, this module grounds every
representative pick of ``ispatialgym/assets/icl/roles/representatives.csv`` into the **v1 spec schema**
(``ispatialgym/sampling/spec.py``) so the existing samplers can produce templates + instances today:

* one spec JSON per grounded alias under ``<out>/specs/<alias>.json`` with a single entry keyed by the v1 problem key
  (``<activity>-<def_id>``: the v1 dispatch is by key, so aliases that share a key live in separate files);
* ``<out>/jobs.json`` (one sampling job per alias: spec, problem key, scene, weight) for
  ``scripts/runpod/dispatch_samp.py`` / a local loop, ``<out>/problems.json`` (alias <-> v1 key <-> pick <-> slots),
  ``<out>/README.md``.

Grounding rules kept from the spec: categories -> synsets happen in the sampler (``knowledge``); ``:require`` grasp
pipelines -> explicit ``grasps.key`` from ``require/grasps.csv`` (the sampler never guesses a suffix); held-out models
are removed from every other pool (disembedding distractors); scene roles are bound per scene from the selection
(B4 inventory names); ``generation`` is seeded from the v1 entry of the same v1 key (the tuned numbers), with the
per-model frame parameters of the alignment targets from the icl columns.

Run::

    python -m ispatialgym.sampling.v1_bridge [--out ispatialgym/assets/task_specs/isg_reps] [--n-instances 2]
        [--slots representative[,variation_geometry,...]] [--subtask <id> ...]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
from typing import Dict, List, Optional

from ispatialgym import REPO_ROOT

ICL = os.path.join(REPO_ROOT, "ispatialgym", "assets", "icl")
IDDL = os.path.join(REPO_ROOT, "ispatialgym", "assets", "iddl")
V1_SPEC = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
SCENE_INVENTORY = os.path.join(REPO_ROOT, "runlogs", "scene_inventory_20260915", "summary.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_reps")

IORS = "inter_object_relation_satisfaction"
CM = "configuration_matching"
TEST_SCENE = "house_single_floor"
HELD_ROOM = "living_room_0"
TABLE_ROOM = "dining_room_0"

#: v2 subtask -> (v1 problem key, iddl file)
V1_KEY = {
    "camera_relocalization-standard": ("camera_relocalization-0", "camera_relocalization/standard.iddl"),
    "camera_relocalization-obstructed": ("camera_relocalization-0", "camera_relocalization/obstructed.iddl"),
    "object_scaling": ("object_scaling-0", "object_scaling/object_scaling.iddl"),
    "mental_rotation": ("mental_rotation-0", "mental_rotation/mental_rotation.iddl"),
    "disembedding": ("disembedding-0", "disembedding/disembedding.iddl"),
    f"{CM}-block_design": ("configuration_matching-0", "configuration_matching/block_design.iddl"),
    f"{CM}-articulation_open_small_scale": ("configuration_matching-1", "configuration_matching/articulation_open_small_scale.iddl"),
    f"{CM}-articulation_close_small_scale": ("configuration_matching-2", "configuration_matching/articulation_close_small_scale.iddl"),
    f"{CM}-articulation_open_large_scale": ("configuration_matching-1", "configuration_matching/articulation_open_large_scale.iddl"),
    f"{CM}-articulation_close_large_scale": ("configuration_matching-2", "configuration_matching/articulation_close_large_scale.iddl"),
    f"{CM}-attachment": ("configuration_matching-3", "configuration_matching/attachment.iddl"),
    "alignment-axial": ("world_alignment-0", "alignment/axial.iddl"),
    "alignment-world_vertical": ("world_alignment-1", "alignment/world_vertical.iddl"),
    f"{IORS}-ontop_small_scale": (f"{IORS}-0", "inter_object_relation_satisfaction/ontop_small_scale.iddl"),
    f"{IORS}-ontop_large_scale": (f"{IORS}-0", "inter_object_relation_satisfaction/ontop_large_scale.iddl"),
    f"{IORS}-under_small_scale": (f"{IORS}-1", "inter_object_relation_satisfaction/under_small_scale.iddl"),
    f"{IORS}-under_large_scale": (f"{IORS}-1", "inter_object_relation_satisfaction/under_large_scale.iddl"),
    f"{IORS}-inside": (f"{IORS}-2", "inter_object_relation_satisfaction/inside.iddl"),
    f"{IORS}-nextto": (f"{IORS}-3", "inter_object_relation_satisfaction/nextto.iddl"),
    f"{IORS}-touching": (f"{IORS}-4", "inter_object_relation_satisfaction/touching.iddl"),
}
#: relative sampling cost (job ordering on the fleet)
WEIGHT = {"camera_relocalization-standard": 1.5, "camera_relocalization-obstructed": 2.5, "object_scaling": 1.0, "mental_rotation": 1.0,
          "disembedding": 4.0, f"{CM}-block_design": 3.0, f"{CM}-attachment": 3.0, "alignment-world_vertical": 3.0,
          "alignment-axial": 3.0}


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_iddl(path: str) -> dict:
    """Alias template + :require forms (the literals are rendered by the v1 bddl_writer for now)."""
    text = open(path).read()
    alias = re.search(r"\(:alias\s+([^\s)]+)\)", text)
    req = re.findall(r"\((grasps|contacts|handle)\s+(\w+)(?:\s+(\w+))?\)", re.search(r"\(:require(.*?)\n\s*\)", text, re.S).group(1)) \
        if ":require" in text else []
    return {"alias": alias.group(1) if alias else None,
            "require": [{"kind": k, "role": r, "pipeline": p or None} for k, r, p in req]}


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


class Grounder:
    def __init__(self, icl_dir: str, n_instances: int, slots: List[str]):
        self.n_instances = n_instances
        self.slots = slots
        self.reps = read_csv(os.path.join(icl_dir, "roles", "representatives.csv"))
        self.cands = read_csv(os.path.join(icl_dir, "roles", "candidates.csv")) + read_csv(os.path.join(icl_dir, "roles", "candidates_extra.csv"))
        self.handles = read_csv(os.path.join(icl_dir, "require", "handles.csv"))
        with open(V1_SPEC) as f:
            self.v1 = json.load(f)
        with open(SCENE_INVENTORY) as f:
            self.inventory = json.load(f)["scenes"]
        self.cand_by = {}
        for r in self.cands:
            self.cand_by.setdefault((r["subtask"], r["role"], r["category"], r["model"]), r)
        self.heldout = {(r["category"], r["model"]) for r in self.reps if r["slot"] == "heldout" and r["kind"] != "scene"}
        self.picks: Dict[str, Dict[str, Dict[str, dict]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in self.reps:
            self.picks[r["pick"]][r["role"]][r["slot"]] = r
        self.problems: List[dict] = []
        self.jobs: List[dict] = []
        self.skipped: List[str] = []

    # -- helpers ------------------------------------------------------------------------------------------------------
    def cand(self, subtask: str, role: str, cat: str, model: str) -> dict:
        return self.cand_by.get((subtask, role, cat, model), {})

    def gen_from_v1(self, v1_key: str) -> dict:
        g = dict(self.v1.get(v1_key, {}).get("generation", {}))
        g.pop("_note", None)
        g["instances_per_template"] = self.n_instances
        g.pop("templates_per_scene", None)
        return g

    def rooms_of(self, scene: str, name: str) -> List[str]:
        d = self.inventory.get(scene, {})
        for sec in ("supports", "partners", "artic"):
            if name in d.get(sec, {}):
                return list(d[sec][name].get("rooms", []))
        return []

    def joint_for(self, cat: str, model: str, preferred: str = "") -> str:
        if preferred:
            return preferred
        for r in self.handles:
            if r["scene"] == "" and r["category"] == cat and r["model"] == model and r["verdict"] in ("handle", "edge", "manual"):
                return r["joint"]
        return ""

    def wall_regions(self, subtask: str, scene: str, room: str) -> List[dict]:
        out = []
        for r in self.cands:
            if r["subtask"] == subtask and r["role"] == "support" and r["scene"] == scene and r.get("room") == room and r.get("region_id"):
                out.append({"id": r["region_id"], "wall": r["model"], "room": room, "p0": json.loads(r["p0"]), "p1": json.loads(r["p1"]),
                            "normal": json.loads(r["normal"]), "hang_z": float(r["hang_z"] or 1.45)})
        return out

    def axis_dicts(self, subtask: str, cat: str, model: str, gen: dict) -> None:
        c = self.cand(subtask, "target", cat, model)
        key = f"{cat}/{model}"
        for col, gkey in (("frame_front", "front_axis"), ("frame_up", "up_axis"), ("frame_long", "long_axis")):
            if c.get(col):
                d = gen.get(gkey)
                d = d if isinstance(d, dict) else {"default": d} if d is not None else {}
                d[key] = c[col]
                gen[gkey] = d

    def scale_of(self, subtask: str, role: str, cat: str, model: str) -> Optional[list]:
        c = self.cand(subtask, role, cat, model)
        s = c.get("scale", "")
        if not s:
            return None
        try:
            v = json.loads(s)
            return [float(x) for x in (v if isinstance(v, list) else [v, v, v])]
        except Exception:
            try:
                return [float(x) for x in re.split(r"[ ,;]+", s.strip("[]()")) if x]
            except Exception:
                return None

    # -- grounding ------------------------------------------------------------------------------------------------------
    def ground_pick(self, pick: str, roles: Dict[str, Dict[str, dict]], slot: str) -> None:
        subtask = pick.split("#")[0]
        v1_key, iddl_file = V1_KEY[subtask]
        iddl = parse_iddl(os.path.join(IDDL, iddl_file))
        v1_entry = self.v1.get(v1_key, {})

        def pick_row(role: str) -> Optional[dict]:
            slots = roles.get(role, {})
            if slot in slots:
                return slots[slot]
            return slots.get("representative") if slot != "representative" else None

        alias_fill: Dict[str, str] = {}
        entry: dict = {"alias": v1_entry.get("alias"), "roles": {}, "scenes": {}, "generation": self.gen_from_v1(v1_key)}
        scene, rooms, fixed = TEST_SCENE, [], {}
        note = []
        req = {r["role"]: r for r in iddl["require"]}

        def spawned(v1_role: str, row: dict, pipeline: Optional[str], extra: Optional[dict] = None, count: Optional[int] = None) -> None:
            cat, model = row["category"], row["model"]
            spec = {"models": {cat: [model]}}
            if pipeline:
                key = row.get("grasp_key") or ""
                if not key:
                    note.append(f"{v1_role}: no admissible {pipeline} grasp key")
                spec["grasps"] = {"key": key}
            sc = self.scale_of(subtask, row["role"], cat, model)
            if sc:
                spec["scale"] = sc
            if row.get("kind") == "group" and row.get("dataset"):
                spec["dataset"] = row["dataset"]
            c = self.cand(subtask, row["role"], cat, model)
            if c.get("dataset"):
                spec["dataset"] = c["dataset"]
            if count:
                spec["count"] = count
            if extra:
                spec.update(extra)
            entry["roles"][v1_role] = spec

        target = pick_row("target") or pick_row("targets") or pick_row("obstacles")
        partner = pick_row("partner")
        support = pick_row("support")
        if target is None and subtask != "camera_relocalization-standard":
            self.skipped.append(f"{pick}/{slot}: no target row")
            return

        # ---- per subtask ------------------------------------------------------------------------------------------------
        if subtask == "camera_relocalization-standard":
            rooms = [HELD_ROOM]
            fixed["floor"] = {"room": HELD_ROOM}
            entry["roles"]["floor"] = {"fixed": True}
        elif subtask == "camera_relocalization-obstructed":
            rooms = [HELD_ROOM]
            fixed["floor"] = {"room": HELD_ROOM}
            entry["roles"]["floor"] = {"fixed": True}
            spawned("obstacles", target, None, count=1)
            alias_fill["obstacles"] = f"{target['category']}_1"
        elif subtask in ("object_scaling", "mental_rotation"):
            rooms = [HELD_ROOM]
            fixed["floor"] = {"room": HELD_ROOM}
            entry["roles"]["floor"] = {"fixed": True}
            spawned("target", target, "floating")
            alias_fill["target"] = target["category"]
        elif subtask == "disembedding":
            spawned("target", target, "tabletop")
            entry["roles"]["tray_source"] = dict(v1_entry["roles"]["tray_source"])
            entry["roles"]["tray_dest"] = dict(v1_entry["roles"]["tray_dest"])
            dis = json.loads(json.dumps(v1_entry["distractors"]))
            for cat, pool in dis.get("pools", {}).items():
                if not isinstance(pool, dict):
                    continue
                pool["same_category"] = [m for m in pool.get("same_category", []) if (cat, m) not in self.heldout]
                for oc, ms in list((pool.get("other_models") or {}).items()):
                    pool["other_models"][oc] = [m for m in ms if (oc, m) not in self.heldout]
            groups = []
            for grp in dis.get("lookalike_groups", []):
                members = grp.get("members") if isinstance(grp, dict) else grp
                keep = [m for m in members if tuple(m.split("/")) not in self.heldout]
                if len(keep) >= 2:
                    groups.append({**grp, "members": keep} if isinstance(grp, dict) else keep)
            dis["lookalike_groups"] = groups
            entry["distractors"] = dis
            alias_fill["target"] = target["category"]
            alias_fill["distractors"] = "distractors"
        elif subtask == f"{CM}-block_design":
            entry["roles"]["blocks"] = dict(v1_entry["roles"]["blocks"])
            alias_fill["targets"] = "block_4"
        elif subtask.startswith(f"{CM}-articulation") and "small" in subtask:
            joint = self.joint_for(target["category"], target["model"], target.get("joint", ""))
            spawned("artic", target, None, extra={"joint": joint, "handle_key": f"{target['category']}__{target['model']}"})
            alias_fill["target"] = target["category"]
        elif subtask.startswith(f"{CM}-articulation"):
            scene = target["scene"]
            rooms = self.rooms_of(scene, target["name"])
            fixed["artic"] = [target["name"]]
            entry["roles"]["artic"] = {"fixed": True, "joint": target["joint"], "handle_key": f"{target['category']}__{target['model']}",
                                       "handle_file": os.path.relpath(os.path.join(os.path.dirname(SCENE_INVENTORY), scene, "artic", target["name"] + ".json"), REPO_ROOT)}
            alias_fill["target"] = target["category"]
            if target.get("status", "").startswith("incomplete"):
                note.append("handle from the B4 heuristic (no canonical annotation)")
        elif subtask == f"{CM}-attachment":
            spawned("male", target, "attachment")
            spawned("female", partner, None)
            alias_fill["target"], alias_fill["partner"] = target["category"], partner["category"]
        elif subtask == "alignment-axial":
            spawned("target", target, None)
            self.axis_dicts(subtask, target["category"], target["model"], entry["generation"])
            alias_fill["target"] = target["category"]
        elif subtask == "alignment-world_vertical":
            rooms = [HELD_ROOM]
            fixed["floor"] = {"room": HELD_ROOM}
            entry["roles"]["floor"] = {"fixed": True}
            entry["roles"]["wall"] = {"fixed": True}
            spawned("target", target, None)
            self.axis_dicts(subtask, target["category"], target["model"], entry["generation"])
            regions = self.wall_regions(subtask, TEST_SCENE, HELD_ROOM)
            if regions:
                fixed["wall_regions"] = regions
                fixed["wall"] = sorted({r["wall"] for r in regions})
            alias_fill["target"], alias_fill["support"] = target["category"], "walls"
        elif subtask.startswith(IORS):
            pipeline = "insert" if subtask.endswith("inside") else "tabletop"
            spawned("target", target, pipeline)
            alias_fill["target"] = target["category"]
            if partner is not None and partner.get("kind") == "scene":
                scene = partner["scene"]
                rooms = self.rooms_of(scene, partner["name"])
                entry["roles"]["partner"] = {"fixed": True}
                fixed["partner"] = [partner["name"]]
                alias_fill["partner"] = partner["category"]
            elif partner is not None:
                spawned("partner", partner, None)
                alias_fill["partner"] = partner["category"]
        else:
            self.skipped.append(f"{pick}: unknown subtask")
            return

        # ---- support / table / floor ----------------------------------------------------------------------------------------
        if support is not None and support.get("kind") == "scene" and support["category"] not in ("walls", "floor"):
            scene = support["scene"]
            entry["roles"]["table"] = {"fixed": True}
            fixed["table"] = [support["name"]]
            rooms = rooms or self.rooms_of(scene, support["name"]) or [TABLE_ROOM]
            alias_fill["support"] = support["category"]
            if partner is not None and partner.get("kind") == "scene":
                # large-scale IORS: the init support must be near the partner; rooms = both
                rooms = sorted(set(rooms) | set(self.rooms_of(scene, partner["name"])))
        elif support is not None and support["category"] == "floor":
            alias_fill["support"] = "floor"
            # v1 IORS picks a table of the task rooms (defaults) - large-scale under may fall back to the partner's top
            note.append("init surface = floor requested; v1 uses a table of the task rooms / the partner top (under)")
        if "floor" not in entry["roles"]:
            entry["roles"]["floor"] = {"fixed": True}
        if "floor" not in fixed:
            fixed["floor"] = {"room": (rooms or [TABLE_ROOM])[0]}
        if not rooms:
            rooms = [TABLE_ROOM]
        entry["scenes"] = {scene: {"rooms": rooms, "fixed": fixed}}

        # ---- alias / ids ----------------------------------------------------------------------------------------------------
        alias = iddl["alias"]
        for k, v in alias_fill.items():
            alias = alias.replace("{" + k + "}", v)
        alias = re.sub(r"\{\w+\}", "x", alias)
        models = "__".join(f"{r}-{roles[r][slot if slot in roles[r] else 'representative']['model'] or roles[r]['representative'].get('name', '')}"
                           for r in ("target", "targets", "obstacles", "partner") if r in roles and roles[r])
        pid = safe(f"{alias}__{slot}__{models}")
        entry["_isg"] = {"subtask": subtask, "alias": alias, "pick": pick, "slot": slot, "iddl": iddl_file, "require": iddl["require"],
                         "scene": scene, "note": note}
        self.problems.append({"id": pid, "alias": alias, "subtask": subtask, "pick": pick, "slot": slot, "v1_key": v1_key, "scene": scene,
                              "roles": {r: {"category": roles[r][s]["category"], "model": roles[r][s]["model"], "name": roles[r][s].get("name", "")}
                                        for r in roles for s in roles[r] if s == slot or (slot != "representative" and s == "representative" and slot not in roles[r])},
                              "note": note, "entry": entry})

    def run(self, subtasks: Optional[List[str]] = None) -> None:
        for pick in sorted(self.picks):
            subtask = pick.split("#")[0]
            if subtasks and subtask not in subtasks:
                continue
            for slot in self.slots:
                if slot != "representative" and not any(slot in roles for roles in self.picks[pick].values()):
                    continue
                self.ground_pick(pick, self.picks[pick], slot)
        if not subtasks or "camera_relocalization-standard" in subtasks:
            roles = {"support": {}}
            self.ground_pick("camera_relocalization-standard#01", roles, "representative")

    def write(self, out: str) -> None:
        os.makedirs(os.path.join(out, "specs"), exist_ok=True)
        for p in self.problems:
            path = os.path.join(out, "specs", p["id"] + ".json")
            with open(path, "w") as f:
                json.dump({"_comment": f"v1_bridge: {p['alias']} ({p['pick']}, slot {p['slot']})", p["v1_key"]: p["entry"]}, f, indent=1)
            self.jobs.append({"id": p["id"], "spec": os.path.relpath(path, REPO_ROOT), "problem": p["v1_key"], "scene": p["scene"],
                              "n_instances": self.n_instances, "max_templates": 1, "weight": WEIGHT.get(p["subtask"], 3.0), "subtask": p["subtask"],
                              "slot": p["slot"], "alias": p["alias"]})
        with open(os.path.join(out, "jobs.json"), "w") as f:
            json.dump(self.jobs, f, indent=1)
        with open(os.path.join(out, "problems.json"), "w") as f:
            json.dump([{k: v for k, v in p.items() if k != "entry"} for p in self.problems], f, indent=1)
        by = collections.Counter(p["subtask"] for p in self.problems)
        lines = ["# ground_isg (v1 bridge) output", "", f"{len(self.problems)} grounded problems (slots: {', '.join(self.slots)}); "
                 f"one spec per alias under `specs/`, jobs in `jobs.json`.", "", "| subtask | problems |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(by.items())]
        if self.skipped:
            lines += ["", "Skipped:", ""] + [f"- {s}" for s in self.skipped]
        with open(os.path.join(out, "README.md"), "w") as f:
            f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icl", default=ICL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-instances", type=int, default=2)
    ap.add_argument("--slots", default="representative", help="comma list of slots to ground (representative, variation_texture, variation_geometry, heldout)")
    ap.add_argument("--subtask", action="append", default=None)
    args = ap.parse_args(argv)
    g = Grounder(args.icl, args.n_instances, [s.strip() for s in args.slots.split(",") if s.strip()])
    g.run(args.subtask)
    g.write(args.out)
    print(f"{len(g.problems)} problems -> {args.out} ({len(g.skipped)} skipped)")


if __name__ == "__main__":
    main()
