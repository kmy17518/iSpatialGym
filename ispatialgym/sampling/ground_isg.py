"""ground_isg — iddl + icl -> grounded BDDL problems + task customs list (``docs/instance_sampling_spec.md`` §9).

For every iddl file (``sampling/iddl.py``) and every category combination admitted by the icl
(``roles/constraints_compatible.csv``; ``--representative-only`` restricts the candidates to the picks of
``roles/representatives.csv``, ``--test-only`` to the ``test = 1`` rows of ``roles/candidates.csv``, ``--all`` takes
every candidate) this module

1. applies the row admissibility of spec §2.2 to the candidate models — ``ready`` (``not_ready``), the iddl
   ``:require`` lines (``missing_grasps:<pipeline>`` / ``missing_handle`` / ``missing_contacts``), the alignment frame
   record (``missing_frame``), symmetric mental-rotation targets (``symmetric_target``), obstacle naturalness
   (``not_natural_in_scene``) — plus the §3.12 look-alike pair rule (``lookalike_pair`` / ``lookalike_of_target``)
   and the global held-out exclusion (every ``heldout`` slot model is blacklisted in every role, reason ``heldout``);
2. renders and validates the BDDL (``sampling/bddl_from_iddl.py``; ids from ``activity_definitions/<activity>/
   problems.json``);
3. writes the customs entry (``sampling/customs.py``, spec §2.3): ``roles``, per-scene ``bindings`` of the scene
   roles, ``whitelist`` (the pick's models; the whole admissible pool with ``--full-whitelists``; always the whole
   admissible pool for group roles), ``blacklist`` with reasons, ``generation`` seeded from the existing entry or
   ``sampling/generation_defaults.py``, ``constraints`` (empty, except the soft ``missing_contacts_pending_B6``
   records while ``require/contacts.csv`` is header-only); IORS small scale (owner 2026-10-11, ``sampling/pair_rules.py``):
   a target bigger than its partner (ontop / under) or than the opening (inside) gets a uniform fit scale with a key tested at
   it, else it is withheld (``generation.target_fit``) - an inside target the entry's ``generation.target_up_axis`` names is
   measured standing on that axis; a nextto mirror pair gets ``generation.handedness``;
4. emits the **fallback aliases** of the primary (owner decision 2026-09-16 B; customs ``fallback_of`` /
   ``fallback_rank`` / ``fallback_kind`` / ``fallback_params``, the primary's ``fallbacks`` list, the same fields in
   ``problems.json``): large-scale IORS init surface (under: support -> the partner's own top -> floor; ontop: support
   -> floor), the wall / axial scale fallback (``<target>_s070`` = ``roles.target.scale`` 0.70, the pilot's
   ``template_meta.scale_hint`` when it recorded one; the axial paperbacks shrink their height only, ``<target>_h070``,
   owner 2026-10-11) and the small-scale articulation next-model fallback
   (``<target>_<model>``: the next admissible model of the category in variation / held-out order). A scheduler
   queues a fallback only after its primary yielded nothing — the sampler never falls back at run time;
5. calls the representative-test hook :func:`run_representative_test` (template + instance 0; implemented by the v2
   sampler, TODO D2 — skipped with a log line until ``ispatialgym.sampling.v2.core`` provides it).

Run::

    python -m ispatialgym.sampling.ground_isg [--representative-only | --test-only | --all] [--subtask ID ...]
        [--slots representative,variation_geometry,...] [--out-bddl DIR] [--customs FILE] [--scene S ...]
        [--full-whitelists] [--distractors same:2,other:1] [--legacy-activity-names] [--dry-run]
        [--no-fallbacks] [--fallback-scale 0.7] [--scale-hints GLOB|none] [--scale-hint ALIAS=SCALE ...]
        [--reseed-generation KEY,...|*]

Decisions taken where the docs leave room (all reported by the CLI, see ``--help`` and the run report):

* a group slot whose seeded ``size`` is a range (``distractors`` ``[1, 3]``) gets a concrete member list per problem
  (the BDDL enumerates objects): ``--distractors same:2,other:1`` = 2 same-category + 1 other-category (tier 1)
  distractors, filled from the pools in deterministic order; the grounded entry's ``generation.groups.distractors``
  carries the concrete ``size`` and the per-category ``members`` counts of the alias. A disembedding pick whose sheet rows
  list its pools (``same_category`` / ``other_category``, handoff 2026-10-10) draws its member set from them instead, the
  pattern read as caps (``Grounder._sheet_pool_members``); the puzzle's ``fixed_layout`` rows are its members
  (``Grounder._layout_members``);
* group alias tokens are ``<category>_<count>``; the custom block (``dataset = ispatialgym-assets``) uses its icl
  ``display_name`` (``block_4``, as ``docs/task_bddl_formats.md`` writes it);
* a scene role bound to the floor (large-scale IORS floor init; small-scale articulation targets taller than 0.6 m,
  §5.4) carries the alias token ``floor`` and the category ``floors``; two scene roles bound to one object (under
  large-scale: the partner's own top) are declared once;
* ``symmetric_180`` empty is read as the icl README says ("not flagged", i.e. no half-turn equivalence); the spec's
  ``missing_frame`` for an unfilled column is behind ``--strict-symmetric-180``.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import hashlib
import itertools
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ispatialgym import ASSETS_ROOT, REPO_ROOT
from ispatialgym.activity.problem import ISGProblemError
from ispatialgym.sampling import customs as C
from ispatialgym.sampling import generation_defaults as G
from ispatialgym.sampling import iddl as I
from ispatialgym.sampling import pair_rules as PR
from ispatialgym.sampling.bddl_from_iddl import (
    GROUNDER,
    GroundingError,
    RenderedProblem,
    RoleGrounding,
    write_problem,
)
from ispatialgym.sampling.customs import Constraint, Customs, CustomsEntry, RoleEntry, SceneEntry

log = logging.getLogger("ispatialgym.ground_isg")

VERSION = "2.0.0"
DEFAULT_ICL = os.path.join(ASSETS_ROOT, "icl")
DEFAULT_IDDL = I.IDDL_ROOT
DEFAULT_BDDL_ROOT = os.path.join(ASSETS_ROOT, "bddl", "activity_definitions")
DEFAULT_CUSTOMS = C.DEFAULT_CUSTOMS_PATH

MODES = ("representative", "test", "all")
SLOTS = ("representative", "variation_texture", "variation_geometry", "heldout")
#: The activity directory of each family; ``--legacy-activity-names`` maps ``alignment`` -> ``world_alignment``.
LEGACY_ACTIVITY_NAMES = {"alignment": "world_alignment"}

FLOOR_CATEGORY = "floors"
FLOOR_ALIAS_TOKEN = "floor"
FLOOR_MODEL_NAMES = ("floor", "floors")
ROOM_ROLE = "room"
TARGET_LIKE_ROLES = ("target", "targets", "obstacles")
CUSTOM_DATASET = "ispatialgym-assets"
#: §5.4: small-scale articulation targets taller than this stand on the floor.
FLOOR_SUPPORT_HEIGHT_M = 0.6
#: Support rows whose sheet height matches no surface of the support (owner 2026-10-11 06:27): sampled on the support's top
#: (``SimWorld.support_top``) whatever their ``top_z`` - keyed (scene, scene object name).
SUPPORT_TOP_SURFACE = {
    ("restaurant_diner", "flat_top_grill_szisej_0"),
    ("restaurant_diner", "stove_ykretu_0"),
    ("house_double_floor_upper", "bookcase_kdnyfh_0"),
}
#: Alias token suffix of a support the objects start inside (sheet note ``init=inside``): its BDDL starts them ``inside`` the
#: support instead of ``ontop``, so its rows never share an alias (and a BDDL) with the ``ontop`` rows of the category.
INSIDE_ALIAS_SUFFIX = "_inside"
#: Library key suffix a pick prefers when several keys of the pipeline are admissible (else the alphabetical first): owner
#: 2026-10-11 item 3 - the touching#03 book starts upright in the shelf, so it is picked with its upright-shelf key (``__bu``).
GRASP_KEY_SUFFIX_PREFERENCE = {"inter_object_relation_satisfaction-touching#03": "__bu"}
#: Owner 2026-10-11: a target bigger than its partner is shrunk before sampling (``pair_rules``: ontop / under -> the partner's top,
#: inside -> the opening) - the fit check of each subtask.
TARGET_FIT_SUBTASKS = {
    "inter_object_relation_satisfaction-ontop_small_scale": "on_partner",
    "inter_object_relation_satisfaction-under_small_scale": "on_partner",
    "inter_object_relation_satisfaction-inside": "inside",
}
#: Owner 2026-10-11: a nextto pair of mirror objects ends with the right one on the robot's right of the left one; the models' hands
#: go to ``generation.handedness`` and the sampler places the partner on the side that leaves the target on its own side.
HANDEDNESS_SUBTASKS = ("inter_object_relation_satisfaction-nextto",)
#: Hand of the mirror-pair models (no sheet column says it): the side of the sole's arch (the medial side) in the canonical frame, toe
#: along +x and up +z - arch on +y (the shoe's own left) = a right shoe. Measured on the visual meshes (sole outline within 1 cm of the
#: bottom, per tenth of the length) 2026-10-11: the nextto#04 targets are right shoes, their partners the left shoes of the pairs.
HANDEDNESS = {
    ("gym_shoe", "kmcbym"): "right", ("gym_shoe", "mcfcwq"): "left",  # the pair of BEHAVIOR's putting_shoes_on_rack
    ("gym_shoe", "picdro"): "right", ("gym_shoe", "ugbxip"): "left",
    ("gym_shoe", "pblxfp"): "right", ("gym_shoe", "rcozpq"): "left",
}
#: §2.2 grasp verdicts: ``ok``, or ``flag`` with flags ⊆ {one_finger}.
TOLERATED_GRASP_FLAGS = {"one_finger"}
HANDLE_ADMISSIBLE_VERDICTS = ("handle", "edge", "manual")
PINCH_RANGE_M = (0.015, 0.080)
PENDING_CONTACTS_REASON = "missing_contacts_pending_B6"

#: Fallback aliases (owner decision 2026-09-16 B). Subtasks with an init-surface chain (§5.6: under support -> the
#: partner's own top -> floor; ontop support -> floor), with a scale fallback (§5.5 / §6.10 ``scale_hint``) and with a
#: next-model fallback (small-scale articulation: the next candidate model of the category in variation / held-out order).
INIT_SURFACE_FALLBACK_SUBTASKS = {
    "inter_object_relation_satisfaction-under_large_scale": ("partner", "floor"),
    "inter_object_relation_satisfaction-ontop_large_scale": ("floor",),
}
SCALE_FALLBACK_SUBTASKS = ("alignment-world_vertical", "alignment-axial")
#: Scale fallbacks that shrink the target's height only, never the whole model (owner 2026-10-11: "if alignment axial paperback
#: fails try scaling only the height (no uniform scale)"): subtask -> target categories. A book's height is the long side of its
#: cover, the model axis of its reviewed ``frame_long`` (the sheet's paperbacks lie cover up: x = 0.30 m cover height, y = 0.22 m
#: width, z = the 8-10 mm thickness the sheet spawns at 2-2.5x for the 2 cm contact class); width and thickness keep the pick's
#: spawn scale. Alias token ``<target>_hNNN`` (:func:`scale_token`).
HEIGHT_SCALE_FALLBACK = {"alignment-axial": ("paperback_book",)}
NEXT_MODEL_FALLBACK_SUBTASKS = ("configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale")
NEXT_MODEL_SLOT_ORDER = ("variation_geometry", "variation_texture", "heldout")
#: The single scale step used when the pilot left no ``template_meta.scale_hint`` for the primary alias.
DEFAULT_FALLBACK_SCALE = 0.7
#: Where the pilot wrote its template metadata (``scale_hint`` of the wall / axial templates), relative to the repo.
DEFAULT_SCALE_HINT_GLOB = os.path.join("runlogs", "runpod", "pilot", "*", "v2__*", "*", "*", "template_meta.json")
#: Disembedding distractor rows of the representative sheet (handoff 2026-10-10, owner 2026-10-11): a pick lists its pools as
#: ``same_category`` (the target's category) / ``other_category`` rows, or the puzzle's ``fixed_layout`` parts; a pick without
#: such rows keeps the ``constraints_compatible.csv`` pools (:meth:`Grounder._distractor_members`).
SHEET_POOL_SLOTS = ("same_category", "other_category")
SHEET_LAYOUT_SLOT = "fixed_layout"
#: Pool row note of a pick whose distractors take the target's pose (``pose: lying like the bottle_of_milk target ...``).
POSE_NOTE_RE = re.compile(r"\bpose: (lying|standing)\b")
#: Fixed-layout row note naming the layout file (repo-relative).
LAYOUT_NOTE_RE = re.compile(r"(ispatialgym/assets/\S+?\.json)")

_TRUE = ("1", "true", "True", "yes")


# ----------------------------------------------------------------------------------------------------------------
# icl access
# ----------------------------------------------------------------------------------------------------------------


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _is_custom_asset(category: Optional[str], model: Optional[str]) -> bool:
    """``category/model`` is one of the repo's own converted assets (``assets/datasets/ispatialgym-assets/objects``)."""
    return bool(category and model) and os.path.isdir(os.path.join(ASSETS_ROOT, "datasets", CUSTOM_DATASET, "objects", category, model))


def _split_rooms(value: str) -> List[str]:
    return [r.strip() for r in (value or "").split(",") if r.strip()]


def _parse_scale(value: str) -> Optional[List[float]]:
    if not value:
        return None
    try:
        v = json.loads(value)
    except Exception:
        try:
            v = [float(x) for x in re.split(r"[ ,;]+", value.strip("[]()")) if x]
        except Exception:
            return None
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    return [float(x) for x in v]


def _pad_clearance(pinch_width: float) -> float:
    """``ispatialgym.robot.conventions.pad_clearance`` (F5) -- the same formula (0.10 - width) / 2 when the conventions
    module is not importable (it needs torch)."""
    try:
        from ispatialgym.robot.conventions import pad_clearance

        return max(0.0, float(pad_clearance(pinch_width)))
    except Exception:
        return max(0.0, 0.5 * (0.10 - float(pinch_width)))


def _scene_object_key(row: Dict[str, str]) -> str:
    """The name a scene candidate row is bound by: the wall region id for wall regions, else the scene object name."""
    return row.get("region_id") or row.get("model") or ""


def footprint_dims(bbox: Sequence[float], pose: Optional[str] = None) -> Tuple[float, float]:
    """``(long, short)`` horizontal sides of an icl bbox ``(x, y, z)`` in the distractor pose of the pick: canonical = (x, y),
    ``lying`` = the two largest sides (long axis horizontal, on the broadest side), ``standing`` = the two smallest (long axis
    vertical)."""
    bx, by, bz = (float(v) for v in bbox)
    if pose == "lying":
        d = sorted((bx, by, bz), reverse=True)[:2]
    elif pose == "standing":
        d = sorted((bx, by, bz))[:2]
    else:
        d = [bx, by]
    return max(d), min(d)


def admissible_draw(draw: Sequence[Tuple[str, str]], target_category: str, same_max: int, other_max: int,
                    group_of: Dict[Tuple[str, str], str]) -> bool:
    """The handoff's draw rule over the sheet pools (the ``same:2, other:1`` pattern read as caps): at most ``same_max`` models
    of the target's category, at most ``other_max`` of each other category, at most one per look-alike group."""
    per_cat = collections.Counter(c for c, _ in draw)
    if per_cat.get(target_category, 0) > same_max:
        return False
    if any(n > other_max for c, n in per_cat.items() if c != target_category):
        return False
    groups = [group_of[cm] for cm in draw if cm in group_of]
    return len(groups) == len(set(groups))


def count_every_draw(pool: Sequence[Tuple[str, str]], area: Dict[Tuple[str, str], float], budget_m2: float, size_max: int,
                     admissible) -> Tuple[int, float, List[Tuple[Tuple[str, str], ...]]]:
    """``(n, worst_m2, draws)``: ``n`` = the largest draw size (and every size below it) whose every admissible draw of the pool
    fits ``budget_m2``, ``worst_m2`` = the largest of those draws, ``draws`` = the admissible draws of size ``n`` in pool order."""
    n_ok, worst, ok_draws = 0, 0.0, []
    for n in range(1, int(size_max) + 1):
        draws = [d for d in itertools.combinations(pool, n) if admissible(d)]
        if not draws:
            break
        largest = max(sum(area[cm] for cm in d) for d in draws)
        if largest > budget_m2:
            break
        n_ok, worst, ok_draws = n, largest, draws
    return n_ok, worst, ok_draws


def stable_index(key: str, n: int) -> int:
    """An index in ``range(n)`` fixed by ``key`` (a re-grounding draws the same)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % int(n)


def support_surface(row: Dict[str, str], label: str) -> Optional[Dict[str, Any]]:
    """The surface a support row of the selection names (note ``init=<ontop|inside>; top_z=<m above the floor> ...``) as the
    customs ``bindings.support_surfaces`` entry ``{"name", "init", "top_z", "row"}`` (``row`` = the combination label, which
    tells a re-grounding which entry it replaces), ``"surface": "top"`` for :data:`SUPPORT_TOP_SURFACE`; None when the note
    gives no ``top_z``. The sampler reads it (``tabletop.support_surface``)."""
    note = row.get("note") or ""
    m = re.search(r"\btop_z=([0-9]*\.?[0-9]+)", note)
    if not m:
        return None
    name = row.get("name") or row.get("model") or ""
    init = re.search(r"\binit=([a-z]+)", note)
    out: Dict[str, Any] = {"name": name, "init": init.group(1) if init else "ontop", "top_z": float(m.group(1)), "row": label}
    if (row.get("scene") or "", name) in SUPPORT_TOP_SURFACE:
        out.update({"surface": "top", "why": "owner 2026-10-11: the sheet height matches no surface of the support"})
    return out


def load_scale_hints(pattern: Optional[str] = None) -> Dict[str, float]:
    """``{alias: scale_hint}`` from the pilot's ``template_meta.json`` files (the diagnostic pass of §5.5 writes the
    largest feasible uniform scale; several runs -> the largest hint, i.e. the one closest to the native scale).
    The alias is the ``alias`` field of the meta (fallback: the ``v2__<alias>`` job directory)."""
    import glob

    pattern = pattern or os.path.join(REPO_ROOT, DEFAULT_SCALE_HINT_GLOB)
    out: Dict[str, float] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                meta = json.load(f)
        except Exception:
            continue
        hint = meta.get("scale_hint")
        if hint in (None, "", False):
            continue
        try:
            hint = float(hint)
        except (TypeError, ValueError):
            continue
        alias = meta.get("alias")
        if not alias:
            job = next((p[4:] for p in path.split(os.sep) if p.startswith("v2__")), None)
            alias = job
        if not alias:
            continue
        out[alias] = max(out.get(alias, 0.0), hint)
    return out


def _parse_vec(value: str) -> Optional[List[float]]:
    if not value:
        return None
    try:
        return [float(x) for x in json.loads(value)]
    except Exception:
        return None


def _asset_bbox(category: str, model: str) -> Optional[List[float]]:
    """``bbox_size`` (native extents, m) of the asset's ``misc/metadata.json`` - the repo's own assets, then BEHAVIOR-1K under
    ``$OMNIGIBSON_DATA_PATH`` (default ``Behavior-1K/datasets``); None when neither has the model."""
    data = os.environ.get("OMNIGIBSON_DATA_PATH") or os.path.join(REPO_ROOT, "Behavior-1K", "datasets")
    for root in (os.path.join(ASSETS_ROOT, "datasets", CUSTOM_DATASET), os.path.join(data, "behavior-1k-assets")):
        path = os.path.join(root, "objects", category, model, "misc", "metadata.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                size = json.load(f).get("bbox_size")
            return [float(v) for v in size] if size and len(size) == 3 else None
        except (OSError, ValueError, TypeError, AttributeError):
            return None
    return None


def _role_scale(re_: RoleEntry, category: str, model: str) -> Optional[List[float]]:
    """The spawn scale a customs role gives ``category/model`` (``"<cat>/<model>"``, the bare model, ``"default"``; None = unit)."""
    val = re_.scale.get(C.model_key(category, model), re_.scale.get(model, re_.scale.get("default")))
    if val is None:
        return None
    return [float(v) for v in val] if isinstance(val, (list, tuple)) else [float(val)] * 3


class ICL:
    """The candidate list: ``roles/`` + ``require/`` sheets with the indexes the grounder needs."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or DEFAULT_ICL
        roles = os.path.join(self.root, "roles")
        req = os.path.join(self.root, "require")
        self.candidates: List[Dict[str, str]] = []
        for r in _read_csv(os.path.join(roles, "candidates.csv")):
            r["_extra"] = ""
            self.candidates.append(r)
        for r in _read_csv(os.path.join(roles, "candidates_extra.csv")):
            r["_extra"] = "1"
            self.candidates.append(r)
        if not self.candidates:
            raise GroundingError(f"no candidates under {roles}")
        self.by_sr: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        self.by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        self.bbox_of_model: Dict[Tuple[str, str], Optional[List[float]]] = {}  # (category, model) -> native bbox (any candidates row / asset)
        for r in self.candidates:
            self.by_sr[(r["subtask"], r["role"])].append(r)
            self.by_key.setdefault((r["subtask"], r["role"], r["category"], r["model"]), r)
            bbox = self._row_bbox(r)
            if bbox is not None and r.get("kind") != "scene":
                self.bbox_of_model.setdefault((r["category"], r["model"]), bbox)
        self.representatives = _read_csv(os.path.join(roles, "representatives.csv"))
        # pick -> role -> slot -> rows (several rows per slot for the roles the selection lists more than once, e.g.
        # the A3 support triplets); insertion order = sheet order
        self.picks: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in self.representatives:
            self.picks[r["pick"]][r["role"]].setdefault(r["slot"], []).append(r)
        self.heldout_models: Set[Tuple[str, str]] = {
            (r["category"], r["model"]) for r in self.representatives if r["slot"] == "heldout" and r["kind"] != "scene" and r["model"]
        }
        # held-out scene objects are per subtask (a scene instance can be the representative partner of one subtask
        # and the generalization instance of another); held-out models are global (spec §3.12 / A9)
        self.heldout_scene: Set[Tuple[str, str, str, str]] = {
            (r["subtask"], r["role"], r["scene"], r["name"]) for r in self.representatives if r["slot"] == "heldout" and r["kind"] == "scene"
        }
        self.compat = _read_csv(os.path.join(roles, "constraints_compatible.csv"))
        self.conflicting = _read_csv(os.path.join(roles, "constraints_conflicting.csv"))
        self.lookalike_groups: Dict[str, Set[Tuple[str, str]]] = collections.defaultdict(set)
        for r in self.conflicting:
            if r.get("relation", "lookalike") == "lookalike":
                self.lookalike_groups[r["group_id"]].add((r["category"], r["model"]))
        self.lookalike_of: Dict[Tuple[str, str], Set[Tuple[str, str]]] = collections.defaultdict(set)
        for members in self.lookalike_groups.values():
            for m in members:
                self.lookalike_of[m] |= members
        # require sheets
        self.grasps = _read_csv(os.path.join(req, "grasps.csv"))
        self.grasps_ok: Dict[Tuple[str, str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        # per library key: the spawn scale its grasps were tested at and its rest orientation (``scale`` / ``rest_quat_wxyz``)
        self.key_scale: Dict[str, Optional[List[float]]] = {}
        self.key_rest: Dict[str, Optional[List[float]]] = {}
        for g in self.grasps:
            if self.grasp_row_admissible(g):
                self.grasps_ok[(g["category"], g["model"], g["require"])].append(g)
            self.key_scale.setdefault(g["key"], _parse_scale(g.get("scale") or ""))
            self.key_rest.setdefault(g["key"], _parse_vec(g.get("rest_quat_wxyz") or ""))
        self.handles = _read_csv(os.path.join(req, "handles.csv"))
        self.handles_canonical: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        self.handles_instance: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        for h in self.handles:
            if h.get("scene"):
                self.handles_instance[(h["scene"], h["instance"])].append(h)
            else:
                self.handles_canonical[(h["category"], h["model"])].append(h)
        self.contacts = _read_csv(os.path.join(req, "contacts.csv"))
        self.contacts_by_model: Dict[Tuple[str, str], List[Dict[str, str]]] = collections.defaultdict(list)
        for c in self.contacts:
            self.contacts_by_model[(c["category"], c["model"])].append(c)

    # -- candidates -------------------------------------------------------------------------------------------------
    def rows(self, subtask: str, role: str) -> List[Dict[str, str]]:
        return list(self.by_sr.get((subtask, role), []))

    def row(self, subtask: str, role: str, category: str, model: str) -> Optional[Dict[str, str]]:
        return self.by_key.get((subtask, role, category, model))

    @staticmethod
    def _row_bbox(row: Dict[str, str]) -> Optional[List[float]]:
        try:
            bbox = [float(row[k]) for k in ("bbox_x", "bbox_y", "bbox_z")]
        except (KeyError, TypeError, ValueError):
            return None
        return bbox if all(v > 0 for v in bbox) else None

    def model_bbox(self, subtask: str, role: str, category: str, model: str) -> Optional[List[float]]:
        """Native bounding box (m, canonical frame, scale 1) of a model: its candidates row of ``(subtask, role)``, else any
        candidates row of the model, else the asset's ``bbox_size`` (sheet models without a candidates row: steaks, place mats)."""
        row = self.row(subtask, role, category, model)
        bbox = self._row_bbox(row) if row is not None else None
        if bbox is None and (category, model) not in self.bbox_of_model:
            self.bbox_of_model[(category, model)] = _asset_bbox(category, model)
        return bbox or self.bbox_of_model.get((category, model))

    def kind_of(self, subtask: str, role: str) -> str:
        kinds = {r["kind"] for r in self.rows(subtask, role)}
        if not kinds:
            raise GroundingError(f"{subtask}: role {role!r} has no rows in roles/candidates.csv (a role without rows is an error)")
        if len(kinds) > 1:
            raise GroundingError(f"{subtask}: role {role!r} has several kinds {sorted(kinds)}")
        return kinds.pop()

    def has_role(self, subtask: str, role: str) -> bool:
        return bool(self.by_sr.get((subtask, role)))

    def room_rows(self, subtask: str) -> List[Dict[str, str]]:
        return self.rows(subtask, ROOM_ROLE)

    def scene_rows(self, subtask: str, role: str, category: Optional[str] = None, scene: Optional[str] = None) -> List[Dict[str, str]]:
        out = []
        for r in self.rows(subtask, role):
            if r["kind"] != "scene":
                continue
            if category is not None and r["category"] != category:
                continue
            if scene is not None and r["scene"] != scene:
                continue
            out.append(r)
        return out

    # -- compatibility ------------------------------------------------------------------------------------------------
    def partner_rows(self, subtask: str, target_category: str, target_model: Optional[str] = None) -> List[Dict[str, str]]:
        out = []
        for r in self.compat:
            if r["subtask"] != subtask or r["relation"] != "partner_of" or r["category_a"] != target_category:
                continue
            if r.get("model_a") and target_model is not None and r["model_a"] != target_model:
                continue
            out.append(r)
        return out

    def distractor_rows(self, target_category: str, relation: str) -> List[Dict[str, str]]:
        return [
            r for r in self.compat
            if r["subtask"] == "disembedding" and r["relation"] == relation and r["category_a"] == target_category
        ]

    def lookalikes(self, category: str, model: str) -> Set[Tuple[str, str]]:
        return set(self.lookalike_of.get((category, model), set()))

    def are_lookalikes(self, a: Tuple[str, str], b: Tuple[str, str]) -> bool:
        return a != b and b in self.lookalike_of.get(a, set())

    # -- require ------------------------------------------------------------------------------------------------------
    @staticmethod
    def grasp_row_admissible(g: Dict[str, str]) -> bool:
        if g.get("verdict") == "ok":
            return True
        if g.get("verdict") == "flag":
            flags = {f.strip() for f in (g.get("flags") or "").split(",") if f.strip()}
            return flags <= TOLERATED_GRASP_FLAGS
        return False

    def admissible_grasps(self, category: str, model: str, pipeline: str) -> List[Dict[str, str]]:
        """§2.2: rows with ``require = grasps:<pipeline>`` and an admissible verdict (``handle_half`` rows are filed
        under ``grasps:floating`` by the builder, so knives count as floating)."""
        return list(self.grasps_ok.get((category, model, f"grasps:{pipeline}"), []))

    def grasp_key(self, category: str, model: str, pipeline: str) -> Optional[str]:
        rows = self.admissible_grasps(category, model, pipeline)
        keys = sorted({r["key"] for r in rows})
        return keys[0] if keys else None

    @staticmethod
    def handle_row_admissible(h: Dict[str, str], instance: bool) -> bool:
        if h.get("verdict") not in HANDLE_ADMISSIBLE_VERDICTS:
            return False
        if (h.get("review_graspable") or "") == "False":
            return False
        if instance and h.get("width_m"):
            try:
                w = float(h["width_m"])
            except ValueError:
                return False
            if not (PINCH_RANGE_M[0] <= w <= PINCH_RANGE_M[1]):
                return False
        return True

    def admissible_handles(self, category: str, model: str, scene: Optional[str] = None, name: Optional[str] = None) -> List[Dict[str, str]]:
        if scene and name:
            rows = self.handles_instance.get((scene, name), [])
            return [h for h in rows if self.handle_row_admissible(h, instance=True)]
        rows = self.handles_canonical.get((category, model), [])
        return [h for h in rows if self.handle_row_admissible(h, instance=False)]

    def contacts_available(self) -> bool:
        return bool(self.contacts)

    def admissible_contacts(self, category: str, model: str) -> List[Dict[str, str]]:
        return [c for c in self.contacts_by_model.get((category, model), []) if c.get("kind") == "push" or c.get("contact_role") == "push"]


# ----------------------------------------------------------------------------------------------------------------
# Admissibility (spec §2.2, §3.12, §3.13)
# ----------------------------------------------------------------------------------------------------------------


def _is_ready(row: Dict[str, str]) -> bool:
    if row.get("kind") == "scene":
        return True
    v = (row.get("ready") or "").strip()
    if v == "0":
        return False
    if v in _TRUE:
        return True
    try:
        from ispatialgym.sampling import knowledge

        return knowledge.is_ready(row["model"])
    except Exception:  # KB unavailable: the icl column is the only evidence
        return True


WALL_SUBTASKS = ("alignment-world_vertical",)
#: §5.5 frame of a wall target: ``generation.<key>`` (per-model dict form) <- icl ``roles/candidates.csv`` column
WALL_FRAME_KEYS = (("front_axis", "frame_front"), ("up_axis", "frame_up"))


def _frame_ok(subtask: str, row: Dict[str, str], strict_symmetric_180: bool) -> Optional[str]:
    """§2.2: alignment targets need a reviewed frame record; wall targets ``symmetric_180`` (strict mode)."""
    front, up, long_ = (row.get(k) or "" for k in ("frame_front", "frame_up", "frame_long"))
    if subtask == "alignment-axial":
        ok = bool(front or up) and bool(long_)
    elif subtask in WALL_SUBTASKS:
        ok = bool(front) and bool(up)
        if ok and strict_symmetric_180 and (row.get("symmetric_180") or "") not in ("0", "1"):
            ok = False
    else:
        return None
    return None if ok else "missing_frame"


def wall_frame_of(row: Dict[str, str], subtask: str) -> Dict[str, str]:
    """The reviewed frame of a wall target row as ``generation`` keys: ``front_axis`` / ``up_axis`` from the icl
    ``frame_front`` / ``frame_up`` columns; empty columns are left out (the sampler rejects an alias without a frame —
    ``missing_frame`` — instead of defaulting one)."""
    out: Dict[str, str] = {}
    for gen_key, col in WALL_FRAME_KEYS:
        val = (row.get(col) or "").strip()
        if val:
            out[gen_key] = val if val[0] in "+-" else "+" + val
    return out


def write_wall_frame(generation: Dict[str, Any], subtask: str, cat: str, model: str, row: Dict[str, str]) -> Dict[str, str]:
    """Owner 2026-09-16 22:55 (spec §5.5 / §6.9): the reviewed icl frame of a whitelisted wall target is written into the
    alias as ``generation.front_axis`` / ``up_axis`` in the per-model dict form ``{"<category>/<model>": "+z", ...}``
    (``AlignmentWall.target_param`` resolves it) — no wall alias carried the keys before, so every target
    hung in the painting-shaped default (+x / +z: the license plate table-flat, run 0916-212037). Other models' entries
    of the dict are kept; a non-dict value left by hand is replaced by the dict (the sampler never defaults). Returns
    the frame written."""
    frame = wall_frame_of(row, subtask)
    for gen_key, val in frame.items():
        cur = generation.get(gen_key)
        if not isinstance(cur, dict):
            cur = {}
        cur[C.model_key(cat, model)] = val
        generation[gen_key] = cur
    return frame


def _symmetry_tags(row: Dict[str, str]) -> List[str]:
    try:
        return C.parse_symmetry_tags(row.get("symmetry") or "")
    except C.CustomsError:
        return []


@dataclass
class Verdict:
    """Admissibility of one candidate model for one role: ``reason`` (None = admitted), ``pending`` (soft records),
    the admissible handle rows (articulation) and the grasp key of the pipeline. ``source`` names who rejected the
    model (``ground_isg`` = the §2.2 checks; ``pilot`` = an owner / pilot blacklist given on the command line)."""

    reason: Optional[str] = None
    pending: List[str] = field(default_factory=list)
    handles: List[Dict[str, str]] = field(default_factory=list)
    grasp_key: Optional[str] = None
    source: str = "ground_isg"


class Admissibility:
    def __init__(self, icl: ICL, strict_symmetric_180: bool = False, extra_blacklist: Optional[Dict[Tuple[str, str, str, str], str]] = None):
        self.icl = icl
        self.strict_symmetric_180 = strict_symmetric_180
        #: (subtask, role, category, model) -> reason: models the pilot rejected (``--blacklist``), checked first
        self.extra_blacklist = dict(extra_blacklist or {})

    def check(self, iddl: I.IDDL, role: str, row: Dict[str, str], scene: Optional[str] = None) -> Verdict:
        """§2.2 row admissibility of ``row`` (a ``roles/candidates.csv`` row of ``(iddl.subtask, role)``)."""
        v = Verdict()
        subtask = iddl.subtask
        category, model = row["category"], row["model"]
        is_scene = row.get("kind") == "scene"
        pilot = self.extra_blacklist.get((subtask, role, category, model))
        if pilot:
            v.reason, v.source = pilot, "pilot"
            return v
        if not is_scene and (category, model) in self.icl.heldout_models:
            v.reason = "heldout"
            return v
        if not _is_ready(row):
            v.reason = "not_ready"
            return v
        for req in iddl.require_of(role):
            if req.kind == "grasps":
                rows = self.icl.admissible_grasps(category, model, req.pipeline or "")
                if not rows:
                    v.reason = req.missing_reason
                    return v
                v.grasp_key = sorted({r["key"] for r in rows})[0]
            elif req.kind == "handle":
                if is_scene:
                    handles = self.icl.admissible_handles(category, model, scene=row.get("scene"), name=model)
                else:
                    handles = self.icl.admissible_handles(category, model)
                if not handles:
                    v.reason = req.missing_reason
                    return v
                v.handles = handles
            elif req.kind == "contacts":
                if self.icl.contacts_available():
                    if not self.icl.admissible_contacts(category, model):
                        v.reason = req.missing_reason
                        return v
                else:
                    v.pending.append(PENDING_CONTACTS_REASON)
        frame = _frame_ok(subtask, row, self.strict_symmetric_180)
        if frame:
            v.reason = frame
            return v
        if subtask == "mental_rotation" and role == "target" and _symmetry_tags(row):
            v.reason = "symmetric_target"
            return v
        if role == "obstacles" and scene is not None:
            natural = [s.strip() for s in (row.get("natural_scenes") or "").split(",") if s.strip()]
            if natural and scene not in natural:
                v.reason = "not_natural_in_scene"
                return v
        return v


def choose_joint(handles: List[Dict[str, str]], preferred: Optional[str] = None) -> Optional[Tuple[str, Dict[str, str]]]:
    """The joint of an articulation target from its admissible handle rows: the preferred joint when it has an
    admissible row, else the reviewer's ``chosen`` row, else bar / knob handles before edge pinches (name order)."""
    if not handles:
        return None
    if preferred:
        rows = [h for h in handles if h.get("joint") == preferred]
        if not rows:
            return None
        handles = rows
    order = {"handle": 0, "manual": 1, "edge": 2}

    def key(h):
        return (0 if (h.get("chosen") or "") in _TRUE else 1, order.get(h.get("verdict"), 9), h.get("name") or "")

    best = sorted(handles, key=key)[0]
    return best["joint"], best


# ----------------------------------------------------------------------------------------------------------------
# Combinations
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RolePick:
    """One role of a combination: the grounded category and its model pool / scene objects."""

    role: str
    kind: str  # sampleable | group | scene
    category: Optional[str] = None  # sampleable / scene (``floors`` for the floor)
    models: List[Tuple[str, str]] = field(default_factory=list)  # whitelist seeds [(category, model)]
    members: Dict[str, int] = field(default_factory=dict)  # group: category -> count (alias order)
    pools: Dict[str, List[str]] = field(default_factory=dict)  # group: category -> admissible pool (whitelist)
    scene_objects: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)  # scene: scene -> candidate rows
    same_as: Optional[str] = None  # scene role bound to another role's object ("partner") or to the floor ("floor")
    joint: Optional[str] = None  # articulation: the pick's joint
    pick_row: Optional[Dict[str, str]] = None
    allow_heldout: bool = False  # the heldout slot itself was requested for this role
    budget: Dict[str, Any] = field(default_factory=dict)  # distractors: the §6.4 tray budget maths (generation.distractor_budget)
    surface: Optional[Dict[str, Any]] = None  # support: the surface the pick row names (:func:`support_surface`)
    pose: Any = None  # distractors: the target's pose they take (generation.distractor_pose: lying | standing | {model: pose})
    layout: Dict[str, Any] = field(default_factory=dict)  # distractors: the puzzle's fixed layout (generation.fixed_layout)

    @property
    def is_floor(self) -> bool:
        return self.kind == "scene" and self.category == FLOOR_CATEGORY

    @property
    def starts_inside(self) -> bool:
        return bool(self.surface) and self.surface.get("init") == "inside"

    def alias_token(self, icl: ICL, subtask: str, scale: Optional[float] = None, with_model: bool = False, height_only: bool = False) -> str:
        """The ``{role}`` slot value: the category (``floor`` for the floor; group roles ``<category>_<count>`` tokens),
        with ``_inside`` for a support the objects start inside (``bookcase_inside``), ``_s<NNN>`` for a scaled fallback
        (``picture_s070`` = spawn scale 0.70), ``_h<NNN>`` for a height-only one (``paperback_book_h070``) and ``_<model>``
        for a model-specific fallback (``blender_eyedvd``)."""
        if self.kind == "group":
            parts = []
            for cat, n in self.members.items():
                row = next((r for r in icl.rows(subtask, self.role) if r["category"] == cat), None)
                token = cat
                if row is not None and (row.get("dataset") or "") == CUSTOM_DATASET and row.get("display_name"):
                    token = row["display_name"].strip().replace(" ", "_")
                parts.append(f"{token}_{n}")
            return "_".join(parts)
        if self.is_floor:
            return FLOOR_ALIAS_TOKEN
        token = self.category or ""
        if self.starts_inside:
            token += INSIDE_ALIAS_SUFFIX
        if with_model and self.models:
            token += f"_{self.models[0][1]}"
        if scale is not None:
            token += scale_token(scale, height_only)
        return token


def scale_token(scale: float, height_only: bool = False) -> str:
    """``0.7`` -> ``_s070``, height only ``_h070`` (two decimals, stays within the parser's alias alphabet)."""
    return f"_{'h' if height_only else 's'}{int(round(float(scale) * 100)):03d}"


@dataclass
class FallbackSpec:
    """What makes a combination a fallback of ``primary`` (customs ``fallback_of`` / ``fallback_rank`` / ``fallback_kind``)."""

    primary: str
    rank: int
    kind: str  # init_surface | scale | next_model
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Combination:
    subtask: str
    pick: str
    slot: str
    roles: Dict[str, RolePick]
    scenes: List[str] = field(default_factory=list)
    rooms: Dict[str, List[str]] = field(default_factory=dict)  # scene -> task rooms
    floor_room: Dict[str, str] = field(default_factory=dict)  # scene -> the floor's room
    notes: List[str] = field(default_factory=list)
    scale: Dict[str, float] = field(default_factory=dict)  # role -> uniform spawn scale (scale fallbacks)
    scale_axes: Dict[str, str] = field(default_factory=dict)  # role -> "height": ``scale`` applies to the height axis only
    model_tokens: Set[str] = field(default_factory=set)  # roles whose alias token carries the model (next-model fallbacks)
    fallback: Optional[FallbackSpec] = None
    selection: Dict[str, Dict[str, str]] = field(default_factory=dict)  # role -> the sheet row the combination was built from

    @property
    def label(self) -> str:
        base = f"{self.pick}/{self.slot}"
        return f"{base}/fallback{self.fallback.rank}:{self.fallback.kind}" if self.fallback else base

    def alias_values(self, icl: ICL, iddl: I.IDDL) -> Dict[str, str]:
        return {r: self.roles[r].alias_token(icl, self.subtask, scale=self.scale.get(r), with_model=r in self.model_tokens,
                                             height_only=self.scale_axes.get(r) == "height") for r in iddl.alias_slots}


class Skip(Exception):
    """A combination that is not grounded (reason + detail)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ----------------------------------------------------------------------------------------------------------------
# Grounder
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class Options:
    mode: str = "representative"
    slots: Tuple[str, ...] = ("representative",)
    scenes: Optional[Tuple[str, ...]] = None
    full_whitelists: bool = False
    distractor_pattern: Tuple[Tuple[str, int], ...] = (("same", 2), ("other", 1))
    legacy_activity_names: bool = False
    strict_symmetric_180: bool = False
    dry_run: bool = False
    representative_test: bool = True
    fallbacks: bool = True  # emit the fallback aliases (owner decision 2026-09-16 B)
    fallback_scale: float = DEFAULT_FALLBACK_SCALE  # scale fallback step when the pilot recorded no scale_hint
    scale_hints: Dict[str, float] = field(default_factory=dict)  # primary alias -> pilot template_meta.scale_hint
    scale_overrides: Dict[str, float] = field(default_factory=dict)  # primary alias -> scale (--scale-hint alias=s; persisted in the customs)
    reseed_generation: Tuple[str, ...] = ()  # generation keys re-seeded from the defaults ("*" = the whole block)
    #: pilot / owner blacklist: (subtask, role, category, model) -> reason (``--blacklist``); written as
    #: "pilot <date>: <reason>" into every scene block of the subtask's aliases and kept by later runs
    blacklist: Dict[Tuple[str, str, str, str], str] = field(default_factory=dict)


@dataclass
class GroundedProblem:
    """What one grounded alias produced (the object the D2 hook receives)."""

    alias: str
    bddl_alias: str
    subtask: str
    activity: str
    definition_id: int
    bddl_path: str
    bddl_text: str
    problem: Any  # ispatialgym.activity.problem.ISGProblem
    entry: CustomsEntry
    iddl: I.IDDL
    combination: Combination
    rendered: RenderedProblem


class Report:
    def __init__(self):
        self.problems: Dict[str, List[Tuple[str, int, str]]] = collections.defaultdict(list)  # subtask -> (alias, id, label)
        self.skipped: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)  # subtask -> (label, reason)
        self.blacklist: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)  # subtask -> reason -> n
        self.pending: Dict[str, int] = collections.Counter()
        self.notes: Dict[str, List[str]] = collections.defaultdict(list)
        self.tests: List[str] = []
        self.fallbacks: Dict[str, List[Tuple[str, str, int, str]]] = collections.defaultdict(list)  # subtask -> (alias, primary, rank, kind)

    def text(self) -> str:
        lines = [f"ground_isg {VERSION} report ({_dt.datetime.now().isoformat(timespec='seconds')})", ""]
        lines.append("| subtask | primaries | fallbacks | groundings | skipped | blacklist entries (reason: n) |")
        lines.append("|---|---|---|---|---|---|")
        for st in sorted(set(self.problems) | set(self.skipped) | set(self.blacklist)):
            hist = ", ".join(f"{k}: {v}" for k, v in sorted(self.blacklist[st].items()))
            fb_aliases = {a for a, _, _, _ in self.fallbacks[st]}
            n_alias = len({a for a, _, _ in self.problems[st]} - fb_aliases)
            lines.append(f"| {st} | {n_alias} | {len(fb_aliases)} | {len(self.problems[st])} | {len(self.skipped[st])} | {hist} |")
        all_fb = {a for v in self.fallbacks.values() for a, _, _, _ in v}
        total = len({a for v in self.problems.values() for a, _, _ in v})
        groundings = sum(len(v) for v in self.problems.values())
        lines.append("")
        lines.append(f"{total} problems written ({total - len(all_fb)} primaries + {len(all_fb)} fallbacks; {groundings} groundings), "
                     f"{sum(len(v) for v in self.skipped.values())} combinations skipped")
        if all_fb:
            lines.append("")
            lines.append("fallback aliases (rank, kind) <- primary:")
            for st in sorted(self.fallbacks):
                for alias, primary, rank, kind in self.fallbacks[st]:
                    lines.append(f"- {alias} ({rank}, {kind}) <- {primary}")
        if self.pending:
            lines.append("")
            lines.append("pending requirements (recorded as soft `blacklist` constraint records, models stay whitelisted):")
            for k, v in sorted(self.pending.items()):
                lines.append(f"- {k}: {v}")
        for st in sorted(self.skipped):
            if self.skipped[st]:
                lines.append("")
                lines.append(f"skipped in {st}:")
                for label, reason in self.skipped[st]:
                    lines.append(f"- {label}: {reason}")
        for st in sorted(self.notes):
            if self.notes[st]:
                lines.append("")
                lines.append(f"notes for {st}:")
                for n in sorted(set(self.notes[st])):
                    lines.append(f"- {n}")
        if self.tests:
            lines.append("")
            lines.append("representative tests:")
            lines.extend(f"- {t}" for t in self.tests)
        return "\n".join(lines) + "\n"

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "problems": {st: [{"alias": a, "id": i, "from": l} for a, i, l in v] for st, v in self.problems.items()},
            "skipped": {st: [{"combination": l, "reason": r} for l, r in v] for st, v in self.skipped.items()},
            "blacklist": {st: dict(c) for st, c in self.blacklist.items()},
            "pending": dict(self.pending),
            "notes": {st: sorted(set(v)) for st, v in self.notes.items()},
            "tests": list(self.tests),
            "fallbacks": {st: [{"alias": a, "fallback_of": p, "fallback_rank": r, "fallback_kind": k} for a, p, r, k in v] for st, v in self.fallbacks.items()},
        }


class Grounder:
    def __init__(
        self,
        iddls: Dict[str, I.IDDL],
        icl: ICL,
        customs: Customs,
        bddl_root: str = DEFAULT_BDDL_ROOT,
        options: Optional[Options] = None,
        iddl_root: Optional[str] = None,
    ):
        self.iddls = iddls
        self.icl = icl
        self.customs = customs
        self.bddl_root = bddl_root
        self.iddl_root = iddl_root or DEFAULT_IDDL
        self.opt = options or Options()
        self.adm = Admissibility(icl, self.opt.strict_symmetric_180, extra_blacklist=self.opt.blacklist)
        self.report = Report()
        self.today = _dt.date.today().isoformat()
        self.grounded: List[GroundedProblem] = []

    # -- helpers ------------------------------------------------------------------------------------------------------
    def activity_of(self, iddl: I.IDDL) -> str:
        if self.opt.legacy_activity_names:
            return LEGACY_ACTIVITY_NAMES.get(iddl.family, iddl.family)
        return iddl.family

    def _reason(self, code: str) -> str:
        return C.blacklist_reason(code, tool="ground_isg", date=self.today)

    @staticmethod
    def _synset(category: str) -> str:
        from ispatialgym.sampling.knowledge import synset_of_category

        return synset_of_category(category)

    def _marked(self, row: Dict[str, str]) -> bool:
        """Whether a candidates row belongs to the selection of the mode (test / all)."""
        if self.opt.mode == "test":
            return (row.get("test") or "") in _TRUE
        if self.opt.mode == "all":
            return not (row.get("status") or "").startswith("geometric_reject")
        return (row.get("representative") or "") in _TRUE

    # -- combinations: representative mode ----------------------------------------------------------------------------
    def combinations(self, subtask: str) -> List[Combination]:
        iddl = self.iddls[subtask]
        if self.opt.mode == "representative":
            return self._combinations_from_picks(iddl)
        return self._combinations_from_candidates(iddl)

    def _combinations_from_picks(self, iddl: I.IDDL) -> List[Combination]:
        subtask = iddl.subtask
        picks = sorted(p for p in self.icl.picks if p.split("#")[0] == subtask)
        out: List[Combination] = []
        if not picks:
            # no object pick (camera_relocalization-standard): one combination from the room rows
            if not iddl.roles or all(self.icl.kind_of(subtask, r) == "scene" for r in iddl.roles):
                try:
                    out.append(self._build(iddl, f"{subtask}#00", "representative", {}))
                except Skip as s:
                    self.report.skipped[subtask].append((f"{subtask}#00/representative", str(s)))
            return out
        for pick in picks:
            roles = self.icl.picks[pick]
            for slot in self.opt.slots:
                if slot != "representative" and not any(slot in slots for slots in roles.values()):
                    continue
                # one row per role from the slot (else the representative row); a role with several rows in the
                # slot (the A3 support triplets: several representative supports per pick) gives one combination
                # per row -- the support token differs per category, same-category supports share the alias and
                # union their scene blocks
                choices: Dict[str, List[Dict[str, str]]] = {}
                for role, slots in roles.items():
                    rows_ = slots.get(slot) or slots.get("representative") or []
                    if rows_:
                        choices[role] = list(rows_)
                multi = [r for r, rs in choices.items() if len(rs) > 1]
                variants: List[Dict[str, Dict[str, str]]] = [{}]
                for role, rs in choices.items():
                    variants = [dict(v, **{role: r}) for v in variants for r in (rs if role in multi else rs[:1])]
                for i, rows in enumerate(variants):
                    if len(variants) == 1:
                        label = pick
                    else:
                        # the A3 support triplets carry their index in the ``triplet`` column (T1 = the 2026-09-15 pick)
                        tags = [f"T{rows[r]['triplet']}" for r in multi if rows[r].get("triplet")]
                        label = f"{pick}~{'~'.join(tags) if len(tags) == len(multi) else i + 1}"
                    try:
                        out.append(self._build(iddl, label, slot, rows))
                    except Skip as s:
                        self.report.skipped[subtask].append((f"{label}/{slot}", str(s)))
        return out

    def _combinations_from_candidates(self, iddl: I.IDDL) -> List[Combination]:
        """``--test-only`` / ``--all``: category combinations from ``roles/candidates.csv`` marks, admitted by
        ``constraints_compatible.csv`` (never a cartesian product of unrelated lists)."""
        subtask = iddl.subtask
        out: List[Combination] = []
        main_role = next((r for r in TARGET_LIKE_ROLES if r in iddl.roles), None)
        support_cats: List[Optional[str]] = [None]
        if "support" in iddl.roles:
            support_cats = sorted({r["category"] for r in self.icl.scene_rows(subtask, "support") if self._marked(r)}) or []
        if main_role is None:
            try:
                out.append(self._build(iddl, f"{subtask}#00", self.opt.mode, {}))
            except Skip as s:
                self.report.skipped[subtask].append((f"{subtask}#00/{self.opt.mode}", str(s)))
            return out
        main_rows = [r for r in self.icl.rows(subtask, main_role) if self._marked(r)]
        by_cat: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for r in main_rows:
            by_cat[r["category"]].append(r)
        n = 0
        for cat in sorted(by_cat):
            rows = by_cat[cat]
            main_row = dict(rows[0])
            main_row["_pool"] = json.dumps([r["model"] for r in rows])  # every marked model of the category
            partner_choices: List[Optional[Dict[str, str]]] = [None]
            if "partner" in iddl.roles:
                partner_choices = []
                pkind = self.icl.kind_of(subtask, "partner")
                for pc in sorted({c["category_b"] for c in self.icl.partner_rows(subtask, cat)}):
                    prow = [r for r in self.icl.rows(subtask, "partner") if r["category"] == pc and self._marked(r)]
                    if not prow:
                        continue
                    p = dict(prow[0])
                    if pkind == "scene":
                        p["name"] = p["model"]
                    p["_pool"] = json.dumps([r["model"] for r in prow])
                    partner_choices.append(p)
            for partner in partner_choices:
                for scat in support_cats:
                    rows_sel: Dict[str, Dict[str, str]] = {main_role: main_row}
                    if partner is not None:
                        rows_sel["partner"] = partner
                    if scat is not None:
                        srow = next(r for r in self.icl.scene_rows(subtask, "support", category=scat) if self._marked(r))
                        s = dict(srow)
                        s["name"] = s["model"]
                        rows_sel["support"] = s
                    n += 1
                    label = f"{subtask}#{self.opt.mode}{n:03d}"
                    try:
                        out.append(self._build(iddl, label, self.opt.mode, rows_sel))
                    except Skip as s:
                        self.report.skipped[subtask].append((f"{label}/{self.opt.mode}", str(s)))
        return out

    # -- building one combination ------------------------------------------------------------------------------------
    def _build(self, iddl: I.IDDL, pick: str, slot: str, rows: Dict[str, Dict[str, str]]) -> Combination:
        subtask = iddl.subtask
        comb = Combination(subtask, pick, slot, {}, selection=dict(rows))
        for role in iddl.roles:
            kind = self.icl.kind_of(subtask, role)
            row = rows.get(role)
            if row is not None:
                row = self._resolve_row(subtask, role, row)
            if kind == "sampleable":
                comb.roles[role] = self._sampleable_pick(iddl, role, row, slot)
            elif kind == "group":
                comb.roles[role] = self._group_pick(iddl, role, row, comb)
            elif kind == "scene":
                comb.roles[role] = self._scene_pick(iddl, role, row, comb)
            else:
                raise Skip("unknown_kind", f"{role}: {kind}")
        self._pair_rules(iddl, comb)
        self._bind_scenes(iddl, comb)
        return comb

    def _pool_of(self, subtask: str, role: str, category: str) -> List[Dict[str, str]]:
        return [r for r in self.icl.rows(subtask, role) if r["category"] == category and r["kind"] != "scene"]

    def _resolve_row(self, subtask: str, role: str, row: Dict[str, str]) -> Dict[str, str]:
        """The ``roles/candidates.csv`` row behind a selection row (per-model facts: bbox, scale, symmetry, dataset,
        natural_scenes ...), with the selection's own fields (``name``, ``scene``, ``joint``, ``_pool``) on top."""
        cand: Optional[Dict[str, str]] = None
        if row.get("kind") == "scene":
            name = row.get("name") or row.get("model")
            cand = self.icl.row(subtask, role, row["category"], name)
            if cand is None:
                cand = next((r for r in self.icl.scene_rows(subtask, role, category=row["category"]) if _scene_object_key(r) == name), None)
        elif row.get("model"):
            cand = self.icl.row(subtask, role, row["category"], row["model"])
        merged: Dict[str, str] = dict(cand) if cand else {}
        for k, v in row.items():
            if v not in ("", None) or k not in merged:
                merged[k] = v
        if cand is None:
            merged["_synthetic"] = "1"
            if row.get("kind") != "scene" and not merged.get("dataset") and _is_custom_asset(row.get("category"), row.get("model")):
                merged["dataset"] = CUSTOM_DATASET
        return merged

    def _sampleable_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], slot: str) -> RolePick:
        subtask = iddl.subtask
        if row is None:
            # fixed sampleable roles without a pick (disembedding trays): the representative-marked rows
            marked = [r for r in self.icl.rows(subtask, role) if (r.get("representative") or "") in _TRUE] if self.opt.mode == "representative" \
                else [r for r in self.icl.rows(subtask, role) if self._marked(r)]
            if not marked:
                raise Skip("no_candidate", f"role {role!r} has no {self.opt.mode} row")
            cats = sorted({r["category"] for r in marked})
            if len(cats) > 1:
                raise Skip("ambiguous_category", f"role {role!r}: {cats}")
            rp = RolePick(role, "sampleable", cats[0], [(r["category"], r["model"]) for r in marked])
            rp.pick_row = marked[0]
            return rp
        category, model = row["category"], row["model"]
        models = [(category, model)]
        if row.get("_pool"):
            models = [(category, m) for m in json.loads(row["_pool"])]
        rp = RolePick(role, "sampleable", category, models, pick_row=row, joint=row.get("joint") or None)
        rp.allow_heldout = slot == "heldout"
        return rp

    def _group_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], comb: Combination) -> RolePick:
        subtask = iddl.subtask
        gen = G.defaults_for(subtask)
        seed = (gen.get("groups") or {}).get(role)
        if seed is None:
            raise Skip("no_group_seed", f"generation_defaults has no groups.{role} for {subtask}")
        size = seed["size"]
        rp = RolePick(role, "group")
        if role == "distractors":
            target = comb.roles.get("target")
            if target is None or not target.models:
                raise Skip("no_target", "distractors need the target first")
            tcat, tmodel = target.models[0]
            sheet = self._sheet_distractor_rows(comb)
            if sheet.get(SHEET_LAYOUT_SLOT):
                self._layout_members(rp, sheet[SHEET_LAYOUT_SLOT], tcat, tmodel)
            elif sheet:
                self._sheet_pool_members(rp, sheet, tcat, tmodel, size, comb)
            else:
                self._distractor_members(rp, tcat, tmodel, size)
            return rp
        if row is None:
            marked = [r for r in self.icl.rows(subtask, role) if self._marked(r)]
            if not marked:
                raise Skip("no_candidate", f"group role {role!r} has no {self.opt.mode} row")
            row = marked[0]
        if isinstance(size, list):
            raise Skip("group_size_range", f"{role}: a range size {size} needs a member pattern (only distractors have one)")
        category, model = row["category"], row["model"]
        rp.members = {category: int(size)}
        rp.pick_row = row
        models = [(category, model)]
        if row.get("_pool"):
            models = [(category, m) for m in json.loads(row["_pool"])]
        rp.models = models
        rp.pools = {category: [m for _, m in models]}
        rp.category = category
        return rp

    def _distractor_members(self, rp: RolePick, tcat: str, tmodel: str, size) -> None:
        """§3.12 rule 1: same-category pool minus the target and its look-alikes, other-category pools (tier order);
        the ``--distractors`` pattern fixes the member counts of the alias."""
        excluded = {(tcat, tmodel)} | self.icl.lookalikes(tcat, tmodel) | self.icl.heldout_models
        # pilot rejections of distractor models (``--blacklist disembedding:distractors:<cat>/<model>=...``) leave the pools too
        # (2026-09-25): applied only per scene below, a category whose every model is rejected kept its member slot in the
        # alias with an empty whitelist (bottle_of_apple_juice/xvrbdy, the two-body asset)
        excluded |= {(c, m) for (st, role, c, m) in getattr(self.adm, "extra_blacklist", {}) if st == "disembedding" and role == "distractors"}
        same = [(r["category_b"], r["model_b"]) for r in self.icl.distractor_rows(tcat, "distractor_same_category")]
        same = [cm for cm in same if cm not in excluded and _is_ready(self.icl.row("disembedding", "distractors", *cm) or {"kind": "group", "model": cm[1]})]
        other_rows = sorted(self.icl.distractor_rows(tcat, "distractor_other_category"), key=lambda r: (int(r.get("tier") or 9), r["category_b"], r["model_b"]))
        other_by_cat: Dict[str, List[str]] = collections.OrderedDict()
        for r in other_rows:
            cm = (r["category_b"], r["model_b"])
            if cm in excluded:
                continue
            other_by_cat.setdefault(r["category_b"], [])
            if cm[1] not in other_by_cat[r["category_b"]]:
                other_by_cat[r["category_b"]].append(cm[1])
        total = size[1] if isinstance(size, list) else int(size)
        members: Dict[str, int] = collections.OrderedDict()
        pools: Dict[str, List[str]] = {}
        remaining = total
        for kind, n in self.opt.distractor_pattern:
            n = min(n, remaining)
            if n <= 0:
                continue
            if kind == "same":
                if len(same) >= n:
                    members[tcat] = members.get(tcat, 0) + n
                    pools[tcat] = sorted({m for _, m in same})
                    remaining -= n
            elif kind == "other":
                # the first other categories (tier order) with a pool of >= 1 model, one member each
                for cat, ms in other_by_cat.items():
                    if remaining <= 0 or n <= 0:
                        break
                    if cat == tcat or cat in members or not ms:
                        continue
                    members[cat] = 1
                    pools[cat] = sorted(ms)
                    remaining -= 1
                    n -= 1
        if remaining > 0:
            # fill from whatever pool is left (same category first, then other categories)
            if same and remaining > 0 and len(same) >= members.get(tcat, 0) + remaining:
                members[tcat] = members.get(tcat, 0) + remaining
                pools[tcat] = sorted({m for _, m in same})
                remaining = 0
            for cat, ms in other_by_cat.items():
                if remaining <= 0:
                    break
                if cat == tcat or cat in members or not ms:
                    continue
                members[cat] = 1
                pools[cat] = sorted(ms)
                remaining -= 1
        lo = size[0] if isinstance(size, list) else int(size)
        if sum(members.values()) < lo:
            raise Skip("empty_distractor_pool", f"target {tcat}/{tmodel}: pools too small after the look-alike / held-out exclusions")
        # owner decision 2026-09-16 4a: the §6.4 tray budget is applied here -- the grown footprints of the member set must
        # fit ``tray_budget`` (generation defaults) of the source tray's inner floor; drop members (last pattern slot first) until they do
        rp.budget = self._distractor_budget(tcat, tmodel, members, pools, lo)
        rp.members = members
        rp.pools = pools
        rp.models = [(c, m) for c, ms in pools.items() for m in ms]

    def _distractor_budget(self, tcat: str, tmodel: str, members: Dict[str, int], pools: Dict[str, List[str]], lo: int) -> Dict[str, Any]:
        """§6.4 / owner decision 2026-09-16 4a, footprint rule and fraction revised 2026-09-24. Footprint of a distractor = the
        grown rectangle ``(max(bbox_x, bbox_y) + 2c) · (min(bbox_x, bbox_y) + 2c)`` (icl dims; ``c`` = the pad clearance of the
        target's tightest tabletop grasp) - the yaw-invariant area the sampler's ``place_distractors`` budgets with the same
        growth; per category the largest model of the pool counts (any draw fits). The set must cover ≤ ``tray_budget`` of
        the source tray's inner floor (``source_tray_inner_floor_m2`` at scale 1 × ``source_tray_scale``²); otherwise
        members are dropped from the end of the pattern (other-category first, then the second same-category one) down to
        ``lo``. Until 2026-09-24 the rule was the conservative square ``(long + 2c)²`` (any yaw's axis-aligned box) at
        0.45: it charged a 22 cm bottle 0.0725 m² for a 0.016 m² footprint and left the milk tray at one distractor with
        86 % of the floor empty."""
        tray_key, tray_scale, floor_m2, fraction = self._source_tray_floor()
        clearance = self._target_pad_clearance(tcat, tmodel)

        def footprint(cat: str) -> Tuple[float, str]:
            """Largest grown rectangle ``(long + 2c) · (short + 2c)`` of the category's pool and the model it belongs to."""
            best, best_m = 0.0, ""
            for m in pools.get(cat, []):
                row = self.icl.row("disembedding", "distractors", cat, m) or {}
                try:
                    x, y = float(row.get("bbox_x") or 0.0), float(row.get("bbox_y") or 0.0)
                except ValueError:
                    x, y = 0.0, 0.0
                long_, short = max(x, y), min(x, y)
                area = (long_ + 2.0 * clearance) * (short + 2.0 * clearance)
                if area > best:
                    best, best_m = area, m
            return best, best_m

        budget: Dict[str, Any] = {"tray": tray_key, "tray_scale": tray_scale, "tray_floor_m2": round(floor_m2, 4), "budget_fraction": fraction,
                                  "pad_clearance_m": round(clearance, 4), "footprint_rule": "rectangle", "dropped": []}
        if floor_m2 > 0:
            areas = {cat: footprint(cat) for cat in members}
            used = lambda: sum(areas[c][0] * n for c, n in members.items())  # noqa: E731
            while used() > fraction * floor_m2 and sum(members.values()) > max(lo, 1):
                cat = list(members)[-1]  # last pattern slot first (other-category, then the second same-category member)
                members[cat] -= 1
                budget["dropped"].append(cat)
                if members[cat] == 0:
                    del members[cat]
            budget.update({"footprints_m2": {c: round(a, 4) for c, (a, _) in areas.items() if c in members},
                           "largest_models": {c: m for c, (_, m) in areas.items() if c in members}})
            if used() > fraction * floor_m2:
                raise Skip("distractor_budget", f"target {tcat}/{tmodel}: even one distractor ({list(members)}) exceeds {fraction:.0%} of the "
                                                f"{floor_m2:.3f} m2 source tray floor ({used():.3f} m2, grown rectangle)")
            for cat in list(pools):
                if cat not in members:
                    del pools[cat]
            budget["used_m2"] = round(used(), 4)
        else:
            budget["used_m2"] = None
        budget["n_distractors"] = sum(members.values())
        return budget

    def _source_tray_floor(self) -> Tuple[Optional[str], float, float, float]:
        """``(tray, tray_scale, floor_m2, fraction)`` of the §6.4 budget: the representative source tray, its inner floor
        (``source_tray_inner_floor_m2`` at scale 1 × ``source_tray_scale``²) and ``tray_budget`` (generation defaults)."""
        gen = G.defaults_for("disembedding")
        tray_scale = float(gen.get("source_tray_scale") or 1.0)
        floors = gen.get("source_tray_inner_floor_m2") or {}
        source_rows = [r for r in self.icl.rows("disembedding", "source") if (r.get("representative") or "") in _TRUE] or self.icl.rows("disembedding", "source")
        tray_key = f"{source_rows[0]['category']}/{source_rows[0]['model']}" if source_rows else None
        floor_native = float(floors.get(tray_key, 0.0)) if tray_key else 0.0
        return tray_key, tray_scale, floor_native * tray_scale * tray_scale, float(gen.get("tray_budget") or 0.60)

    def _target_pad_clearance(self, tcat: str, tmodel: str) -> float:
        """Pad clearance of the target's tightest admissible tabletop grasp (conventions.pad_clearance; icl width_m)."""
        widths = [float(g["width_m"]) for g in self.icl.admissible_grasps(tcat, tmodel, "tabletop") if g.get("width_m")]
        return _pad_clearance(min(widths)) if widths else 0.0

    def _sheet_distractor_rows(self, comb: Combination) -> Dict[str, List[Dict[str, str]]]:
        """The pick's distractor rows of the representative sheet by slot (``same_category`` / ``other_category`` /
        ``fixed_layout``); empty for a pick without them."""
        slots = self.icl.picks.get(comb.pick.split("~")[0], {}).get("distractors", {})
        return {s: list(slots[s]) for s in SHEET_POOL_SLOTS + (SHEET_LAYOUT_SLOT,) if slots.get(s)}

    def _layout_members(self, rp: RolePick, rows: List[Dict[str, str]], tcat: str, tmodel: str) -> None:
        """The puzzle pick (handoff 2026-10-10, disembedding#01): its ``fixed_layout`` parts are always all placed, together with
        the target, as one layout at the x / y the layout file of their notes gives (``generation.fixed_layout``; the sampler
        reads the file). Not drawn from a pool and outside the distractor budget."""
        files = sorted({m.group(1) for m in (LAYOUT_NOTE_RE.search(r.get("note") or "") for r in rows) if m})
        if len(files) != 1:
            raise Skip("fixed_layout", f"target {tcat}/{tmodel}: the fixed_layout rows name {files or 'no'} layout file, need one")
        path = os.path.join(REPO_ROOT, files[0])
        if not os.path.isfile(path):
            raise Skip("fixed_layout", f"target {tcat}/{tmodel}: layout file {files[0]} is not in this checkout")
        with open(path, "rb") as f:
            raw = f.read()
        parts = json.loads(raw.decode("utf-8")).get("parts") or {}
        missing = [m for m in [tmodel] + [r["model"] for r in rows] if m not in parts]
        if missing:
            raise Skip("fixed_layout", f"{files[0]} has no part {missing}")
        if (parts[tmodel].get("role") or "target") != "target":
            raise Skip("fixed_layout", f"{files[0]}: {tmodel} is the {parts[tmodel].get('role')}, not the target")
        resolved = [self._resolve_row("disembedding", "distractors", r) for r in rows]
        members: Dict[str, int] = collections.OrderedDict()
        pools: Dict[str, List[str]] = collections.OrderedDict()
        for r in resolved:
            members[r["category"]] = members.get(r["category"], 0) + 1
            pools.setdefault(r["category"], []).append(r["model"])
        rp.members, rp.pools = members, pools
        rp.models = [(c, m) for c, ms in pools.items() for m in ms]
        rp.pick_row = resolved[0]
        rp.layout = {"file": files[0], "sha256": hashlib.sha256(raw).hexdigest(), "target": f"{tcat}/{tmodel}", "parts": [f"{c}/{m}" for c, m in rp.models]}
        rp.budget = {"fixed_layout": files[0], "n_distractors": len(rp.models), "rule": "not drawn from a pool, outside the distractor budget (owner 2026-10-10)"}

    def _sheet_pool_members(self, rp: RolePick, sheet: Dict[str, List[Dict[str, str]]], tcat: str, tmodel: str, size, comb: Combination) -> None:
        """Handoff 2026-10-10 (owner 2026-10-11): the pick's ``same_category`` / ``other_category`` rows are its distractor pools,
        minus the target, its look-alikes and the held-out models. A draw takes at most ``same`` (2) models of the target's
        category, at most ``other`` (1) of each other category and at most one per look-alike group (:func:`admissible_draw`; the
        ``--distractors`` pattern gives the caps). The member count is the largest size whose every admissible draw fits the §6.4
        budget (grown rectangles in the pick's pose, ``tray_budget`` of the source tray floor; :func:`count_every_draw`). The member
        set of the alias (categories and counts: part of the alias and of the BDDL) is one of those draws, fixed per (pick, target,
        support category) by :func:`stable_index`; the sampler draws the models of each category per template. Pool notes with
        ``pose: lying`` / ``pose: standing`` give ``generation.distractor_pose``."""
        excluded = {(tcat, tmodel)} | self.icl.lookalikes(tcat, tmodel) | self.icl.heldout_models
        excluded |= {(c, m) for (st, role, c, m) in getattr(self.adm, "extra_blacklist", {}) if st == "disembedding" and role == "distractors"}
        pool: List[Tuple[str, str]] = []
        poses: Dict[Tuple[str, str], Optional[str]] = {}
        for slot in SHEET_POOL_SLOTS:
            for r in sheet.get(slot, []):
                cm = (r["category"], r["model"])
                if cm in excluded or cm in poses or not _is_ready(self.icl.row("disembedding", "distractors", *cm) or {"kind": "group", "model": cm[1]}):
                    continue
                m = POSE_NOTE_RE.search(r.get("note") or "")
                poses[cm] = m.group(1) if m else None
                pool.append(cm)
        if not pool:
            raise Skip("empty_distractor_pool", f"target {tcat}/{tmodel}: the sheet pools are empty after the look-alike / held-out exclusions")
        tray_key, tray_scale, floor_m2, fraction = self._source_tray_floor()
        clearance = self._target_pad_clearance(tcat, tmodel)
        area: Dict[Tuple[str, str], float] = {}
        for cm in pool:
            row = self.icl.row("disembedding", "distractors", *cm) or {}
            try:
                bbox = [float(row.get(k) or "") for k in ("bbox_x", "bbox_y", "bbox_z")]
            except ValueError:
                raise Skip("distractor_budget", f"{cm[0]}/{cm[1]}: no icl bbox for its footprint") from None
            long_, short = footprint_dims(bbox, poses[cm])
            area[cm] = (long_ + 2.0 * clearance) * (short + 2.0 * clearance)
        caps = dict(self.opt.distractor_pattern)
        same_max, other_max = int(caps.get("same", 2)), int(caps.get("other", 1))
        group_of = {m: g for g, ms in self.icl.lookalike_groups.items() for m in ms}
        lo = size[0] if isinstance(size, list) else int(size)
        hi = size[1] if isinstance(size, list) else int(size)
        budget_m2 = fraction * floor_m2 if floor_m2 > 0 else float("inf")
        n, worst, draws = count_every_draw(pool, area, budget_m2, hi, lambda d: admissible_draw(d, tcat, same_max, other_max, group_of))
        if n < max(lo, 1):
            raise Skip("distractor_budget", f"target {tcat}/{tmodel}: not every draw of {max(lo, 1)} sheet distractor(s) fits {fraction:.0%} of the "
                                            f"{floor_m2:.3f} m2 source tray floor (grown rectangle)")
        support = (comb.selection.get("support") or {}).get("category") or ""
        draw = draws[stable_index(f"{comb.pick.split('~')[0]}|{tcat}/{tmodel}|{support}", len(draws))]
        order = [c for c, _ in pool]
        cats = sorted({c for c, _ in draw}, key=lambda c: (c != tcat, order.index(c)))
        rp.members = collections.OrderedDict((c, sum(1 for d, _ in draw if d == c)) for c in cats)
        rp.pools = collections.OrderedDict((c, [m for pc, m in pool if pc == c]) for c in cats)
        rp.models = [(c, m) for c, ms in rp.pools.items() for m in ms]
        posed = [cm for cm in rp.models if poses.get(cm)]
        if posed:
            labels = {poses[cm] for cm in posed}
            rp.pose = labels.pop() if len(labels) == 1 and len(posed) == len(rp.models) else {f"{c}/{m}": poses[(c, m)] for c, m in posed}
        rp.budget = {"tray": tray_key, "tray_scale": tray_scale, "tray_floor_m2": round(floor_m2, 4), "budget_fraction": fraction,
                     "pad_clearance_m": round(clearance, 4), "footprint_rule": "rectangle",
                     "pose": rp.pose if isinstance(rp.pose, str) else ("per_model" if rp.pose else "canonical"),
                     "source": "representatives.csv same_category / other_category rows",
                     "draw_rule": {"size_max": hi, "same_category_max": same_max, "other_category_max": other_max, "lookalike_group_max": 1},
                     "count_rule": "every admissible draw fits", "model_footprints_m2": {f"{c}/{m}": round(a, 4) for (c, m), a in area.items()},
                     "n_draws": len(draws), "worst_draw_m2": round(worst, 4), "used_m2": round(worst, 4), "n_distractors": n,
                     "draw": [f"{c}/{m}" for c, m in draw], "dropped": []}

    def _scene_pick(self, iddl: I.IDDL, role: str, row: Optional[Dict[str, str]], comb: Combination) -> RolePick:
        subtask = iddl.subtask
        rp = RolePick(role, "scene")
        if row is None:
            raise Skip("no_scene_pick", f"scene role {role!r} has no {self.opt.mode} row")
        rp.allow_heldout = comb.slot == "heldout"
        category = row["category"]
        name = row.get("name") or row.get("model")
        scene = row.get("scene")
        if category in FLOOR_MODEL_NAMES or name in FLOOR_MODEL_NAMES:
            # the floor of the scene: the row is the room anchor (the floor rows of the icl carry no room; the
            # partner / target of the combination decides the room)
            rp.category = FLOOR_CATEGORY
            rp.same_as = "floor"
            rp.pick_row = row
            rp.scene_objects = {scene: [row]} if scene else {}
            return rp
        rp.category = category
        rp.pick_row = row
        rp.joint = row.get("joint") or None
        # the same scene object as another role of the combination (under large-scale: partner's own top)
        for other, orp in comb.roles.items():
            if orp.kind == "scene" and other != role and orp.pick_row is not None and orp.pick_row.get("scene") == scene \
                    and (orp.pick_row.get("name") or orp.pick_row.get("model")) == name:
                rp.same_as = other
        # small-scale articulation: targets taller than 0.6 m stand on the floor (§5.4)
        if role == "support" and subtask.startswith("configuration_matching-articulation") and "small_scale" in subtask:
            target = comb.roles.get("target")
            trow = target.pick_row if target else None
            if trow is not None and trow.get("bbox_z"):
                try:
                    if float(trow["bbox_z"]) > FLOOR_SUPPORT_HEIGHT_M:
                        comb.notes.append(f"{target.category}: taller than {FLOOR_SUPPORT_HEIGHT_M} m -> support = floor of the picked "
                                          f"support's room (§5.4)")
                        rp.category = FLOOR_CATEGORY
                        rp.same_as = "floor"
                        rp.scene_objects = {scene: [row]} if scene else {}  # the picked support anchors the room
                        return rp
                except ValueError:
                    pass
        # candidate scene objects per scene: the pick's object + the marked rows of the category. Wall regions are
        # keyed by their region id (``<room>/<wall>/<rNN>``, the pick's ``name``), other scene objects by name.
        objects: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
        for r in self.icl.scene_rows(subtask, role, category=category):
            is_pick = r["scene"] == scene and _scene_object_key(r) == name
            if is_pick or self._marked(r) or (self.opt.mode == "representative" and (r.get("representative") or "") in _TRUE):
                if (r.get("status") or "").startswith("geometric_reject") and not is_pick:
                    continue
                if (subtask, role, r["scene"], _scene_object_key(r)) in self.icl.heldout_scene and not (is_pick and rp.allow_heldout):
                    continue  # held-out rule: the role's held-out scene object is never bound in another combination
                if not any(_scene_object_key(o) == _scene_object_key(r) for o in objects[r["scene"]]):
                    objects[r["scene"]].append(r)
        if scene and not any(_scene_object_key(o) == name for o in objects.get(scene, [])):
            # the pick names an object the candidates sheet does not list: keep it (the selection is explicit) and say so
            fake = {k: v for k, v in row.items() if k in ("subtask", "role", "category", "scene", "room", "joint", "handle_key", "note", "status")}
            fake.update({"kind": "scene", "model": name, "representative": "1", "_synthetic": "1"})
            # the representative sheet has no room column: its support rows give the rooms in the note ("rooms=a,b")
            m = None if fake.get("room") else re.search(r"\brooms=([^\s;]+)", row.get("note") or "")
            if m:
                fake["room"] = m.group(1).strip(",")
            objects[scene].append(fake)
            comb.notes.append(f"{role}: {name}@{scene} is not a roles/candidates.csv row of {subtask} (bound from the selection only)")
        rp.scene_objects = dict(objects)
        if role == "support" and rp.same_as is None:
            rp.surface = support_surface(row, comb.pick)
        return rp

    def _pair_rules(self, iddl: I.IDDL, comb: Combination) -> None:
        """§3.12 rule 2 (look-alike target–partner pair), the compat sheet for target–partner pairs (attachment ready)."""
        subtask = iddl.subtask
        target, partner = comb.roles.get("target"), comb.roles.get("partner")
        if target is None or partner is None:
            return
        tcat = target.category
        if partner.kind == "sampleable":
            pcat = partner.category
            rows = self.icl.partner_rows(subtask, tcat)
            if not any(r["category_b"] == pcat for r in rows):
                raise Skip("not_compatible", f"{tcat} -> {pcat} is not a partner_of pair of {subtask}")
            if subtask == "configuration_matching-attachment":
                ok_models = []
                for tm in [m for _, m in target.models]:
                    for pm in [m for _, m in partner.models]:
                        r = next((c for c in rows if c["model_a"] == tm and c["model_b"] == pm), None)
                        if r is None:
                            continue
                        if (r.get("ready") or "1") not in _TRUE:
                            comb.notes.append(f"attachment pair {tm} -> {pm}: partner KB entry not ready")
                            continue
                        ok_models.append((tm, pm))
                if not ok_models:
                    raise Skip("not_compatible", f"no ready attachment pair between {tcat} {[m for _, m in target.models]} and {pcat}")
            for tm in [m for _, m in target.models]:
                for pm in [m for _, m in partner.models]:
                    if self.icl.are_lookalikes((tcat, tm), (pcat, pm)):
                        if len(target.models) == 1 and len(partner.models) == 1:
                            raise Skip("lookalike_pair", f"{tcat}/{tm} and {pcat}/{pm} share a look-alike group (§3.12 rule 2)")
                        comb.notes.append(f"look-alike pair {tcat}/{tm} – {pcat}/{pm} excluded by the sampler (§3.12 rule 2)")
        else:
            pcat = partner.category
            rows = self.icl.partner_rows(subtask, tcat)
            if rows and not any(r["category_b"] == pcat for r in rows):
                raise Skip("not_compatible", f"{tcat} -> scene partner {pcat} is not a partner_of pair of {subtask}")
            # §6.12 under-large partner admissibility from the B4 note: clearance >= target height + 0.17
            if subtask.endswith("under_large_scale") and target.pick_row is not None:
                extra = G.defaults_for(subtask)["under_clearance_extra_m"]
                try:
                    h = float(target.pick_row.get("bbox_z") or 0.0)
                except ValueError:
                    h = 0.0
                for scene, objs in list(partner.scene_objects.items()):
                    keep = []
                    for o in objs:
                        m = re.search(r"under_clearance_max=([0-9.]+)", o.get("note") or "")
                        if m and h and float(m.group(1)) < h + extra:
                            comb.notes.append(f"{o['model']}@{scene}: under clearance {m.group(1)} < {h:.2f} + {extra} (under_clearance)")
                            continue
                        keep.append(o)
                    partner.scene_objects[scene] = keep
                if not any(partner.scene_objects.values()):
                    raise Skip("under_clearance", f"no partner with enough under clearance for {tcat} (height {h:.2f} m)")

    def _bind_scenes(self, iddl: I.IDDL, comb: Combination) -> None:
        """Which scenes get a block and which rooms: room rows bound the scene; otherwise every scene where each
        scene role has at least one candidate object."""
        subtask = iddl.subtask
        room_rows = self.icl.room_rows(subtask)
        scene_roles = [rp for rp in comb.roles.values() if rp.kind == "scene" and not rp.is_floor and not rp.same_as]
        if room_rows:
            for r in room_rows:
                sc = r["scene"]
                if self.opt.scenes and sc not in self.opt.scenes:
                    continue
                comb.scenes.append(sc)
                comb.rooms[sc] = _split_rooms(r.get("room")) or [r["model"]]
                comb.floor_room[sc] = comb.rooms[sc][0]
        else:
            candidates = set(C.SCENE_CANDIDATES)
            for rp in scene_roles:
                candidates &= {s for s, objs in rp.scene_objects.items() if objs}
            for rp in comb.roles.values():
                if rp.is_floor and rp.scene_objects and not scene_roles:
                    candidates &= set(rp.scene_objects)
            if self.opt.scenes:
                candidates &= set(self.opt.scenes)
            for sc in sorted(candidates, key=lambda s: C.SCENE_CANDIDATES.index(s) if s in C.SCENE_CANDIDATES else 99):
                rooms: List[str] = []
                primary: Optional[str] = None
                # the primary room (the floor's) comes from the partner (large-scale IORS), the scene target
                # (large-scale articulation) or the support (tabletop); a floor-bound support contributes its anchor row
                for rname in ("partner", "target", "support"):
                    rp = comb.roles.get(rname)
                    if rp is None or rp.kind != "scene" or (rp.same_as and not rp.is_floor):
                        continue
                    for o in rp.scene_objects.get(sc, []):
                        for room in _split_rooms(o.get("room")):
                            if room not in rooms:
                                rooms.append(room)
                            if primary is None:
                                primary = room
                comb.scenes.append(sc)
                comb.rooms[sc] = rooms
                if primary is not None:
                    comb.floor_room[sc] = primary
        if not comb.scenes:
            missing = [rp.role for rp in scene_roles if not any(rp.scene_objects.values())]
            raise Skip("no_scene", f"no candidate scene has bindings for every scene role (empty: {missing})")

    # -- grounding ----------------------------------------------------------------------------------------------------
    def ground(self, comb: Combination) -> GroundedProblem:
        iddl = self.iddls[comb.subtask]
        subtask = iddl.subtask
        activity = self.activity_of(iddl)
        # alias
        alias = iddl.fill_alias(comb.alias_values(self.icl, iddl))
        # BDDL groundings
        groundings: Dict[str, RoleGrounding] = {}
        for role in iddl.roles:
            rp = comb.roles[role]
            if rp.kind == "group":
                members = []
                for cat, n in rp.members.items():
                    members.extend([(cat, self._synset(cat))] * n)
                dataset = None
                if rp.pick_row is not None and rp.pick_row.get("dataset"):
                    dataset = rp.pick_row["dataset"]
                groundings[role] = RoleGrounding(role, "group", rp.category, members=members, dataset=dataset)
            elif rp.kind == "sampleable":
                groundings[role] = RoleGrounding(role, "sampleable", rp.category, self._synset(rp.category))
            elif rp.same_as:
                groundings[role] = RoleGrounding(role, "scene", rp.category, same_as=rp.same_as)
            else:
                groundings[role] = RoleGrounding(role, "scene", rp.category, self._synset(rp.category))
        # no date in the file header: a re-grounding rewrites identical text (the registry keeps the grounding date)
        comment = f"{GROUNDER}: {alias} ({I.relative_path(iddl.path, self.iddl_root) if iddl.path else iddl.subtask}; {comb.label})"
        if comb.fallback is not None:
            comment += f"\nfallback {comb.fallback.rank} ({comb.fallback.kind}) of {comb.fallback.primary}"
            if comb.fallback.params:
                comment += ": " + ", ".join(f"{k} = {v}" for k, v in comb.fallback.params.items())
        if iddl.comment:
            comment += "\n" + iddl.comment
        meta: Dict[str, Any] = {"subtask": subtask, "iddl": I.relative_path(iddl.path, self.iddl_root) if iddl.path else None,
                                "pick": comb.pick, "slot": comb.slot, "mode": self.opt.mode,
                                "fallback_of": None, "fallback_rank": None, "fallback_kind": None}
        if comb.fallback is not None:
            meta.update({"fallback_of": comb.fallback.primary, "fallback_rank": comb.fallback.rank, "fallback_kind": comb.fallback.kind})
        init_predicates = {("ontop", r): "inside" for r, rp in comb.roles.items() if rp.starts_inside}
        rendered = write_problem(
            self.bddl_root,
            activity,
            iddl,
            groundings,
            alias,
            comment=comment,
            meta=meta,
            dry_run=self.opt.dry_run,
            init_predicates=init_predicates or None,
        )
        entry = self._customs_entry(iddl, comb, alias, rendered.definition_id)
        gp = GroundedProblem(
            alias=alias,
            bddl_alias=rendered.alias,
            subtask=subtask,
            activity=activity,
            definition_id=rendered.definition_id,
            bddl_path=os.path.join(self.bddl_root, activity, f"problem{rendered.definition_id}.bddl"),
            bddl_text=rendered.text,
            problem=rendered.problem,
            entry=entry,
            iddl=iddl,
            combination=comb,
            rendered=rendered,
        )
        self.grounded.append(gp)
        self.report.problems[subtask].append((alias, rendered.definition_id, comb.label))
        if comb.fallback is not None:
            self.report.fallbacks[subtask].append((alias, comb.fallback.primary, comb.fallback.rank, comb.fallback.kind))
        for n in comb.notes:
            self.report.notes[subtask].append(n)
        return gp

    # -- fallback aliases (owner decision 2026-09-16 B) ------------------------------------------------------------------
    def fallbacks_for(self, comb: Combination, alias: str, entry: Optional[CustomsEntry] = None) -> List[Combination]:
        """The fallback combinations of a grounded primary, in rank order (empty for a fallback combination itself).

        A primary whose ``target`` whitelist is empty in every scene (the pilot blacklisted the pick, e.g. the hangers'
        ``unstable_init``) gets the next admissible model of the category as its rank-1 fallback instead of the
        model-bound fallbacks (a scaled alias of a blacklisted model would be as empty)."""
        if comb.fallback is not None or not self.opt.fallbacks:
            return []
        subtask = comb.subtask
        if entry is not None and self._target_blacklisted(entry):
            target = comb.roles.get("target")
            if target is not None and target.kind == "sampleable":
                return self._next_model_fallbacks(comb, alias)
        if subtask in INIT_SURFACE_FALLBACK_SUBTASKS:
            return self._init_surface_fallbacks(comb, alias)
        if subtask in SCALE_FALLBACK_SUBTASKS:
            return self._scale_fallbacks(comb, alias)
        if subtask in NEXT_MODEL_FALLBACK_SUBTASKS:
            return self._next_model_fallbacks(comb, alias)
        return []

    @staticmethod
    def _target_blacklisted(entry: CustomsEntry) -> bool:
        """The sampleable ``target`` of the entry has no whitelisted model in any scene block (all blacklisted)."""
        t = entry.roles.get("target")
        if t is None or not t.sampleable or not entry.scenes:
            return False
        return all(not any(sc.whitelist.get("target", {}).values()) for sc in entry.scenes.values())

    @staticmethod
    def _derive(comb: Combination, primary: str, rank: int, kind: str, params: Dict[str, Any]) -> Combination:
        """A copy of ``comb`` (roles shared unless replaced by the caller) marked as fallback ``rank`` of ``primary``."""
        return Combination(
            subtask=comb.subtask,
            pick=comb.pick,
            slot=comb.slot,
            roles=dict(comb.roles),
            scale=dict(comb.scale),
            scale_axes=dict(comb.scale_axes),
            model_tokens=set(comb.model_tokens),
            fallback=FallbackSpec(primary, rank, kind, dict(params)),
            selection=dict(comb.selection),
        )

    def _init_surface_fallbacks(self, comb: Combination, alias: str) -> List[Combination]:
        """§5.6 init-surface classes after the grounded one: under support -> partner's own top -> floor; ontop
        support -> floor. The support role is re-bound; everything else (target, partner, scenes) stays."""
        support, partner = comb.roles.get("support"), comb.roles.get("partner")
        if support is None or partner is None or partner.kind != "scene":
            return []
        chain = list(INIT_SURFACE_FALLBACK_SUBTASKS[comb.subtask])
        current = "floor" if support.is_floor else ("partner" if support.same_as == "partner" else "support")
        later = chain[chain.index(current) + 1:] if current in chain else chain
        out: List[Combination] = []
        for rank, cls in enumerate(later, start=1):
            if cls == "partner":
                rp = RolePick("support", "scene", partner.category, same_as="partner", scene_objects=dict(partner.scene_objects), pick_row=partner.pick_row)
            else:
                rp = RolePick("support", "scene", FLOOR_CATEGORY, same_as="floor", scene_objects={sc: [] for sc in partner.scene_objects}, pick_row=partner.pick_row)
            fb = self._derive(comb, alias, rank, "init_surface", {"init_surface": cls})
            fb.roles["support"] = rp
            fb.notes = list(comb.notes)
            try:
                self._bind_scenes(self.iddls[comb.subtask], fb)
            except Skip as s:
                self.report.skipped[comb.subtask].append((fb.label, str(s)))
                continue
            out.append(fb)
        return out

    def scale_for(self, alias: str, height_only: bool = False) -> Tuple[float, str]:
        """The scale of the scale fallback of ``alias`` and its source, in priority order: a ``--scale-hint alias=s``
        override of this run (``override``), an override persisted in the customs by an earlier run (the existing
        ``scale`` fallback entry of the alias with ``scale_source = override``), the pilot's ``template_meta.scale_hint``
        (``template_meta.scale_hint``; a uniform scale, so not for a ``height_only`` fallback), else
        :attr:`Options.fallback_scale` (``default_step``)."""
        if alias in self.opt.scale_overrides:
            return float(self.opt.scale_overrides[alias]), "override"
        for e in self.customs.entries.values():
            if e.fallback_of == alias and e.fallback_kind == "scale" and e.fallback_params.get("scale_source") == "override" \
                    and e.is_live:
                return float(e.fallback_params["scale"]), "override"
        hint = None if height_only else self.opt.scale_hints.get(alias)
        if hint:
            return float(hint), "template_meta.scale_hint"
        return float(self.opt.fallback_scale), "default_step"

    def _height_axis(self, subtask: str, category: str, model: str) -> int:
        """The model axis of a book's height (:data:`HEIGHT_SCALE_FALLBACK`): its reviewed ``frame_long`` (icl), else the longer
        of the bbox's x / y sides."""
        row = self.icl.row(subtask, "target", category, model) or {}
        long_ = (row.get("frame_long") or "").strip().lstrip("+-")
        if long_ in ("x", "y", "z"):
            return "xyz".index(long_)
        try:
            return 0 if float(row.get("bbox_x") or 0.0) >= float(row.get("bbox_y") or 0.0) else 1
        except ValueError:
            return 0

    def _scale_fallbacks(self, comb: Combination, alias: str) -> List[Combination]:
        """§5.5 / §6.10 ``scale_hint``: the target spawned at the pilot's hint (else :attr:`Options.fallback_scale`);
        the same models (variation / held-out included) run at that scale. Alias token ``<target>_sNNN``. A target category of
        :data:`HEIGHT_SCALE_FALLBACK` gets the height-only variant instead: the scale applies to its height axis only (the other
        axes keep the pick's spawn scale), alias token ``<target>_hNNN``, ``fallback_params.scale_axes = "height"``."""
        target = comb.roles.get("target")
        if target is None or target.kind != "sampleable" or "target" in comb.scale:
            return []
        height = target.category in HEIGHT_SCALE_FALLBACK.get(comb.subtask, ())
        scale, source = self.scale_for(alias, height_only=height)
        if scale <= 0 or scale >= 1.0:
            return []
        params: Dict[str, Any] = {"scale": round(scale, 3), "scale_source": source}
        if height:
            params.update({"scale_axes": "height", "height_axis": {C.model_key(c, m): "xyz"[self._height_axis(comb.subtask, c, m)] for c, m in target.models}})
        fb = self._derive(comb, alias, 1, "scale", params)
        fb.scale["target"] = round(scale, 3)
        if height:
            fb.scale_axes["target"] = "height"
        fb.scenes, fb.rooms, fb.floor_room = list(comb.scenes), dict(comb.rooms), dict(comb.floor_room)
        return [fb]

    def _next_model_fallbacks(self, comb: Combination, alias: str) -> List[Combination]:
        """Small-scale articulation yield fallback: the next candidate model of the category in variation / held-out
        order (``roles/representatives.csv`` slots of the pick) that passes the ``(handle target)`` requirement; the
        alias token carries the model (``blender_eyedvd``) because the category alone would be the primary's alias."""
        target = comb.roles.get("target")
        if target is None or target.kind != "sampleable" or "target" in comb.model_tokens or not target.models:
            return []
        iddl = self.iddls[comb.subtask]
        pick_id = comb.pick.split("~")[0]
        slots = self.icl.picks.get(pick_id, {}).get("target", {})
        current = target.models[0]
        seen = {current}
        candidates: List[Dict[str, str]] = []
        for slot in NEXT_MODEL_SLOT_ORDER:
            for row in slots.get(slot, []):
                cm = (row["category"], row["model"])
                if cm[0] == current[0] and cm not in seen:
                    seen.add(cm)
                    candidates.append(row)
        for row in candidates:
            resolved = self._resolve_row(comb.subtask, "target", row)
            is_heldout = (resolved["category"], resolved["model"]) in self.icl.heldout_models
            verdict = self._recheck_without_heldout(iddl, "target", resolved, None) if is_heldout else self.adm.check(iddl, "target", resolved)
            if verdict.reason:
                self.report.notes[comb.subtask].append(f"{alias}: next-model fallback candidate {row['category']}/{row['model']} ({row['slot']}) is {verdict.reason}")
                continue
            if comb.subtask in TARGET_FIT_SUBTASKS and "partner" in comb.roles:
                primary = self.customs.entries.get(alias)
                gen = primary.generation if primary is not None else G.defaults_for(comb.subtask)
                p_re = (primary.roles.get("partner") if primary is not None else None) or RoleEntry()
                fit = self._target_fit_verdict(iddl, gen, p_re, resolved["category"], resolved["model"], comb.roles["partner"].models, None)
                if fit["status"] in ("pending_scaled_grasps", "below_floor"):
                    self.report.notes[comb.subtask].append(f"{alias}: next-model fallback candidate {row['category']}/{row['model']} ({row['slot']}) "
                                                           f"would be withheld by the target fit ({fit['status']}, x{fit['record']['scale']:.2f})")
                    continue
            rp = RolePick("target", "sampleable", target.category, [(resolved["category"], resolved["model"])], pick_row=resolved,
                          joint=resolved.get("joint") or None, allow_heldout=True)
            fb = self._derive(comb, alias, 1, "next_model", {"model": resolved["model"], "slot": row["slot"], "primary_model": current[1]})
            fb.roles["target"] = rp
            fb.model_tokens.add("target")
            fb.notes = list(comb.notes)
            # the support follows the new model (a shorter / taller model may change the floor rule of §5.4)
            support_row = comb.roles["support"].pick_row if "support" in comb.roles else None
            if support_row is not None:
                try:
                    fb.roles["support"] = self._scene_pick(iddl, "support", dict(support_row), fb)
                    self._bind_scenes(iddl, fb)
                except Skip as s:
                    self.report.skipped[comb.subtask].append((fb.label, str(s)))
                    continue
            else:
                fb.scenes, fb.rooms, fb.floor_room = list(comb.scenes), dict(comb.rooms), dict(comb.floor_room)
            return [fb]
        self.report.notes[comb.subtask].append(
            f"{alias}: no next-model fallback ({target.category}/{current[1]} is the only admissible model of the category in the selection)")
        return []

    def _customs_entry(self, iddl: I.IDDL, comb: Combination, alias: str, definition_id: int) -> CustomsEntry:
        subtask = iddl.subtask
        existing = self.customs.entries.get(alias)
        existing_dict = existing.to_dict() if existing is not None else None
        entry = self.customs.get_or_create_entry(alias, iddl=I.relative_path(iddl.path, self.iddl_root) if iddl.path else None, definition_id=definition_id)
        entry.generation = G.seed_generation(subtask, existing_dict, reseed=self.opt.reseed_generation or None)
        if comb.fallback is not None:
            entry.set_fallback(comb.fallback.primary, comb.fallback.rank, comb.fallback.kind, comb.fallback.params)
        else:
            entry.fallback_of, entry.fallback_rank, entry.fallback_kind, entry.fallback_params = None, None, None, {}
        old_roles = dict(entry.roles)
        old_scenes = dict(entry.scenes)
        entry.roles = {}
        # scene blocks of this combination are rebuilt below; blocks of other scenes (another support of the same
        # category grounded earlier, A3 triplets) stay
        entry.scenes = {sc: se for sc, se in old_scenes.items() if sc not in comb.scenes}
        pose_roles = set(iddl.pose_evaluated_roles())
        for comp, members in iddl.groups.items():
            if comp in pose_roles:
                pose_roles |= set(members)
        verdicts: Dict[Tuple[str, str, str], Verdict] = {}

        # ---- roles ---------------------------------------------------------------------------------------------------
        for role in iddl.roles:
            rp = comb.roles[role]
            re_ = RoleEntry()
            if rp.kind == "group":
                cats = list(rp.members)
                re_.category = cats[0] if len(cats) == 1 else cats
                re_.sampleable = True
                if rp.pick_row is not None and rp.pick_row.get("dataset"):
                    re_.dataset = rp.pick_row["dataset"]
                if rp.pick_row is not None and rp.kind == "group" and role != "distractors":
                    re_.representative = rp.pick_row["model"]
                seed = entry.generation.setdefault("groups", {}).setdefault(role, G.group_seed(subtask, role))
                seed["size"] = sum(rp.members.values())
                seed["members"] = dict(rp.members)
                if rp.budget:
                    entry.generation["distractor_budget"] = dict(rp.budget)  # owner decision 2026-09-16 4a (§6.4 budget at grounding)
                if role == "distractors" and subtask == "disembedding":
                    # handoff 2026-10-10: the sheet's pose rule and the puzzle's layout; a re-grounding drops what the sheet no longer says
                    for key, value in (("distractor_pose", rp.pose), ("fixed_layout", rp.layout)):
                        if value:
                            entry.generation[key] = json.loads(json.dumps(value))
                        else:
                            entry.generation.pop(key, None)
            elif rp.kind == "sampleable":
                re_.category = rp.category
                re_.sampleable = True
                if rp.pick_row is not None:
                    re_.representative = rp.pick_row["model"]
                    if rp.pick_row.get("dataset"):
                        re_.dataset = rp.pick_row["dataset"]
                elif rp.models:
                    re_.representative = rp.models[0][1]
            else:
                re_.category = rp.category
                re_.sampleable = False
            for req in iddl.require_of(role):
                re_.require.update(req.customs_value())
            # an alias grounded before (or by another slot of this run: same categories, other models) keeps its
            # per-model facts and its representative model; the whitelists are unioned below
            old = old_roles.get(role)
            if old is not None:
                re_.scale.update(old.scale)
                re_.joint.update(old.joint)
                re_.handle_key.update(old.handle_key)
                re_.grasp_keys.update(old.grasp_keys)
                re_.orientation_equivalences.update(old.orientation_equivalences)
                if old.representative and comb.slot != "representative":
                    re_.representative = old.representative
            entry.roles[role] = re_
        entry.roles["floor"] = RoleEntry(sampleable=False)

        # ---- per-model facts of the whitelisted / pool models -------------------------------------------------------
        role_models: Dict[str, List[Tuple[str, str, Dict[str, str]]]] = {}
        for role in iddl.roles:
            rp = comb.roles[role]
            if rp.kind == "scene":
                continue
            wanted: List[Tuple[str, str]] = list(rp.models)
            if rp.kind == "group" and role == "distractors":
                wanted = [(c, m) for c, ms in rp.pools.items() for m in ms]
            elif self.opt.full_whitelists:
                cats = list(rp.members) if rp.kind == "group" else [rp.category]
                for cat in cats:
                    for r in self._pool_of(subtask, role, cat):
                        if (cat, r["model"]) not in wanted:
                            wanted.append((cat, r["model"]))
            rows = []
            for cat, m in wanted:
                r = self.icl.row(subtask, role, cat, m)
                if r is None and role == "distractors":
                    r = self.icl.row("disembedding", "distractors", cat, m)
                if r is None:
                    r = {"subtask": subtask, "role": role, "category": cat, "model": m, "kind": rp.kind, "ready": "", "_synthetic": "1"}
                rows.append((cat, m, r))
            role_models[role] = rows

        # ---- scenes ---------------------------------------------------------------------------------------------------
        for sc in comb.scenes:
            se = SceneEntry(rooms=list(comb.rooms.get(sc, [])))
            for role in iddl.roles:
                rp = comb.roles[role]
                if rp.kind != "scene":
                    continue
                if rp.is_floor:
                    se.bindings[role] = {"room": comb.floor_room.get(sc)} if comb.floor_room.get(sc) else {"room": None}
                    continue
                if rp.same_as and rp.same_as != "floor":
                    other = comb.roles[rp.same_as]
                    se.bindings[role] = [o["model"] for o in other.scene_objects.get(sc, [])]
                    continue
                objs = rp.scene_objects.get(sc, [])
                if objs and any(o.get("region_id") or "/" in (o.get("model") or "") for o in objs):
                    # wall regions (§5.5): bindings.support = wall object names, bindings.support_regions = the segments
                    def _wall(o):
                        return o["model"] if o.get("region_id") else (o["model"].split("/")[1] if o["model"].count("/") == 2 else o["model"])

                    se.bindings[role] = sorted({_wall(o) for o in objs})
                    se.bindings[f"{role}_regions"] = [
                        {"id": o.get("region_id") or o["model"], "wall": _wall(o), "room": o.get("room"), "p0": _parse_vec(o.get("p0")),
                         "p1": _parse_vec(o.get("p1")), "normal": _parse_vec(o.get("normal")),
                         "hang_z": float(o["hang_z"]) if o.get("hang_z") else None,
                         "length_m": float(o["length_m"]) if o.get("length_m") else None,
                         "max_object_m": float(o["max_object_m"]) if o.get("max_object_m") else None}
                        for o in objs
                    ]
                else:
                    se.bindings[role] = [o["model"] for o in objs]
                if rp.surface is not None and rp.pick_row is not None and rp.pick_row.get("scene") == sc:
                    se.bindings[f"{role}_surfaces"] = [dict(rp.surface)]
                # scene articulated targets: joint / handle key per scene object (§5.3)
                if any(r.kind == "handle" for r in iddl.require_of(role)):
                    for o in objs:
                        v = self.adm.check(iddl, role, o, scene=sc)
                        if v.reason:
                            self.report.blacklist[subtask][v.reason] += 1
                            entry.roles[role].joint.pop(f"{sc}/{o['model']}", None)
                            comb.notes.append(f"{o['model']}@{sc}: {v.reason} (scene object kept in bindings, no admissible joint)")
                            continue
                        preferred = rp.joint if (rp.pick_row is not None and o["model"] == (rp.pick_row.get("name") or rp.pick_row.get("model"))) else (o.get("joint") or None)
                        chosen = choose_joint(v.handles, preferred) or choose_joint(v.handles)
                        if chosen is None:
                            continue
                        joint, hrow = chosen
                        entry.roles[role].joint[f"{sc}/{o['model']}"] = joint
                        entry.roles[role].handle_key[f"{sc}/{o['model']}"] = hrow.get("key") or o.get("handle_key") or ""
            se.bindings["floor"] = {"room": comb.floor_room.get(sc)} if comb.floor_room.get(sc) else {"room": None}
            # rejections a sampler / the pilot recorded in this scene block earlier (any reason not ground_isg's own) bind
            # before the §2.2 checks: the model stays blacklisted with that reason and gets no per-model facts
            carried: Dict[Tuple[str, str, str], str] = {}
            old_block = old_scenes.get(sc)
            if old_block is not None:
                carried = {(role, cat, m): reason for role, cat, m, reason in old_block.blacklist_reasons() if not reason.startswith("ground_isg")}
            # whitelists / blacklists of the spawned roles
            for role, rows in role_models.items():
                rp = comb.roles[role]
                for cat, m, r in rows:
                    if (role, cat, m) in carried:
                        se.blacklist_add(role, cat, m, carried[(role, cat, m)])
                        self.report.blacklist[subtask][C.reason_code(carried[(role, cat, m)])] += 1
                        continue
                    v = verdicts.get((role, cat, m))
                    if v is None or role == "obstacles":
                        v = self.adm.check(iddl, role, r, scene=sc)
                        if rp.allow_heldout and v.reason == "heldout" and (cat, m) in rp.models:
                            v = self._recheck_without_heldout(iddl, role, r, sc)
                        verdicts[(role, cat, m)] = v
                    if role == "distractors" and self.icl.are_lookalikes((cat, m), tuple(comb.roles["target"].models[0])):
                        v = Verdict(reason="lookalike_of_target")
                    if v.reason:
                        reason = self._reason(v.reason) if v.source == "ground_isg" else C.blacklist_reason(v.reason, tool=v.source, date=self.today)
                        se.blacklist_add(role, cat, m, reason)
                        self.report.blacklist[subtask][v.reason] += 1
                        continue
                    se.whitelist_add(role, cat, m)
                    self._model_facts(entry.roles[role], iddl, role, cat, m, r, v, rp)
                    if subtask in WALL_SUBTASKS and role == "target":
                        # §5.5 frame per model (owner 22:55): the reviewed icl frame goes into generation.front_axis / up_axis
                        # (dict form); the sampler rejects an alias without it instead of defaulting
                        frame = write_wall_frame(entry.generation, subtask, cat, m, r)
                        if len(frame) < len(WALL_FRAME_KEYS):
                            comb.notes.append(f"{cat}/{m}: incomplete reviewed frame in the icl ({frame}); the sampler will reject missing_frame")
                    for p in v.pending:
                        rec = Constraint("blacklist", {"scene": sc, role: C.model_key(cat, m)}, False,
                                         C.constraint_source("ground_isg", VERSION, date=self.today), {"reason": p, "pending": True})
                        if entry.add_constraint(rec):
                            self.report.pending[p] += 1
            # held-out models of every spawned role's categories (global rule)
            for role in iddl.roles:
                rp = comb.roles[role]
                if rp.kind == "scene":
                    continue
                cats = list(rp.members) if rp.kind == "group" else [rp.category]
                for cat, m in sorted(self.icl.heldout_models):
                    if cat in cats and not (rp.allow_heldout and (cat, m) in rp.models):
                        if not se.is_blacklisted(role, cat, m):
                            se.blacklist_add(role, cat, m, self._reason("heldout"))
                            self.report.blacklist[subtask]["heldout"] += 1
            # an existing scene block: its whitelisted models of the same categories stay (union), the rejections a
            # sampler recorded stay (ground_isg's own reasons were re-derived above)
            old = old_scenes.get(sc)
            if old is not None:
                for role, cats in old.whitelist.items():
                    rp = comb.roles.get(role)
                    if rp is None or rp.kind == "scene":
                        continue
                    own = set(rp.members) if rp.kind == "group" else {rp.category}
                    for cat, models in cats.items():
                        if cat in own:
                            for m in models:
                                se.whitelist_add(role, cat, m)
                for role, cat, m, reason in old.blacklist_reasons():
                    if not se.is_blacklisted(role, cat, m) and not reason.startswith("ground_isg"):
                        se.blacklist_add(role, cat, m, reason)
                # scene objects another combination of the same alias bound in this scene (a second representative
                # support of the category, A3 triplets) stay bound
                for role, old_val in old.bindings.items():
                    new_val = se.bindings.get(role)
                    if new_val is None and role.endswith("_surfaces") and isinstance(old_val, list):
                        se.bindings[role] = list(old_val)
                    elif isinstance(old_val, list) and isinstance(new_val, list):
                        if role.endswith("_surfaces"):
                            # a re-grounded row replaces its own entry; the other rows' entries stay
                            rows = {e.get("row") for e in new_val if isinstance(e, dict)}
                            new_val.extend(e for e in old_val if isinstance(e, dict) and e.get("row") not in rows)
                        elif role.endswith("_regions"):
                            ids = {r.get("id") for r in new_val if isinstance(r, dict)}
                            new_val.extend(r for r in old_val if isinstance(r, dict) and r.get("id") not in ids)
                        else:
                            new_val.extend(o for o in old_val if o not in new_val)
                    for room in old.rooms:
                        if room not in se.rooms:
                            se.rooms.append(room)
            entry.scenes[sc] = se

        # ---- role scales (§5.5 dict form, per whitelisted model + "default"): the scale fallbacks of the target and the
        # disembedding source tray at generation.source_tray_scale (owner decision 2026-09-16 4b; the destination stays 1.0)
        role_scales = dict(comb.scale)
        if subtask == "disembedding" and "source" in entry.roles:
            tray_scale = entry.generation.get("source_tray_scale")
            if tray_scale not in (None, 1, 1.0):
                role_scales["source"] = float(tray_scale)
        primary = self.customs.entries.get(comb.fallback.primary) if comb.fallback is not None else None
        for role, s in role_scales.items():
            re_ = entry.roles.get(role)
            if re_ is None:
                continue
            height = comb.scale_axes.get(role) == "height"
            # the unscaled spawn scale of a fallback's model is its primary's (a re-grounded fallback carries its own scaled value)
            base_of = primary.roles[role].scale if primary is not None and role in primary.roles else re_.scale
            # once per model: a model whitelisted in several scene blocks (one per support) must not be scaled again
            models = list(dict.fromkeys((cat, m) for sc in entry.scenes.values() for cat, ms in sc.whitelist.get(role, {}).items() for m in ms))
            for cat, m in models:
                key = C.model_key(cat, m)
                base = base_of.get(key)
                if height:
                    vec = [float(b) for b in base] if isinstance(base, list) else [float(base if base is not None else 1.0)] * 3
                    axis = self._height_axis(subtask, cat, m)
                    vec[axis] *= s
                    re_.scale[key] = [round(v, 4) for v in vec]
                else:
                    re_.scale[key] = [round(float(b) * s, 4) for b in base] if isinstance(base, list) else s
            if not height:
                re_.scale["default"] = s
        if subtask in TARGET_FIT_SUBTASKS:
            self._apply_target_fit(iddl, comb, entry, alias)
        if subtask in HANDEDNESS_SUBTASKS:
            self._write_handedness(entry, subtask, alias)
        return entry

    def _recheck_without_heldout(self, iddl: I.IDDL, role: str, row: Dict[str, str], scene: str) -> Verdict:
        """The ``heldout`` slot was requested for this role: run the other checks on the held-out model."""
        saved = set(self.icl.heldout_models)
        self.icl.heldout_models.discard((row["category"], row["model"]))
        try:
            return self.adm.check(iddl, role, row, scene=scene)
        finally:
            self.icl.heldout_models |= saved

    def _admissible_keys(self, iddl: I.IDDL, role: str, cat: str, m: str) -> Set[str]:
        keys: Set[str] = set()
        for req in iddl.require_of(role):
            if req.kind == "grasps":
                keys |= {r["key"] for r in self.icl.admissible_grasps(cat, m, req.pipeline or "")}
        return keys

    def _preferred_grasp_key(self, iddl: I.IDDL, role: str, cat: str, m: str, rp: RolePick, default: str, row: Optional[Dict[str, str]] = None,
                             keys: Optional[Set[str]] = None) -> str:
        """The library key of a model among its admissible keys for the role's pipeline (or among ``keys``): the key its candidates
        row names (``grasp_key`` column) when admissible, else the pick's :data:`GRASP_KEY_SUFFIX_PREFERENCE`, else a spawn-scaled
        floating key (``__sf``: grasps computed and attach-tested on the scaled mesh, owner 2026-10-11 - its ``scale`` comes with the
        candidates row), else ``default`` (the alphabetically first)."""
        if keys is None:
            keys = set()
            for req in iddl.require_of(role):
                if req.kind == "grasps":
                    keys |= {r["key"] for r in self.icl.admissible_grasps(cat, m, req.pipeline or "")}
        named = (row or {}).get("grasp_key") or ""
        if f"{named}__sf" in keys:  # the spawn-scaled floating key of the named one (owner 2026-10-11: the 2x thick teaspoon)
            return f"{named}__sf"
        if named in keys:
            return named
        suffix = GRASP_KEY_SUFFIX_PREFERENCE.get((rp.pick_row or {}).get("pick") or "")
        for sfx in ([suffix] if suffix else []) + ["__sf"]:
            hits = sorted(k for k in keys if k.endswith(sfx))
            if hits:
                return hits[0]
        return default

    def _model_facts(self, re_: RoleEntry, iddl: I.IDDL, role: str, cat: str, m: str, row: Dict[str, str], v: Verdict, rp: RolePick) -> None:
        key = C.model_key(cat, m)
        subtask = iddl.subtask
        if v.grasp_key:
            re_.grasp_keys[key] = self._preferred_grasp_key(iddl, role, cat, m, rp, v.grasp_key, row)
        gk = re_.grasp_keys.get(key) or ""
        # a spawn-scaled key's grasps were tested at its scale (``__sf``, ``__s``, ``__st``, a target-fit key): the model spawns at exactly that scale
        key_scale = self.icl.key_scale.get(gk) if gk else None
        if key_scale is not None:
            re_.scale[key] = key_scale
        scale = _parse_scale(row.get("scale") or "")
        if scale is None and rp.pick_row is not None and (rp.pick_row.get("category"), rp.pick_row.get("model")) == (cat, m):
            scale = _parse_scale(rp.pick_row.get("scale") or "")  # the sheet's spawn scale of the pick's own model (alignment-axial#02: 2.5x thickness)
        if scale is not None and key not in re_.scale:
            re_.scale[key] = scale
        if v.handles:
            preferred = rp.joint if (rp.pick_row is not None and rp.pick_row.get("model") == m) else (row.get("joint") or None)
            chosen = choose_joint(v.handles, preferred) or choose_joint(v.handles)
            if chosen is not None:
                joint, hrow = chosen
                re_.joint.setdefault(key, joint)
                re_.handle_key.setdefault(key, hrow.get("key") or row.get("handle_key") or f"{cat}__{m}")
        if role in set(iddl.pose_evaluated_roles()) | {mm for comp, ms in iddl.groups.items() if comp in iddl.pose_evaluated_roles() for mm in ms}:
            tags = self._orientation_equivalences(subtask, role, row)
            if tags is not None:
                re_.orientation_equivalences[key] = tags

    # -- owner rules 2026-10-11 of the IORS small-scale pairs: target fit scale, handedness --------------------------------------
    @staticmethod
    def _whitelisted(entry: CustomsEntry, role: str) -> List[Tuple[str, str]]:
        return sorted({(c, m) for sc in entry.scenes.values() for c, ms in sc.whitelist.get(role, {}).items() for m in ms})

    def _in_selection(self, subtask: str, role: str, cat: str, m: str) -> bool:
        if self.opt.mode == "representative":
            return any(r.get("category") == cat and r.get("model") == m for p, roles in self.icl.picks.items() if p.split("#")[0] == subtask
                       for rows in roles.get(role, {}).values() for r in rows)
        row = self.icl.row(subtask, role, cat, m)
        return row is not None and self._marked(row)

    def _shrunk_key(self, key: Optional[str], base: Optional[List[float]]) -> bool:
        """``key``'s grasps were tested at a uniform shrink of ``base`` (None = unit): a target-fit key, never a model's own."""
        f = PR.uniform_factor(self.icl.key_scale[key], base) if key and self.icl.key_scale.get(key) is not None else None
        return f is not None and f < 1.0 - 1e-3

    def _target_base(self, iddl: I.IDDL, cat: str, m: str, rp: Optional[RolePick]) -> Tuple[Optional[str], Optional[List[float]]]:
        """The library key a target model gets at its own spawn scale and that scale (None = unit), as :meth:`_model_facts` derives
        them, with the keys tested at a uniform shrink of it (target-fit keys) left out."""
        row = self.icl.row(iddl.subtask, "target", cat, m) or {}
        row_scale = _parse_scale(row.get("scale") or "")
        keys = self._admissible_keys(iddl, "target", cat, m)
        own = {k for k in keys if not self._shrunk_key(k, row_scale)}
        if not own:
            return None, row_scale
        key = self._preferred_grasp_key(iddl, "target", cat, m, rp or RolePick("target", "sampleable"), sorted(own)[0], row or None, keys=own)
        return key, self.icl.key_scale.get(key) or row_scale

    def _target_fit_verdict(self, iddl: I.IDDL, gen: Dict[str, Any], p_re: RoleEntry, cat: str, m: str, partners: Sequence[Tuple[str, str]],
                            rp: Optional[RolePick]) -> Dict[str, Any]:
        """Fit of target ``cat/m`` with every partner it can be drawn with (``pair_rules``; thresholds from ``gen``): ``status`` =
        ``fits`` / ``scaled`` (a key tested at the fit scale exists) / ``pending_scaled_grasps`` / ``below_floor`` / ``no_bbox``, the
        model's own key and scale (``base_key`` / ``base_scale``) and, when it does not fit as is, the ``record`` for
        ``generation.target_fit``. The target is measured at its key's rest - inside: standing on ``gen["target_up_axis"]`` when that
        names an axis for the model (owner 2026-10-11, inside#03: the sampler stands it so at the goal; the record carries
        ``up_axis``)."""
        subtask = iddl.subtask
        kind = TARGET_FIT_SUBTASKS[subtask]
        base_key, base = self._target_base(iddl, cat, m, rp)
        out: Dict[str, Any] = {"status": "fits", "base_key": base_key, "base_scale": base}
        bbox = self.icl.model_bbox(subtask, "target", cat, m)
        if bbox is None:
            out["status"] = "no_bbox"
            return out
        up = None
        if kind == "inside":
            try:
                up = PR.target_up_axis(gen.get("target_up_axis"), cat, m)
            except ValueError as e:
                raise GroundingError(f"{subtask}: generation.target_up_axis of {cat}/{m}: {e}") from e
        t_ext = PR.rest_extents(bbox, base, PR.standing_quat_wxyz(up) if up else (self.icl.key_rest.get(base_key) if base_key else None))
        worst = None
        for pc, pm in partners:
            pb = self.icl.model_bbox(subtask, "partner", pc, pm)
            if pb is None:
                continue
            pkey = p_re.grasp_keys.get(C.model_key(pc, pm))
            p_ext = PR.rest_extents(pb, _role_scale(p_re, pc, pm), self.icl.key_rest.get(pkey) if pkey else None)
            if kind == "inside":
                fit = PR.fit_inside(t_ext, p_ext, float(gen.get("opening_margin_m", 0.01)), float(gen.get("max_protrusion_fraction", 0.5)))
            else:
                fit = PR.fit_on_partner(t_ext, p_ext, float(gen.get("partner_footprint_margin_m", 0.01)), gen.get("fit_tiers") or PR.FIT_TIERS)
            if worst is None or fit.scale < worst[1].scale:
                worst = ((pc, pm), fit)
        if worst is None or not worst[1].bigger:
            return out
        (pc, pm), fit = worst
        s = PR.floor_scale(fit.scale)
        floor = float(gen.get("target_fit_scale_min", PR.TARGET_FIT_SCALE_MIN))
        unit = base or [1.0, 1.0, 1.0]
        rec: Dict[str, Any] = {"rule": kind, "scale": s, "partner": C.model_key(pc, pm), "limit": fit.limit, **fit.facts, **({"up_axis": up} if up else {}),
                               "base_scale": base, "spawn_scale": [round(b * s, 4) for b in unit]}
        if s < floor - 1e-9:
            rec.update(status="below_floor", scale_min=floor)
        else:
            best = None
            for k in self._admissible_keys(iddl, "target", cat, m):
                f = PR.uniform_factor(self.icl.key_scale.get(k), base) if self.icl.key_scale.get(k) is not None else None
                if f is not None and floor - 1e-6 <= f <= s + 1e-6 and (best is None or f > best[1]):
                    best = (k, f)
            if best is not None:
                rec.update(status="scaled", grasp_key=best[0], scale=round(best[1], 4), spawn_scale=list(self.icl.key_scale[best[0]]))
            else:
                pipeline = next((r.pipeline for r in iddl.require_of("target") if r.kind == "grasps"), None)
                rec.update(status="pending_scaled_grasps", needs={"pipeline": pipeline, "base_key": base_key, "scale": rec["spawn_scale"]})
        out.update(status=rec["status"], record=rec)
        return out

    def _set_target_facts(self, re_: RoleEntry, key: str, grasp_key: Optional[str], scale: Optional[List[float]]) -> None:
        if grasp_key:
            re_.grasp_keys[key] = grasp_key
        else:
            re_.grasp_keys.pop(key, None)
        if scale is not None:
            re_.scale[key] = list(scale)
        else:
            re_.scale.pop(key, None)

    def _withhold_target(self, entry: CustomsEntry, subtask: str, cat: str, m: str, rec: Dict[str, Any]) -> None:
        if rec["status"] == "below_floor":
            code = "partner_too_small" if rec["rule"] == "on_partner" else "does_not_fit_inside"
            text = f"{code} (needs x{rec['scale']:.2f} < {rec['scale_min']:.2f} to fit {rec['partner']})"
        else:
            code = f"pending_scaled_grasps:{(rec.get('needs') or {}).get('pipeline')}"
            text = f"{code} (x{rec['scale']:.2f} to fit {rec['partner']})"
        for se in entry.scenes.values():
            if not se.is_blacklisted("target", cat, m):
                se.blacklist_add("target", cat, m, self._reason(text))
                self.report.blacklist[subtask][code] += 1

    def _apply_target_fit(self, iddl: I.IDDL, comb: Combination, entry: CustomsEntry, alias: str) -> None:
        """Owner 2026-10-11 (IORS small scale, ``pair_rules``): a whitelisted target model bigger than a whitelisted partner model
        (ontop / under: the partner's top; inside: the opening) is shrunk before sampling - the largest uniform factor (0.01 steps)
        that fits the smallest partner it can be drawn with (the sampler draws any target with any partner of the alias). The factor
        needs a library key whose grasps were tested at that scale (``grasps.csv`` ``scale`` = the model's own scale x a factor in
        [``target_fit_scale_min``, fit factor], the largest): ``roles.target.scale`` / ``grasp_keys`` then carry it. Without one the
        model is withheld - blacklisted ``pending_scaled_grasps:<pipeline>``, never spawned scaled with grasps of another scale -, as
        is a model that needs less than ``generation.target_fit_scale_min`` (the sampler's reason: ``partner_too_small`` /
        ``does_not_fit_inside``). ``generation.target_fit`` records every model that does not fit as is (factor, bounding partner,
        status, the key a grasp-library run must provide). A model withheld by an earlier combination (another slot of the alias)
        or grounding is re-evaluated with the alias's partners of now: still too big -> withheld with the new factor; fitting now
        -> back in the whitelist, as the whitelist union would have carried it."""
        subtask = iddl.subtask
        t_re, p_re = entry.roles.get("target"), entry.roles.get("partner")
        if t_re is None or p_re is None:
            return
        prev = dict(entry.generation.get("target_fit") or {})
        targets, partners = self._whitelisted(entry, "target"), self._whitelisted(entry, "partner")
        carried: List[Tuple[str, str]] = []
        for key, rec in prev.items():
            cat, m = key.split("/", 1)
            if rec.get("status") not in ("pending_scaled_grasps", "below_floor") or (cat, m) in targets or not self._in_selection(subtask, "target", cat, m):
                continue
            blocks = list(entry.scenes.values())
            if blocks and all(se.is_blacklisted("target", cat, m) and not se.blacklist["target"][cat][m].split(": ", 1)[-1].startswith(
                    ("pending_scaled_grasps", "partner_too_small", "does_not_fit_inside")) for se in blocks):
                continue  # blacklisted for another reason everywhere (held out, no grasps): the record says nothing more
            carried.append((cat, m))
        records: Dict[str, Dict[str, Any]] = {}
        for cat, m in targets + carried:
            key = C.model_key(cat, m)
            v = self._target_fit_verdict(iddl, entry.generation, p_re, cat, m, partners, comb.roles.get("target"))
            restore = (prev.get(key) or {}).get("status") == "scaled" or (cat, m) in carried or self._shrunk_key(t_re.grasp_keys.get(key), v["base_scale"])
            if v["status"] == "no_bbox":
                self.report.notes[subtask].append(f"{alias}: target fit of {key} not checked (no bounding box in the icl or the asset)")
            if v["status"] in ("fits", "no_bbox", "scaled") and (cat, m) in carried:
                for se in entry.scenes.values():
                    se.whitelist_add("target", cat, m)
            if v["status"] in ("fits", "no_bbox"):
                if restore:  # the earlier fit key / scale of this rule, or a model withheld before: the model's own key and scale
                    self._set_target_facts(t_re, key, v["base_key"], v["base_scale"])
                continue
            rec = v["record"]
            records[key] = rec
            if v["status"] == "scaled":
                self._set_target_facts(t_re, key, rec["grasp_key"], rec["spawn_scale"])
                continue
            if restore:
                self._set_target_facts(t_re, key, v["base_key"], v["base_scale"])
            self._withhold_target(entry, subtask, cat, m, rec)
        if records:
            entry.generation["target_fit"] = dict(sorted(records.items()))
        else:
            entry.generation.pop("target_fit", None)

    def _write_handedness(self, entry: CustomsEntry, subtask: str, alias: str) -> None:
        """Owner 2026-10-11 (nextto): ``generation.handedness`` = ``{"<category>/<model>": "left" | "right"}`` (:data:`HANDEDNESS`) of
        every whitelisted target / partner model that has a hand; the sampler leaves the target on its own side of the partner (the
        right shoe on the robot's right of the left shoe). A drawable same-hand pair and a model of a handed category without a hand
        are reported."""
        handed_cats = {c for c, _ in HANDEDNESS}
        hands: Dict[str, str] = {}
        by_role: Dict[str, Dict[str, str]] = {"target": {}, "partner": {}}
        for role in by_role:
            for cat, m in self._whitelisted(entry, role):
                h = HANDEDNESS.get((cat, m))
                if h:
                    hands[C.model_key(cat, m)] = by_role[role][C.model_key(cat, m)] = h
                elif cat in handed_cats:
                    self.report.notes[subtask].append(f"{alias}: {role} {cat}/{m} has no hand in ground_isg.HANDEDNESS (its side is not enforced)")
        for t, ht in by_role["target"].items():
            for p, hp in by_role["partner"].items():
                if t != p and ht == hp:  # one model is never drawn in both roles
                    self.report.notes[subtask].append(f"{alias}: target {t} and partner {p} are both {ht} (the sampler rejects the pair: handedness_pair)")
        if hands:
            entry.generation["handedness"] = dict(sorted(hands.items()))
        else:
            entry.generation.pop("handedness", None)

    def _report_target_fit(self, subtask: str) -> None:
        """One report note per model of ``generation.target_fit`` of the aliases this run produced for ``subtask`` (final state)."""
        for alias in sorted({gp.alias for gp in self.grounded if gp.subtask == subtask}):
            entry = self.customs.entries.get(alias)
            records = (entry.generation.get("target_fit") or {}) if entry is not None else {}
            for key, rec in sorted(records.items()):
                need = rec.get("needs")
                what = {"scaled": f"key {rec.get('grasp_key')}", "below_floor": f"withheld, below target_fit_scale_min {rec.get('scale_min')}",
                        "pending_scaled_grasps": f"withheld, needs {(need or {}).get('pipeline')} grasps of {(need or {}).get('base_key')} at {(need or {}).get('scale')}"}
                standing = f" standing on {rec['up_axis']}" if rec.get("up_axis") else ""
                self.report.notes[subtask].append(f"{alias}: target fit {key}{standing} x{rec['scale']:.2f} ({rec['partner']}): {what.get(rec['status'], rec['status'])}")
            if entry is None or TARGET_FIT_SUBTASKS[subtask] != "inside" or entry.generation.get("target_up_axis") is None:
                continue
            for cat, m in self._whitelisted(entry, "target"):
                up = PR.target_up_axis(entry.generation["target_up_axis"], cat, m)
                if up and C.model_key(cat, m) not in records:
                    self.report.notes[subtask].append(f"{alias}: target fit {C.model_key(cat, m)} standing on {up} (generation.target_up_axis): fits as is")

    @staticmethod
    def _orientation_equivalences(subtask: str, role: str, row: Dict[str, str]) -> Optional[List[str]]:
        """§3.13 rule 2: the symmetry tags of the model (wall targets: the reviewed ``symmetric_180`` half turn about
        the wall normal = front axis); None where the orientation is not evaluated (object scaling depth metric,
        inside target, blocks)."""
        if subtask == "object_scaling":
            return None
        if subtask == "configuration_matching-block_design":
            return None
        if subtask == "inter_object_relation_satisfaction-inside" and role == "target":
            return None
        if subtask in WALL_SUBTASKS:
            if (row.get("symmetric_180") or "") == "1" and row.get("frame_front"):
                return [C.half_turn_tag(row["frame_front"])]
            return []
        if not (row.get("symmetry") or ""):
            return None  # not surveyed (scene rows, trays, articulated targets): no claim
        return _symmetry_tags(row)

    # -- run ------------------------------------------------------------------------------------------------------------
    def run(self, subtasks: Optional[Sequence[str]] = None) -> Report:
        wanted = list(subtasks) if subtasks else sorted(self.iddls)
        unknown = [s for s in wanted if s not in self.iddls]
        if unknown:
            raise GroundingError(f"unknown subtask id(s) {unknown}; known: {sorted(self.iddls)}")
        for subtask in wanted:
            combos = self.combinations(subtask)
            if not combos and not self.report.skipped.get(subtask):
                self.report.skipped[subtask].append((subtask, f"no_combination: no candidate rows carry the {self.opt.mode} mark"))
            for comb in combos:
                gp = self._ground_or_skip(comb)
                if gp is None:
                    continue
                # fallback aliases behind the primary (owner decision 2026-09-16 B); the primary lists them in rank order
                fallback_aliases: List[str] = []
                for fb in self.fallbacks_for(comb, gp.alias, gp.entry):
                    fgp = self._ground_or_skip(fb)
                    if fgp is not None and fgp.alias not in fallback_aliases:
                        fallback_aliases.append(fgp.alias)
                if self.opt.fallbacks:
                    self._update_fallback_list(gp, fallback_aliases)
                if self.opt.representative_test and not self.opt.dry_run:
                    for sc in comb.scenes:
                        outcome = run_representative_test(gp, self.customs, sc)
                        if outcome:
                            self.report.tests.append(f"{gp.alias} @ {sc}: {outcome}")
            self._supersede_stale_primaries(subtask)
            self._prune_stale_scene_blocks(subtask)
            if subtask in TARGET_FIT_SUBTASKS:
                self._report_target_fit(subtask)
            if subtask in WALL_SUBTASKS:
                self._write_dormant_wall_frames(subtask)
            if self.opt.reseed_generation:
                # entries of the subtask this run did not produce (superseded fallbacks) follow the spec number too
                produced = {gp.alias for gp in self.grounded if gp.subtask == subtask}
                iddl_rel = I.relative_path(self.iddls[subtask].path, self.iddl_root) if self.iddls[subtask].path else None
                for alias, entry in self.customs.entries.items():
                    if alias not in produced and entry.iddl == iddl_rel:
                        entry.generation = G.seed_generation(subtask, entry.to_dict(), reseed=self.opt.reseed_generation)
            if not self.opt.dry_run:
                self.customs.save()
        return self.report

    def _write_dormant_wall_frames(self, subtask: str) -> None:
        """§5.5 frames for the wall aliases of ``subtask`` this run did not produce (superseded fallbacks such as
        ``painting_s070``, which the first-demo loop still samples): every whitelisted target model gets its reviewed icl
        frame written into ``generation`` (idempotent; a model without an icl row is left alone and noted — the sampler
        rejects it ``missing_frame``)."""
        produced = {gp.alias for gp in self.grounded if gp.subtask == subtask}
        iddl_rel = I.relative_path(self.iddls[subtask].path, self.iddl_root) if self.iddls[subtask].path else None
        for alias, entry in self.customs.entries.items():
            if alias in produced or entry.iddl != iddl_rel:
                continue
            for sc in entry.scenes.values():
                for cat, models in sc.whitelist.get("target", {}).items():
                    for m in models:
                        row = self.icl.row(subtask, "target", cat, m)
                        if row is None:
                            self.report.notes[subtask].append(f"{alias}: dormant alias, {cat}/{m} has no icl row for its §5.5 frame")
                            continue
                        write_wall_frame(entry.generation, subtask, cat, m, row)

    def _update_fallback_list(self, gp: GroundedProblem, produced: List[str]) -> None:
        """``primary.fallbacks`` = the fallbacks produced by this run (rank order). A fallback entry of an earlier run
        that this run no longer produces (another scale after a new hint, another next model) is **superseded**: it
        keeps its entry, id and BDDL (ids are never reused) but leaves the primary's list and is marked
        ``superseded_by = <the new alias of its kind>`` (or ``true``), so a scheduler never queues it."""
        kinds = {a: self.customs.entries[a].fallback_kind for a in produced if a in self.customs.entries}
        for alias, e in self.customs.entries.items():
            if e.fallback_of != gp.alias or alias in produced:
                continue
            if e.superseded_by:
                continue
            newer = next((a for a, k in kinds.items() if k == e.fallback_kind), None)
            e.superseded_by = newer or True
            self.report.notes[gp.subtask].append(f"{alias}: superseded fallback of {gp.alias}" + (f" (now {newer})" if newer else ""))
        # a superseded fallback that this run produces again (a hint went back) is live again
        for a in produced:
            e = self.customs.entries.get(a)
            if e is not None:
                e.superseded_by = None
        live = [a for a in produced if a in self.customs.entries]
        gp.entry.fallbacks = sorted(live, key=lambda a: (self.customs.entries[a].fallback_rank or 0, a))

    def _supersede_stale_primaries(self, subtask: str) -> None:
        """A primary of an earlier grounding that this run replaced -- same pick / slot / mode, another alias (the
        distractor token changed with the tray budget, the support pick changed) -- is marked ``superseded_by`` the new
        alias, together with its fallbacks; a produced alias is live again. Uses the ``problems.json`` records (pick /
        slot / mode per alias)."""
        iddl = self.iddls[subtask]
        activity_dir = os.path.join(self.bddl_root, self.activity_of(iddl))
        if not os.path.isdir(activity_dir):
            return
        from ispatialgym.sampling.bddl_from_iddl import ProblemRegistry

        registry = ProblemRegistry.load(activity_dir, self.activity_of(iddl))
        primaries = [gp for gp in self.grounded if gp.subtask == subtask and gp.combination.fallback is None]
        produced = {gp.alias for gp in primaries}
        by_key: Dict[Tuple[str, str, str], str] = {}
        for gp in primaries:  # every grounding, not one per alias: two picks may share an alias (same category)
            by_key[(gp.combination.pick, gp.combination.slot, self.opt.mode)] = gp.alias
        for rec in registry.records:
            if rec.get("subtask") != subtask or rec.get("fallback_of") or rec.get("legacy"):
                continue
            alias = rec.get("full_alias", rec["alias"])
            entry = self.customs.entries.get(alias)
            if entry is None:
                continue
            if alias in produced:
                entry.superseded_by = None
                continue
            newer = by_key.get((rec.get("pick"), rec.get("slot"), rec.get("mode")))
            if newer is None and rec.get("mode") == self.opt.mode and rec.get("slot") in self.opt.slots and not self.opt.scenes:
                newer = True  # the pick itself left the selection (no grounding of this pick / slot any more)
            if newer and newer != alias and not entry.superseded_by:
                entry.superseded_by = newer
                self.report.notes[subtask].append(f"{alias}: superseded primary" + (f" (now {newer})" if newer is not True else " (its pick left the selection)"))
                for fb_alias in entry.fallbacks:
                    fb = self.customs.entries.get(fb_alias)
                    if fb is not None and not fb.superseded_by:
                        fb.superseded_by = newer
                entry.fallbacks = []

    def _prune_stale_scene_blocks(self, subtask: str) -> None:
        """An alias this run produced keeps only the scene blocks its combinations produced: a scene block of an earlier
        grounding whose scene object left the selection (the crowded kitchen picks of the large-scale articulation)
        goes -- unless the run was restricted with ``--scene`` (a partial run never drops blocks)."""
        if self.opt.scenes:
            return
        produced: Dict[str, Set[str]] = {}
        for gp in self.grounded:
            if gp.subtask == subtask:
                produced.setdefault(gp.alias, set()).update(gp.combination.scenes)
        for alias, scenes in produced.items():
            entry = self.customs.entries.get(alias)
            if entry is None:
                continue
            stale = [sc for sc in entry.scenes if sc not in scenes]
            for sc in stale:
                del entry.scenes[sc]
                self.report.notes[subtask].append(f"{alias}: scene block {sc} dropped (its scene object left the selection)")
            for key in list(entry.roles.get("target", RoleEntry()).joint):  # scene-object joints of the dropped blocks
                if "/" in key and key.split("/")[0] in stale:
                    entry.roles["target"].joint.pop(key, None)
                    entry.roles["target"].handle_key.pop(key, None)

    def _ground_or_skip(self, comb: Combination) -> Optional[GroundedProblem]:
        subtask = comb.subtask
        try:
            return self.ground(comb)
        except Skip as s:
            self.report.skipped[subtask].append((comb.label, str(s)))
        except ISGProblemError as e:
            # the rendered BDDL is rejected by ispatialgym.activity.problem (e.g. a synset the INSTANCE_RE of the
            # parser does not accept): nothing is written for this combination
            self.report.skipped[subtask].append((comb.label, f"bddl_rejected: {e}"))
            log.warning("%s: BDDL rejected by the parser: %s", comb.label, e)
        except (GroundingError, I.IDDLError, KeyError) as e:
            self.report.skipped[subtask].append((comb.label, f"error: {e}"))
            log.warning("%s: %s", comb.label, e)
        return None


# ----------------------------------------------------------------------------------------------------------------
# D2 hook
# ----------------------------------------------------------------------------------------------------------------


def run_representative_test(problem: GroundedProblem, customs: Customs, scene: str) -> None:
    """**D2 HOOK — representative-model test of a grounded alias (spec §9): template + instance 0.**

    ``ground_isg`` calls this once per grounded problem and candidate scene after the BDDL and the customs entry are
    written. The v2 sampler (TODO D2, ``ispatialgym.sampling.v2``) implements it as::

        ispatialgym.sampling.v2.core.run_representative_test(problem: ispatialgym.sampling.ground_isg.GroundedProblem,
                                                             customs: ispatialgym.sampling.customs.Customs,
                                                             scene: str) -> Optional[str]

    where ``problem`` carries ``alias`` / ``bddl_alias`` / ``subtask`` / ``activity`` / ``definition_id`` /
    ``bddl_path`` / ``bddl_text`` / ``problem`` (the parsed ``ISGProblem``) / ``entry`` (the live ``CustomsEntry``,
    same object as ``customs[problem.alias]``) / ``iddl`` / ``combination`` (the picks per role), and the
    implementation (a) runs the full initialization once with the representative models on the scene's bindings, (b)
    on success keeps template + instance 0 and appends the learned constraints to ``problem.entry.constraints``, (c) on
    failure moves the representative model to ``problem.entry.scenes[scene].blacklist`` with the rejection reason
    (``customs.blacklist_reason`` format) and retries the next representative once, and (d) returns a one-line
    outcome (or None). ``ground_isg`` saves the customs file after every subtask, so in-place edits of ``problem.entry``
    are persisted. The import is lazy (the sampler needs the simulator stack); until the function exists the test is
    skipped with a log line and the whitelists stay as the icl derived them.
    """
    try:
        from ispatialgym.sampling.v2 import core as v2_core  # noqa: WPS433 (lazy: needs OmniGibson)
    except Exception as e:  # ImportError, or the simulator stack failing to import
        _log_skip(f"ispatialgym.sampling.v2.core not available ({e})", problem.alias, scene)
        return None
    impl = getattr(v2_core, "run_representative_test", None)
    if impl is None:
        _log_skip("ispatialgym.sampling.v2.core has no run_representative_test yet", problem.alias, scene)
        return None
    return impl(problem, customs, scene)


_SKIP_LOGGED: Set[str] = set()


def _log_skip(why: str, alias: str, scene: str) -> None:
    """One INFO line per reason (the per-problem lines go to DEBUG)."""
    if why not in _SKIP_LOGGED:
        _SKIP_LOGGED.add(why)
        log.info("representative test skipped (TODO D2): %s — first for %s @ %s; further skips logged at DEBUG", why, alias, scene)
    else:
        log.debug("representative test skipped for %s @ %s: %s", alias, scene, why)


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------


def _parse_pattern(text: str) -> Tuple[Tuple[str, int], ...]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        kind, _, n = part.partition(":")
        if kind not in ("same", "other") or not n.isdigit():
            raise argparse.ArgumentTypeError(f"--distractors expects 'same:N,other:M', got {text!r}")
        out.append((kind, int(n)))
    if not out:
        raise argparse.ArgumentTypeError("--distractors must name at least one pool")
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ispatialgym.sampling.ground_isg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--representative-only", dest="mode", action="store_const", const="representative", help="ground the picks of roles/representatives.csv (default)")
    mode.add_argument("--test-only", dest="mode", action="store_const", const="test", help="ground the test = 1 candidates of roles/candidates.csv")
    mode.add_argument("--all", dest="mode", action="store_const", const="all", help="ground every admitted candidate combination")
    ap.set_defaults(mode="representative")
    ap.add_argument("--subtask", action="append", default=None, help="iddl subtask id (repeatable); default: all 21")
    ap.add_argument("--slots", default="representative", help="comma list of representatives.csv slots to ground (representative, variation_texture, variation_geometry, heldout)")
    ap.add_argument("--scene", action="append", default=None, help="restrict the scene blocks to these scenes (repeatable)")
    ap.add_argument("--iddl", default=DEFAULT_IDDL, help="iddl root")
    ap.add_argument("--icl", default=DEFAULT_ICL, help="icl root")
    ap.add_argument("--out-bddl", default=DEFAULT_BDDL_ROOT, help="activity_definitions root the problem<N>.bddl / problems.json are written to")
    ap.add_argument("--customs", default=DEFAULT_CUSTOMS, help="task customs list (created when absent, entries updated in place)")
    ap.add_argument("--full-whitelists", action="store_true", help="whitelist every admissible candidate of the category, not only the pick's model")
    ap.add_argument("--distractors", type=_parse_pattern, default=(("same", 2), ("other", 1)), help="disembedding distractor member pattern (default same:2,other:1)")
    ap.add_argument("--legacy-activity-names", action="store_true", help="write the alignment family under the retired activity name world_alignment")
    ap.add_argument("--strict-symmetric-180", action="store_true", help="spec 2.2 literally: wall targets without a filled symmetric_180 are missing_frame")
    ap.add_argument("--no-representative-test", action="store_true", help="do not call the D2 representative-test hook")
    ap.add_argument("--no-fallbacks", action="store_true", help="do not emit the fallback aliases (owner decision 2026-09-16 B)")
    ap.add_argument("--fallback-scale", type=float, default=DEFAULT_FALLBACK_SCALE,
                    help="uniform target scale of the wall / axial scale fallback when the pilot recorded no scale_hint (default 0.7)")
    ap.add_argument("--scale-hints", default=None,
                    help=f"glob of pilot template_meta.json files whose scale_hint sets the fallback scale per alias (default {DEFAULT_SCALE_HINT_GLOB}; "
                         "'none' = ignore the pilot)")
    ap.add_argument("--scale-hint", action="append", default=None, metavar="ALIAS=SCALE",
                    help="fallback scale of one primary alias (repeatable; wins over the pilot hint and is persisted in the customs entry as "
                         "scale_source = override, so later re-groundings keep it)")
    ap.add_argument("--reseed-generation", default=None, metavar="KEY[,KEY...]|*",
                    help="re-seed these generation keys of every re-grounded entry from generation_defaults.py (a spec number the owner "
                         "changed); '*' = the whole block")
    ap.add_argument("--blacklist", action="append", default=None, metavar="SUBTASK:ROLE:CATEGORY/MODEL=REASON",
                    help="pilot / owner rejection of a model (repeatable): blacklisted as 'pilot <date>: REASON' in every scene block of the "
                         "subtask's aliases (kept by later runs); a primary whose target is left without a model gets the next model of the "
                         "category as its rank-1 fallback")
    ap.add_argument("--dry-run", action="store_true", help="ground and report, write nothing")
    ap.add_argument("--report", default=None, help="also write the report (markdown) to this path; .json alongside")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    slots = tuple(s.strip() for s in args.slots.split(",") if s.strip())
    bad = [s for s in slots if s not in SLOTS]
    if bad:
        print(f"unknown slot(s) {bad}; expected {SLOTS}", file=sys.stderr)
        return 2
    hints: Dict[str, float] = {}
    if not args.no_fallbacks and (args.scale_hints or "").lower() != "none":
        hints = load_scale_hints(args.scale_hints)
        log.info("scale hints from the pilot: %d aliases%s", len(hints), f" ({', '.join(f'{a}={s}' for a, s in sorted(hints.items()))})" if hints else "")
    overrides: Dict[str, float] = {}
    for item in args.scale_hint or []:
        alias, _, value = item.partition("=")
        try:
            s = float(value)
        except ValueError:
            s = -1.0
        if not alias or not (0.0 < s < 1.0):
            print(f"--scale-hint expects ALIAS=SCALE with 0 < SCALE < 1, got {item!r}", file=sys.stderr)
            return 2
        overrides[alias.strip()] = s
    reseed = tuple(k.strip() for k in (args.reseed_generation or "").split(",") if k.strip())
    blacklist: Dict[Tuple[str, str, str, str], str] = {}
    for item in args.blacklist or []:
        spec, _, reason = item.partition("=")
        parts = spec.split(":")
        if len(parts) != 3 or "/" not in parts[2] or not reason.strip():
            print(f"--blacklist expects SUBTASK:ROLE:CATEGORY/MODEL=REASON, got {item!r}", file=sys.stderr)
            return 2
        cat, model = parts[2].split("/", 1)
        blacklist[(parts[0].strip(), parts[1].strip(), cat.strip(), model.strip())] = reason.strip()
    opt = Options(
        mode=args.mode,
        slots=slots,
        scenes=tuple(args.scene) if args.scene else None,
        full_whitelists=args.full_whitelists,
        distractor_pattern=args.distractors,
        legacy_activity_names=args.legacy_activity_names,
        strict_symmetric_180=args.strict_symmetric_180,
        dry_run=args.dry_run,
        representative_test=not args.no_representative_test,
        fallbacks=not args.no_fallbacks,
        fallback_scale=args.fallback_scale,
        scale_hints=hints,
        scale_overrides=overrides,
        reseed_generation=reseed,
        blacklist=blacklist,
    )
    iddls = I.load_all(args.iddl)
    icl = ICL(args.icl)
    customs = Customs.load(args.customs)
    if customs.comment is None:
        customs.comment = ("iSpatialGym task customs list (docs/instance_sampling_spec.md 2.3), written by ground_isg; one entry per grounded "
                           "alias: roles, per-scene bindings / whitelist / blacklist, generation (seeded from sampling/generation_defaults.py), constraints.")
    grounder = Grounder(iddls, icl, customs, bddl_root=args.out_bddl, options=opt, iddl_root=args.iddl)
    report = grounder.run(args.subtask)
    text = report.text()
    print(text)
    if args.report:
        with open(args.report, "w") as f:
            f.write(text)
        with open(os.path.splitext(args.report)[0] + ".json", "w") as f:
            json.dump(report.to_json(), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
