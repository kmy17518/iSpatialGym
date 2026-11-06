import json, os, hashlib, glob, sys

from typing import *

def fast_value(n, d): return d

def _tro_object_names(tro_path: str) -> List[str]:
    """The scene objects an instance's TRO sets (restore rewrites their state): the ``ispatialgym.object_bindings`` scene object names of
    its BDDL scope (task objects, support, floor) plus any top-level entry name already in scene-object form."""
    with open(tro_path) as f:
        raw = json.load(f)
    names = {str(b.get("scene_object_name")) for b in ((raw.get("ispatialgym") or {}).get("object_bindings") or {}).values()
             if isinstance(b, dict) and b.get("scene_object_name")}
    names |= {k for k, v in raw.items() if isinstance(v, dict) and "root_link" in v}
    return sorted(names)

def _scene_leaves(template_dir: str, tro_path: str) -> Dict[str, Any]:
    """The template scene as the instance's world keeps it: ``template.json`` flattened to ``{path: value}`` without the entities the
    instance's restore rewrites (the TRO's bound objects, the robot), the robot poses of the task metadata, the file's own path, the
    ``is_asleep`` flags, the linear / angular velocities (PhysX state: an object still swaying when one copy was saved) and the cached
    ``non_kin/AABB`` bounding boxes (derived from the pose and the geometry; they differ between saves of an unmoved object)."""
    names = set(_tro_object_names(tro_path))
    with open(os.path.join(template_dir, "template.json")) as f:
        doc = json.load(f)
    for sect in ((((doc.get("state") or {}).get("registry") or {}).get("object_registry") or {}), (((doc.get("objects_info") or {}).get("init_info")) or {})):
        for n in list(sect):
            if n in names or n.startswith("robot"):
                sect.pop(n, None)
    ((doc.get("metadata") or {}).get("task") or {}).pop("robot_poses", None)
    ((doc.get("init_info") or {}).get("args") or {}).pop("scene_file", None)
    out: Dict[str, Any] = {}

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{p}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{p}[{i}]")
        elif not (p.endswith("/is_asleep") or "/lin_vel[" in p or "/ang_vel[" in p or "/non_kin/AABB/" in p):
            # PhysX state and the cached bounding box (a derived value: equal poses, boxes 1.3 cm apart between an eval and a train save);
            # poses and joints are compared
            out[p] = x

    walk(doc, "")
    return out

def template_key(template_dir: str, tmeta: dict, tro_path: Optional[str] = None, robot_name: Optional[str] = None) -> str:
    """2026-10-31 train-fast (COLLECTOR_PREBUILD): the template's identity independent of where its copy lives - template id, the activity's
    BDDL problem and the meta fields the world build reads (scene, load_room_instances, problem, scene_variant). Two copies with one key
    are the same template; whether their saved scenes agree is :func:`template_compatible`'s question."""
    h = hashlib.sha256()
    activity, did = tmeta["problem"].rsplit("-", 1)
    bddl = os.path.join(template_dir, "activity_definitions", activity, f"problem{did}.bddl")
    if os.path.exists(bddl):
        h.update(open(bddl, "rb").read())
    h.update(json.dumps({k: tmeta.get(k) for k in ("template_id", "scene", "load_room_instances", "problem", "scene_variant")}, sort_keys=True,
                        default=str).encode())
    return h.hexdigest()[:24]

def template_compatible(dir_a: str, tro_a: str, dir_b: str, tro_b: str, tol: Optional[float] = None) -> Tuple[bool, str]:
    """2026-10-31 train-fast (COLLECTOR_PREBUILD): may a world built from template copy ``dir_a`` (via its instance ``tro_a``) run the
    instances of copy ``dir_b``? Yes when their scenes (:func:`_scene_leaves`) have the same entries, equal non-numeric values and numbers
    within ``tol`` (``ISPATIALGYM_FAST_TEMPLATE_TOL_M``, 5 mm / 5e-3: two saves of one template settle untouched objects ~1e-6 apart; a
    scene object pushed 1-2 cm or a scene variant that moved chairs away is a different world). Returns (ok, why)."""
    tol = float(fast_value("TEMPLATE_TOL_M", 0.005)) if tol is None else float(tol)
    A, B = _scene_leaves(dir_a, tro_a), _scene_leaves(dir_b, tro_b)
    if set(A) != set(B):
        extra = sorted(set(A) ^ set(B))
        return False, f"{len(extra)} scene entries only in one copy (e.g. {extra[0][:120]})"
    worst, where = 0.0, ""
    for k, va in A.items():
        vb = B[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
            d = abs(float(va) - float(vb))
            if d > worst:
                worst, where = d, k
        elif va != vb:
            return False, f"{k[:120]}: {str(va)[:40]} vs {str(vb)[:40]}"
    if worst > tol:
        return False, f"{where[:120]} differs by {worst:.4f} (> {tol})"
    return True, f"{len(A)} scene entries agree within {worst:.2g}"


F = "/vision/u/minyeong/projects/isg_home/trainfast"; T85 = "/vision/u/minyeong/projects/isg_home/train85"
for key in sys.argv[1:]:
    jobs = sorted(glob.glob(f"{F}/out/*/{key}/g*")) + sorted(glob.glob(f"{T85}/out/{key}/g*"))
    env = dict(l.strip().split("=", 1) for l in open(sorted(glob.glob(f"{F}/out/*/{key}/g*"))[0] + "/job.env") if "=" in l)
    counted = env["COUNTED_TEMPLATE"].strip("'")
    copies = []
    for j in jobs:
        tros = sorted(glob.glob(j + "/inst/*/*/instances/*-tro_state.json"))
        if tros:
            copies.append((j.split("/out/")[-1], os.path.dirname(os.path.dirname(tros[0])), tros[0]))
    ctro = sorted(glob.glob(counted + "/instances/*-tro_state.json"))[0]
    vs_counted = [template_compatible(counted, ctro, td, tro)[0] for _, td, tro in copies]
    ref = copies[0]
    vs_job, whys = [], []
    for name, td, tro in copies[1:]:
        ok, why = template_compatible(ref[1], ref[2], td, tro)
        vs_job.append(ok)
        if not ok: whys.append(f"{name}: {why[:90]}")
    print(f"{key}: {len(copies)} job copies; vs counted {sum(vs_counted)}/{len(copies)}; vs another job copy {sum(vs_job)}/{len(vs_job)}" + (f"; refusal: {whys[0]}" if whys else ""))

