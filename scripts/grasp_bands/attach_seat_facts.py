"""i7-attach (2026-11-07): record the ORACLE's seat axis of every attach pair of the grasp library, in the simulator.

The library's attachment rule (``candidates.attachment_frames``) needs the direction a male part moves onto its partner. The oracle's
attach program (``execute._seat_axis`` -> ``predicates.seat_axis``) takes it from the FEMALE link: its +z, unless that deviates more
than 45 deg from the direction partner hull centroid -> female link (then that direction), fixed in the female link frame. A male
seats with its link frame on the female link frame (``AttachedTo``: position and orientation within tolerance), so in the male link
frame the seat axis is the same local vector ``loc``. This script loads each (male, partner) pair, calls ``predicates.seat_axis``
itself and records, per pair:

* ``female``: the matched female link (``AttachedTo._get_parent_candidates``, every candidate), its pose in the partner's root frame,
  the boundary-point centroid in the root frame (``predicates._boundary_points``), the axis in the world / root frame (partner at the
  identity orientation), ``loc`` (the axis in the female link frame), the branch ("meta link +z" / "centroid->female") and the angle;
* ``male``: the male link's pose in the male root frame (live) and ``seat_axis_male_root`` = R_male_link @ loc: the oracle's seat axis
  in the male's own frame when seated (pointing out of the seat); the male's mating direction is its negative.

    python -u scripts/grasp_bands/attach_seat_facts.py --pairs pairs.json --out facts.json

``pairs.json``: ``[{"id": <library entry id>, "male": {"category", "model", "scale"}, "partners": [{"category", "model", "scale"}]}]``.
One ``SEAT_FACTS {json}`` line per pair; the whole table is written to ``--out`` (schema ``isg-attach-seats/1``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback


def _q(t):
    return [round(float(v), 8) for v in t.tolist()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    pairs = json.load(open(args.pairs))

    # the grasp library's own validation world (robot, slab, cuRobo): objects added to a playing simulator the way it adds them
    from ispatialgym.grasping import validate_tabletop as VT

    t0 = time.time()
    world = VT.TabletopWorld()
    og, th, scene = world.og, world.th, world.scene
    import omnigibson.utils.transform_utils as T
    from omnigibson.objects.dataset_object import DatasetObject
    from omnigibson.object_states import AttachedTo

    from ispatialgym.sampling import predicates as PRED

    print(f"SEAT_FACTS_SETUP {round(time.time() - t0, 1)} s", flush=True)

    def load(spec, name, pos):
        kw = {}
        sc = [float(v) for v in (spec.get("scale") or [1.0, 1.0, 1.0])]
        if any(abs(v - 1.0) > 1e-9 for v in sc):
            kw["scale"] = th.tensor(sc, dtype=th.float32)
        # as the validation world loads its objects (validate_tabletop.TabletopWorld.load): free base, then posed and kept still; every
        # pose below is read relative to the object's live root pose, so a settling step does not matter
        # both parts "attachable", as the attachment task spawns them (sim_utils.spawn extra_abilities: AttachedTo follows the
        # abilities, and the taxonomy does not give it to every model with attachment meta links - teapot mvrhya)
        from ispatialgym.sampling import knowledge as K

        abilities = K.abilities_of_category(spec["category"])
        abilities.update({"attachable": {}})
        o = DatasetObject(name=name, category=spec["category"], model=spec["model"], dataset_name=spec.get("dataset") or "behavior-1k-assets",
                          fixed_base=False, abilities=abilities, **kw)
        scene.add_object(o)
        o.set_position_orientation(th.tensor(pos, dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0]))
        o.keep_still()
        return o

    table = []
    k = 0
    for pr in pairs:
        rec = {"id": pr["id"], "male": dict(pr["male"]), "partners": []}
        k += 1
        male = None
        try:
            male = load(pr["male"], f"seat_m{k}", [0.0, -20.0, 1.0])
            og.sim.step()
            rp, rq = (x.double() for x in male.get_position_orientation())
            if AttachedTo not in male.states:
                rec["male"]["error"] = "no AttachedTo state"
            else:
                rec["male"]["links"] = {}
                for ln, link in male.states[AttachedTo].links.items():
                    cp, cq = (x.double() for x in link.get_position_orientation())
                    p_rel, q_rel = T.relative_pose_transform(cp, cq, rp, rq)
                    rec["male"]["links"][ln] = {"meta_link_id": link.meta_link_id, "sub_id": str(link.meta_link_sub_id), "position_root": _q(p_rel), "orientation_root_xyzw": _q(q_rel)}
            for j, ps in enumerate(pr["partners"]):
                prec = {"category": ps["category"], "model": ps["model"], "scale": ps.get("scale") or [1.0, 1.0, 1.0], "female": []}
                partner = None
                try:
                    partner = load(ps, f"seat_p{k}_{j}", [0.0, 20.0, 1.0])
                    og.sim.step()
                    fp, fq = (x.double() for x in partner.get_position_orientation())
                    if AttachedTo not in partner.states or AttachedTo not in male.states:
                        prec["error"] = "no AttachedTo state"
                    else:
                        cands = male.states[AttachedTo]._get_parent_candidates(partner) or {}
                        pts = PRED._boundary_points(partner).double()
                        cen = pts.mean(dim=0) if len(pts) else None
                        for child_name, parent_names in cands.items():
                            for parent_name in sorted(parent_names):
                                PRED._SEAT_AXIS_LOCAL.clear()
                                link = partner.states[AttachedTo].links[parent_name]
                                lp, lq = (x.double() for x in link.get_position_orientation())
                                axis_w = PRED.seat_axis(male, partner, parent_name).double()
                                loc = PRED._rotate(PRED._qinv(lq), axis_w)
                                z = PRED._rotate(lq, th.tensor([0.0, 0.0, 1.0], dtype=th.float64))
                                f = {"child_link": child_name, "parent_link": parent_name, "meta_link_id": link.meta_link_id, "sub_id": str(link.meta_link_sub_id),
                                     "position_root": _q(T.relative_pose_transform(lp, lq, fp, fq)[0]),
                                     "orientation_root_xyzw": _q(T.relative_pose_transform(lp, lq, fp, fq)[1]),
                                     "axis_root": _q(PRED._rotate(PRED._qinv(fq), axis_w)), "loc": _q(loc / th.linalg.norm(loc)),
                                     "z_root": _q(PRED._rotate(PRED._qinv(fq), z)), "n_boundary_points": int(len(pts))}
                                if cen is not None:
                                    geo = lp - cen
                                    f["centroid_root"] = _q(PRED._rotate(PRED._qinv(fq), cen - fp))
                                    f["centroid_to_female_m"] = round(float(th.linalg.norm(geo)), 6)
                                    if float(th.linalg.norm(geo)) > 0:
                                        g = geo / th.linalg.norm(geo)
                                        f["angle_deg"] = round(math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(g, z)))))), 3)
                                f["branch"] = "meta link +z" if float(th.dot(axis_w, z)) > 1.0 - 1e-9 else "centroid->female"
                                ml = rec["male"].get("links", {}).get(child_name)
                                if ml is not None:
                                    from ispatialgym.grasping import schema as S
                                    import numpy as np

                                    Rm = S.quat_to_mat(ml["orientation_root_xyzw"])
                                    f["seat_axis_male_root"] = [round(float(v), 8) for v in (Rm @ np.asarray(f["loc"], dtype=float))]
                                prec["female"].append(f)
                        if not prec["female"]:
                            prec["error"] = "no matching female link"
                except Exception as e:  # noqa: BLE001
                    prec["error"] = repr(e)
                    prec["tb"] = traceback.format_exc()[-2500:]
                finally:
                    if partner is not None:
                        scene.remove_object(partner)
                        og.sim.step()
                rec["partners"].append(prec)
        except Exception as e:  # noqa: BLE001
            rec["error"] = repr(e)
            rec["tb"] = traceback.format_exc()[-2500:]
        finally:
            if male is not None:
                scene.remove_object(male)
                og.sim.step()
        print("SEAT_FACTS " + json.dumps(rec), flush=True)
        table.append(rec)
        with open(args.out, "w") as fh:
            json.dump({"schema": "isg-attach-seats/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "scripts/grasp_bands/attach_seat_facts.py "
                       "(predicates.seat_axis on the loaded pair)", "pairs": table}, fh, indent=1)
    print(f"SEAT_FACTS_DONE {len(table)} pairs {round(time.time() - t0, 1)} s", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
