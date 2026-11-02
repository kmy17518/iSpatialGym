"""Stack probe (campaign vec_next, 2026-10-30, the lid-on-teapot rocking): build an N-clone world of an instance list's first instance,
restore the TRO into every slot, optionally TELEPORT the target onto the partner's attachment point (``--stack``: the end state of the
attachment demos without the robot's trajectory), hold the robot for --steps control steps and print, every --every steps and per slot,
whether the task's physical objects are asleep, their speeds and the partner's tilt from its restored orientation. Knobs (all applied to
the task objects of every slot after the restore, before stepping):

  --sleep-threshold T     physxRigidBody / physxArticulation sleepThreshold (mass-normalised KE; stock 5e-5, campaign default 0.005)
  --stabilization S       physxScene:enableStabilization + per-object stabilizationThreshold S (sim stop / play)
  --restitution R         bind a physics material (friction kept at --friction, default 0.5 / 0.5) with restitution R to every collision
                          mesh of the task objects (sim stop / play)
  --angular-damping D     physxRigidBody:angularDamping on every link (stock 0.05)
  --linear-damping D      physxRigidBody:linearDamping on every link (stock 0)
  --pos-iters N / --vel-iters N   solver iteration counts of the task objects
  --contact-offset C      physxCollision:contactOffset of the task objects' collision meshes

    python scripts/vec_eval/probe_stack.py --instances <alias.json> --num-envs 1 --steps 240 --every 30 --stack --sleep-threshold 0.005
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import omnigibson as og  # noqa: F401
import torch as th

from ispatialgym.oracle.vec_loader import VecInstanceLoader  # noqa: E402


def _quat_angle(q0: th.Tensor, q1: th.Tensor) -> float:
    d = float(th.abs(th.dot(q0, q1)).clamp(0.0, 1.0))
    return 2.0 * math.acos(d)


def _quat_rotate(q: th.Tensor, v: th.Tensor) -> th.Tensor:
    u, s = q[:3], q[3]
    return 2.0 * th.dot(u, v) * u + (s * s - th.dot(u, u)) * v + 2.0 * s * th.cross(u, v, dim=0)


def _contacts(obj, scene_idx):
    """Which other objects the links of @obj touch right now ({object prim name: pair count}) from OmniGibson's contact view."""
    from omnigibson.utils.usd_utils import RigidContactAPI

    paths = [link.prim_path for link in obj.links.values()]
    try:
        pairs = RigidContactAPI.get_contact_pairs(scene_idx=scene_idx, query_set=paths, with_set=None, current_only=True)
    except Exception as ex:  # noqa: BLE001
        return f"?{type(ex).__name__}:{str(ex)[:60]}"
    others = {}
    for a, b in pairs:
        if b in paths:
            continue
        key = b.split("/")[-2] if b.count("/") >= 2 else b
        others[key] = others.get(key, 0) + 1
    return others


def _objects_of(loader: VecInstanceLoader, i: int):
    return {inst: obj for inst, obj in loader.physical_of(i).items() if not (obj.fixed_base or obj.kinematic_only)}


def _describe(i: int, inst: str, obj) -> None:
    links = {}
    for name, link in obj.links.items():
        try:
            links[name] = round(float(link.mass), 5)
        except Exception:  # noqa: BLE001
            links[name] = None
    mats = {}
    for name, link in obj.links.items():
        for cname, cm in getattr(link, "collision_meshes", {}).items():
            try:
                m = cm.get_applied_physics_material()
                if m is None:
                    mats[f"{name}/{cname}"] = "unbound(PhysX default)"
                else:
                    mats[f"{name}/{cname}"] = f"{m.prim_path} sf={m.get_static_friction()} df={m.get_dynamic_friction()} rest={m.get_restitution()}"
            except Exception as ex:  # noqa: BLE001
                mats[f"{name}/{cname}"] = f"?{type(ex).__name__}"
            break  # one collision mesh per link is enough
    extra = {}
    for attr in ("physxRigidBody:angularDamping", "physxRigidBody:linearDamping", "physxRigidBody:solverPositionIterationCount",
                 "physxRigidBody:solverVelocityIterationCount", "physxRigidBody:maxDepenetrationVelocity", "physxRigidBody:sleepThreshold",
                 "physxRigidBody:stabilizationThreshold"):
        try:
            extra[attr.split(":")[1]] = obj.root_link.get_attribute(attr)
        except Exception:  # noqa: BLE001
            extra[attr.split(":")[1]] = "n/a"
    physx_masses = None  # what PhysX really uses (a meta link with physics:mass 0 is not massless in PhysX)
    for getter in ("get_body_masses", "get_masses"):
        for view in (getattr(obj, "_articulation_view", None), getattr(obj.root_link, "_rigid_prim_view", None)):
            fn = getattr(view, getter, None) if view is not None else None
            if fn is None:
                pv = getattr(view, "_physics_view", None) if view is not None else None
                fn = getattr(pv, getter, None) if pv is not None else None
            if fn is not None and physx_masses is None:
                try:
                    physx_masses = [round(float(x), 6) for x in th.as_tensor(fn()).flatten().tolist()]
                except Exception as ex:  # noqa: BLE001
                    physx_masses = f"?{type(ex).__name__}"
    coll = {}
    try:
        cm = next(iter(obj.root_link.collision_meshes.values()))
        for attr in ("physxCollision:contactOffset", "physxCollision:restOffset", "physics:approximation"):
            try:
                coll[attr.split(":")[1]] = cm.get_attribute(attr)
            except Exception:  # noqa: BLE001
                coll[attr.split(":")[1]] = "n/a"
    except Exception:  # noqa: BLE001
        pass
    print(f"PROBE slot {i} {inst}: mass={float(obj.mass):.4f} articulated={obj.articulated} n_dof={obj.n_dof} n_joints={obj.n_joints} "
          f"n_fixed_joints={getattr(obj, 'n_fixed_joints', '?')} links={links} physx_masses={physx_masses} sleep_thr={obj.sleep_threshold} stab_thr={obj.stabilization_threshold} "
          f"root_attrs={extra} coll={coll} materials={mats}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--scene-offsets", default=None)
    p.add_argument("--layout", default="grid-centered")
    p.add_argument("--scene-margin", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--stack", action="store_true", help="teleport the target onto the partner's attachment point after the restore")
    p.add_argument("--no-attach", action="store_true", help="with --stack: do not create the attachment joint (let the lid rest on the rim)")
    p.add_argument("--no-stack", action="store_true", help="overrides --stack: leave the objects where the restore put them")
    p.add_argument("--lift", type=float, default=0.0, help="with --stack: teleport this many mm above the aligned pose (along the partner's z)")
    p.add_argument("--stock-attach", action="store_true", help="ISG_ATTACH_ROOT_JOINT=0: the stock meta-link attachment joint (the pair keeps colliding)")
    p.add_argument("--target", default=None, help="instance name of the object to teleport (default: the first 'target_*' object)")
    p.add_argument("--partner", default=None, help="instance name of the object it attaches to (default: the first 'partner_*' object)")
    p.add_argument("--sleep-threshold", type=float, default=None)
    p.add_argument("--stabilization", type=float, default=None)
    p.add_argument("--stabilization-at-launch", type=float, default=None,
                   help="enable physxScene:enableStabilization when the simulator applies its physics settings (before the first play; the "
                        "stop / play route segfaults) and set this stabilizationThreshold on the task objects after the restore")
    p.add_argument("--restitution", type=float, default=None)
    p.add_argument("--friction", type=float, default=0.5)
    p.add_argument("--angular-damping", type=float, default=None)
    p.add_argument("--linear-damping", type=float, default=None)
    p.add_argument("--pos-iters", type=int, default=None)
    p.add_argument("--vel-iters", type=int, default=None)
    p.add_argument("--contact-offset", type=float, default=None)
    p.add_argument("--force-sleep-at", type=int, default=None, help="call obj.sleep() on every task object at this control step (do they stay asleep?)")
    args = p.parse_args()
    if args.no_stack:
        args.stack = False
    os.environ["ISG_ATTACH_ROOT_JOINT"] = "0" if args.stock_attach else "1"
    if args.scene_offsets:
        os.environ["ISG_SCENE_OFFSETS"] = args.scene_offsets
    with open(args.instances) as f:
        spec = json.load(f)
    tro = spec["instances"][0]["tro"]
    pov = {}
    if args.sleep_threshold is not None:
        pov["rigid_sleep_threshold"] = args.sleep_threshold
    if args.stabilization_at_launch is not None:
        # the Simulator class only exists after og.launch(); Environment.load runs after the launch and before the first play
        import omnigibson.envs.env_base as _eb

        _orig_load = _eb.Environment.load

        def _load_with_stabilization(self, *a, **k):
            og.sim._physics_context.enable_stablization(True)  # Isaac's spelling; a USD edit on the stopped scene
            print("PROBE physics settings: enableStabilization set before the first play", flush=True)
            return _orig_load(self, *a, **k)

        _eb.Environment.load = _load_with_stabilization
    loader = VecInstanceLoader(num_envs=args.num_envs, behavior_output=True, tiled_rendering=False, drop_depth=True, external_camera_hw=None,
                               layout=args.layout, scene_margin=args.scene_margin, physics_overrides=pov or None)
    env = loader.build_env(tro)
    import omnigibson.lazy as lazy

    stage = og.sim.stage
    scene_prim = next((prim for prim in stage.Traverse() if prim.GetTypeName() == "PhysicsScene"), None)
    if scene_prim is not None:
        for name in ("physxScene:enableStabilization", "physxScene:solverType", "physxScene:timeStepsPerSecond", "physxScene:bounceThreshold",
                     "physxScene:frictionOffsetThreshold", "physxScene:frictionCorrelationDistance", "physxScene:enableEnhancedDeterminism"):
            a = scene_prim.GetAttribute(name)
            print(f"PROBE scene attr {name} = {a.Get() if a and a.IsValid() else 'n/a'}", flush=True)
    obs, info = loader.restore_batch([tro] * args.num_envs)
    objs = {i: _objects_of(loader, i) for i in range(args.num_envs)}
    for i in range(args.num_envs):
        for inst, obj in objs[i].items():
            try:
                _describe(i, inst, obj)
            except Exception as ex:  # noqa: BLE001
                print(f"PROBE slot {i} {inst}: describe failed ({ex!r})", flush=True)

    # ---- knobs needing a stopped simulator
    need_stop = args.stabilization is not None   # restitution is bound live (a stop / play here segfaults: K2 / lane-B 'stab')
    if need_stop:
        og.sim.stop()
        if args.stabilization is not None and scene_prim is not None:
            with og.sim.editing_usd():
                a = scene_prim.GetAttribute("physxScene:enableStabilization") or scene_prim.CreateAttribute("physxScene:enableStabilization", lazy.pxr.Sdf.ValueTypeNames.Bool)
                a.Set(True)
        for i in range(args.num_envs):
            for inst, obj in objs[i].items():
                if args.stabilization is not None:
                    obj.stabilization_threshold = args.stabilization
        og.sim.play()
        og.sim.update_handles()
    if args.restitution is not None:
        for i in range(args.num_envs):
            for inst, obj in objs[i].items():
                mat = lazy.isaacsim.core.api.materials.PhysicsMaterial(prim_path=f"/World/isg_probe_mat_{i}_{inst.replace('.', '_')}", name=f"probe_mat_{i}_{inst}",
                                                                       static_friction=args.friction, dynamic_friction=args.friction, restitution=args.restitution)
                for link in obj.links.values():
                    for cm in getattr(link, "collision_meshes", {}).values():
                        cm.apply_physics_material(mat)
        print(f"PROBE restitution material bound live: {args.restitution} (friction {args.friction})", flush=True)
    # ---- live knobs (USD edits PhysX picks up while playing)
    for i in range(args.num_envs):
        for inst, obj in objs[i].items():
            if args.stabilization_at_launch is not None:
                obj.stabilization_threshold = args.stabilization_at_launch
            with og.sim.editing_usd():
                for link in obj.links.values():
                    if args.angular_damping is not None:
                        link.set_attribute("physxRigidBody:angularDamping", float(args.angular_damping))
                    if args.linear_damping is not None:
                        link.set_attribute("physxRigidBody:linearDamping", float(args.linear_damping))
                    if args.contact_offset is not None:
                        for cm in getattr(link, "collision_meshes", {}).values():
                            cm.set_attribute("physxCollision:contactOffset", float(args.contact_offset))
            if args.pos_iters is not None:
                obj.solver_position_iteration_count = int(args.pos_iters)
            if args.vel_iters is not None:
                obj.solver_velocity_iteration_count = int(args.vel_iters)
    print(f"PROBE attach: ISG_ATTACH_ROOT_JOINT={os.environ['ISG_ATTACH_ROOT_JOINT']}", flush=True)
    print(f"PROBE knobs: sleep_threshold={args.sleep_threshold} stabilization={args.stabilization} restitution={args.restitution} angular_damping={args.angular_damping} "
          f"linear_damping={args.linear_damping} pos_iters={args.pos_iters} vel_iters={args.vel_iters} contact_offset={args.contact_offset}", flush=True)
    for i in range(args.num_envs):
        for inst, obj in objs[i].items():
            try:
                _describe(i, inst, obj)
            except Exception as ex:  # noqa: BLE001
                print(f"PROBE slot {i} {inst}: describe failed ({ex!r})", flush=True)

    # ---- the stack: the target teleported onto the partner's attachment meta link
    pairs = {}
    if args.stack:
        from omnigibson.object_states import AttachedTo

        for i in range(args.num_envs):
            names = list(objs[i])
            tname = args.target or next((n for n, o in objs[i].items() if o.name.startswith("target_")), names[0])
            pname = args.partner or next((n for n, o in objs[i].items() if o.name.startswith("partner_")), names[-1])
            target, partner = objs[i][tname], objs[i][pname]
            # attachment meta links: the partner's F link and the target's M link (positions in their base frames)
            def _meta(obj, kind):
                for lname, link in obj.links.items():
                    if "attachment" in lname and f"{kind}_" in lname:
                        return link
                return None
            f_link, m_link = _meta(partner, "F"), _meta(target, "M")
            p_pos, p_quat = partner.get_position_orientation()
            if f_link is not None and m_link is not None:
                f_pos, _ = f_link.get_position_orientation()
                m_pos, _ = m_link.get_position_orientation()
                t_pos, t_quat_now = target.get_position_orientation()
                # move the target so its M link lands on the partner's F link, orientation = the partner's: the M offset in the
                # target's own frame, re-expressed under the partner's orientation (+2 mm so the rim contact starts from above)
                m_off_world = m_pos - t_pos
                m_off_local = _quat_rotate(th.tensor([-t_quat_now[0], -t_quat_now[1], -t_quat_now[2], t_quat_now[3]]), m_off_world)
                new_pos = f_pos - _quat_rotate(p_quat, m_off_local)  # M on F exactly: the pose AttachedTo._attach aligns the child to
                if args.lift:
                    new_pos = new_pos + _quat_rotate(p_quat, th.tensor([0.0, 0.0, args.lift / 1000.0]))
                target.set_position_orientation(new_pos, p_quat)
                target.set_linear_velocity(th.zeros(3))
                target.set_angular_velocity(th.zeros(3))
                print(f"PROBE slot {i} stack: {tname} teleported onto {pname} (F {f_pos.tolist()} / M-offset {m_off_local.tolist()}) -> {new_pos.tolist()}", flush=True)
                if not args.no_attach and AttachedTo in target.states:
                    # one control step first: a teleport moves the root link now but the meta links only at the next physics step, and
                    # _attach aligns on the meta links' poses (forcing it right after the teleport yanked the lid metres away)
                    env.step(th.stack([loader.hold_action(k) for k in range(args.num_envs)]))
                    st = target.states[AttachedTo]
                    if st.parent is None:
                        ok = st.set_value(partner, True)
                        print(f"PROBE slot {i} stack: AttachedTo.set_value({pname}, True) -> {ok}; parent={getattr(st.parent, 'name', None)}", flush=True)
                    else:
                        print(f"PROBE slot {i} stack: auto-attached during the first step to {st.parent.name}", flush=True)
            else:
                print(f"PROBE slot {i} stack: no attachment meta links found (F={f_link is not None}, M={m_link is not None}); skipping", flush=True)
            pairs[i] = (tname, pname)
            if AttachedTo in target.states:
                pairs[i] = (tname, pname, target.states[AttachedTo])
    q0 = {(i, inst): obj.get_position_orientation()[1].clone() for i in range(args.num_envs) for inst, obj in objs[i].items()}
    p0 = {(i, inst): obj.get_position_orientation()[0].clone() for i in range(args.num_envs) for inst, obj in objs[i].items()}
    actions = th.stack([loader.hold_action(i) for i in range(args.num_envs)])
    first_asleep = {}
    max_tilt = {}
    last_q = dict(q0)
    step_rot_max = {}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        env.step(actions)
        if args.force_sleep_at is not None and step == args.force_sleep_at:
            for i in range(args.num_envs):
                for inst, obj in objs[i].items():
                    try:
                        obj.sleep()
                    except Exception as ex:  # noqa: BLE001
                        print(f"PROBE force-sleep {inst}: {ex!r}", flush=True)
            print(f"PROBE force-sleep applied at step {step}", flush=True)
        for i in range(args.num_envs):
            for inst, obj in objs[i].items():
                q = obj.get_position_orientation()[1]
                tilt = _quat_angle(q0[(i, inst)], q)
                max_tilt[(i, inst)] = max(max_tilt.get((i, inst), 0.0), tilt)
                dstep = _quat_angle(last_q[(i, inst)], q)
                if step > args.steps - 60:
                    step_rot_max[(i, inst)] = max(step_rot_max.get((i, inst), 0.0), dstep)
                last_q[(i, inst)] = q.clone()
                try:
                    asleep = bool(obj.is_asleep) if not callable(obj.is_asleep) else bool(obj.is_asleep())
                except Exception:  # noqa: BLE001
                    asleep = False
                if asleep and (i, inst) not in first_asleep:
                    first_asleep[(i, inst)] = step
        if step % args.every == 0 or step == 1:
            for i in range(args.num_envs):
                parts = []
                for inst, obj in objs[i].items():
                    try:
                        v = float(th.linalg.norm(obj.get_linear_velocity())) * 1000
                        w = float(th.linalg.norm(obj.get_angular_velocity())) * 1000
                        asleep = bool(obj.is_asleep) if not callable(obj.is_asleep) else bool(obj.is_asleep())
                    except Exception as ex:  # noqa: BLE001
                        v, w, asleep = float("nan"), float("nan"), f"?{ex!r}"[:40]
                    pos, q = obj.get_position_orientation()
                    tilt = _quat_angle(q0[(i, inst)], q) * 1000
                    dz = float(pos[2] - p0[(i, inst)][2]) * 1000
                    parts.append(f"{inst.split('.')[0]}: asleep={asleep} v={v:.2f}mm/s w={w:.2f}mrad/s tilt={tilt:.2f}mrad dz={dz:.2f}mm")
                att = ""
                if i in pairs and len(pairs[i]) == 3:
                    st = pairs[i][2]
                    att = f" | attached={st.parent.name if st.parent is not None else None}"
                scene_idx = env.scenes[i].idx if hasattr(env, "scenes") else i
                touch = " | contacts " + " ".join(f"{inst.split('.')[0]}={_contacts(obj, scene_idx)}" for inst, obj in objs[i].items())
                print(f"PROBE step {step:4d} slot {i} | " + " | ".join(parts) + att + touch, flush=True)
    for i in range(args.num_envs):
        for inst, obj in objs[i].items():
            try:
                asleep = bool(obj.is_asleep) if not callable(obj.is_asleep) else bool(obj.is_asleep())
            except Exception:  # noqa: BLE001
                asleep = None
            print(f"PROBE summary slot {i} {inst}: asleep_end={asleep} first_asleep_step={first_asleep.get((i, inst))} max_tilt={max_tilt[(i, inst)] * 1000:.2f}mrad "
                  f"last60_step_rot_max={step_rot_max.get((i, inst), 0.0) * 1000:.3f}mrad", flush=True)
    print(f"PROBE done {args.steps} steps in {time.time() - t0:.1f} s", flush=True)
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
