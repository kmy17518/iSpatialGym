"""Diagnostic renders for the handle review: per model an overview (closed) and per openable joint a zoom (closed, with lettered
sphere markers at the inferred handle candidates) and an 'opened' view (joint at 60 % of its range). Markers: A, B, C = protruding
handle candidates (red, green, blue), F = edge fallback (yellow), X = flush handle (grey). Output artic/render/<key>__*.png and a
composite strip artic/strips/<key>.png."""
import json, os, sys, glob, time, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th
from PIL import Image, ImageDraw, ImageFont
import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject

OUT = "/tmp/glib/artic"; os.makedirs(f"{OUT}/render", exist_ok=True); os.makedirs(f"{OUT}/strips", exist_ok=True)
COLORS = {"A": (1.0, 0.1, 0.1), "B": (0.1, 0.8, 0.1), "C": (0.1, 0.3, 1.0), "D": (1.0, 0.5, 0.0), "F": (1.0, 0.9, 0.1), "X": (0.5, 0.5, 0.5)}
files = sorted(glob.glob(f"{OUT}/handles/*.json"))
shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
files = [f for i, f in enumerate(files) if i % nshard == shard]
env = og.Environment(configs={"scene": {"type": "Scene"}}); scene = env.scene
for _ in range(3): og.sim.step()
cam = og.sim.viewer_camera; cam.set_position_orientation(th.tensor([2.0, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
for _ in range(6): og.sim.render()
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception: FONT = ImageFont.load_default()


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f); r = th.linalg.cross(f, up)
    if th.linalg.norm(r) < 1e-3: r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
    r = r / th.linalg.norm(r); u = th.linalg.cross(r, f); return T.mat2quat(th.stack([r, u, -f], dim=1))


def grab(cam_pos, target, path, size=420):
    cam.set_position_orientation(th.as_tensor(cam_pos, dtype=th.float32), look_at(th.as_tensor(cam_pos, dtype=th.float32), th.as_tensor(target, dtype=th.float32)))
    for _ in range(3): og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy(); h, w = rgb.shape[:2]; s = min(h, w)
    Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)


# marker pool (created once; stop/play resets object poses, so never stop the sim between placing the object and shooting)
POOL = {}
og.sim.stop()
for letter, col in COLORS.items():
    POOL[letter] = []
    for k in range(6):
        m = PrimitiveObject(name=f"mk_{letter}_{k}", primitive_type="Sphere", radius=0.016, visual_only=True, rgba=[*col, 1.0])
        scene.add_object(m); POOL[letter].append(m)
og.sim.play(); og.sim.step()
FAR = th.tensor([50.0, 50.0, -5.0])


def add_markers(items, root_pos, radius=None):
    """items: list of (letter, position in object frame). Unused pool spheres are parked far away."""
    used = {k: 0 for k in POOL}
    for letter, p in items:
        k = used.get(letter, 0)
        if k < len(POOL[letter]):
            POOL[letter][k].set_position_orientation(th.as_tensor(root_pos + np.array(p), dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0])); used[letter] = k + 1
    for letter, ms in POOL.items():
        for m in ms[used[letter]:]: m.set_position_orientation(FAR, th.tensor([0.0, 0.0, 0.0, 1.0]))
    og.sim.step()


def clear_markers():
    for ms in POOL.values():
        for m in ms: m.set_position_orientation(FAR, th.tensor([0.0, 0.0, 0.0, 1.0]))


done = 0; t0 = time.time()
for f in files:
    H = json.load(open(f)); key = H["key"]
    if os.path.exists(f"{OUT}/strips/{key}.png"): continue
    obj = None
    try:
        og.sim.stop(); obj = DatasetObject(name=f"o_{key}", category=H["category"], model=H["model"], fixed_base=True, include_default_states=False)
        scene.add_object(obj); og.sim.play(); og.sim.step()
        alo, ahi = obj.aabb; rp0, _ = obj.root_link.get_position_orientation()
        obj.set_position_orientation(th.tensor([0.0, 0.0, float(rp0[2] - alo[2])]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        inv = json.load(open(f"{OUT}/inv/{key}.json"))
        q_closed = th.tensor([j["closed_q"] for j in inv["joints"]] + [0.0] * 0) if False else None
        names = list(obj.joints.keys()); q = obj.get_joint_positions().clone()
        for j in H["joints"]:
            if j["joint"] in names: q[names.index(j["joint"])] = j["closed_q"]
        obj.set_joint_positions(q); obj.keep_still(); og.sim.step()
        rp, _ = obj.root_link.get_position_orientation(); root = rp.numpy().astype(np.float64); alo, ahi = obj.aabb; ext = (ahi - alo).numpy(); c = ((alo + ahi) / 2).numpy()
        n0 = np.array(H["joints"][0].get("outward_normal", [1, 0, 0])) if H["joints"] else np.array([1.0, 0, 0])
        if abs(n0[2]) > 0.7: n0 = np.array([1.0, 0, 0])
        side = np.cross(n0, [0, 0, 1.0]); d = 1.6 * float(np.linalg.norm(ext)) + 0.3
        # markers for all joints
        items = []
        for j in H["joints"]:
            for k, h in enumerate(j.get("handles", [])[:4]): items.append(("ABCD"[k], h["contact"]))
            if j.get("fallback") and j.get("status") != "handle": items.append(("F", j["fallback"]["contact"]))
            for x in j.get("flush_handles", [])[:2]: items.append(("X", x["center"]))
        add_markers(items, root, radius=max(0.012, 0.02 * float(np.linalg.norm(ext))))
        views = []
        grab(c + n0 * d * 0.8 + side * d * 0.45 + np.array([0, 0, d * 0.35]), c, f"{OUT}/render/{key}__overview.png"); views.append(("overview (closed)", f"{OUT}/render/{key}__overview.png"))
        for ji, j in enumerate(H["joints"]):
            jitems = [("ABCD"[k], h["contact"]) for k, h in enumerate(j.get("handles", [])[:4])] + ([("F", j["fallback"]["contact"])] if j.get("fallback") and j.get("status") != "handle" else []) + [("X", x["center"]) for x in j.get("flush_handles", [])[:2]]
            add_markers(jitems, root)                                                            # zoom shows only this joint's markers
            lo, hi = np.array(j["link_bbox_min"]) + root, np.array(j["link_bbox_max"]) + root; lc = (lo + hi) / 2; lext = float(np.linalg.norm(hi - lo)); n = np.array(j.get("outward_normal", n0))
            sd = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.7 else np.array([1.0, 0, 0])
            dz = 1.3 * lext + 0.15
            grab(lc + n * dz * 0.85 + sd * dz * 0.35 + np.array([0, 0, dz * 0.3]), lc, f"{OUT}/render/{key}__{ji}_zoom.png"); views.append((f"{j['joint']} ({j.get('kind')}, {j['status']})", f"{OUT}/render/{key}__{ji}_zoom.png"))
        # opened view: every openable joint at 60 % of its range
        add_markers(items, root); q2 = q.clone()
        for j in H["joints"]:
            if j["joint"] in names:
                i = names.index(j["joint"]); q2[i] = j["closed_q"] + j["open_sign"] * 0.6 * (j["upper"] - j["lower"])
        obj.set_joint_positions(q2); obj.keep_still(); og.sim.step()
        grab(c + n0 * d * 0.8 + side * d * 0.45 + np.array([0, 0, d * 0.35]), c, f"{OUT}/render/{key}__open.png"); views.append(("opened 60 % (markers stay at closed positions)", f"{OUT}/render/{key}__open.png"))
        obj.set_joint_positions(q); obj.keep_still(); og.sim.step()
        # composite strip
        S = 420; cols = min(len(views), 4); rows = (len(views) + cols - 1) // cols
        strip = Image.new("RGB", (cols * S, rows * (S + 26) + 30), (255, 255, 255)); dr = ImageDraw.Draw(strip)
        hdr = f"{H['category']} / {H['model']}   aabb {np.round(ext, 2).tolist()} m   joints: " + ", ".join(f"{j['joint']}={j.get('kind')}/{j['status']}" for j in H["joints"])
        dr.text((6, 5), hdr[:180], fill=(0, 0, 0), font=FONT)
        for k, (title, path) in enumerate(views):
            im = Image.open(path).resize((S, S)); x, y = (k % cols) * S, 30 + (k // cols) * (S + 26)
            strip.paste(im, (x, y + 26)); dr.text((x + 4, y + 4), title[:60], fill=(0, 0, 0), font=FONT)
        legend = "markers: A/B/C/D = protruding handle candidates (red/green/blue/orange), F = edge fallback (yellow), X = flush handle (grey)"
        dr.text((6, strip.height - 0), "", fill=(0, 0, 0)); strip = strip.crop((0, 0, strip.width, strip.height)); dr.text((6, rows * (S + 26) + 30 - 0), legend, fill=(0, 0, 0), font=FONT) if False else None
        strip2 = Image.new("RGB", (strip.width, strip.height + 28), (255, 255, 255)); strip2.paste(strip, (0, 0)); ImageDraw.Draw(strip2).text((6, strip.height + 4), legend, fill=(0, 0, 0), font=FONT)
        strip2.save(f"{OUT}/strips/{key}.png")
        done += 1; print(f"[{done}/{len(files)}] {key}: {len(H['joints'])} joints, {len(items)} markers", flush=True)
    except Exception as e:
        print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc()
    finally:
        clear_markers()
        try:
            og.sim.stop()
            if obj is not None: scene.remove_object(obj)
            og.sim.play(); og.sim.step()
        except Exception as e: print("cleanup failed", key, repr(e))
print(f"RENDER DONE {done} in {time.time() - t0:.0f}s", flush=True); og.shutdown()
