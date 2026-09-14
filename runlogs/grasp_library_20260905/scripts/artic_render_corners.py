"""Second-round renders for the no-handle joints: per joint a strip with (1) a zoom on the movable part from the outer side with the
four corners of its OUTER face marked (TL red, TR green, BL blue, BR orange; 'left/right' as seen in this view) plus a grey
sphere at the face centre, (2) a view from above/side, (3) the part opened 60 %. Output artic/strips2/<key>__<joint>.png"""
import json, os, sys, time, traceback
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
import numpy as np, torch as th
from PIL import Image, ImageDraw, ImageFont
import omnigibson as og, omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.objects.primitive_object import PrimitiveObject
A = "/tmp/glib/artic"; os.makedirs(f"{A}/strips2", exist_ok=True); os.makedirs(f"{A}/render2", exist_ok=True)
todo = json.load(open(f"{A}/nohandle_joints.json"))
shard, nshard = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
keys = sorted({t["key"] for t in todo}); keys = [k for i, k in enumerate(keys) if i % nshard == shard]
env = og.Environment(configs={"scene": {"type": "Scene"}}); scene = env.scene
for _ in range(3): og.sim.step()
cam = og.sim.viewer_camera; cam.set_position_orientation(th.tensor([2.0, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
for _ in range(6): og.sim.render()
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except Exception: FONT = ImageFont.load_default()
COL = {"TL": (1, 0.1, 0.1), "TR": (0.1, 0.8, 0.1), "BL": (0.1, 0.3, 1.0), "BR": (1.0, 0.5, 0.0), "C": (0.55, 0.55, 0.55)}
og.sim.stop(); POOL = {}
for k, c in COL.items():
    POOL[k] = PrimitiveObject(name=f"mk_{k}", primitive_type="Sphere", radius=0.02, visual_only=True, rgba=[*c, 1.0]); scene.add_object(POOL[k])
og.sim.play(); og.sim.step(); FAR = th.tensor([50.0, 50.0, -5.0])


def look_at(cam_pos, target, up=th.tensor([0.0, 0.0, 1.0])):
    f = target - cam_pos; f = f / th.linalg.norm(f); r = th.linalg.cross(f, up)
    if th.linalg.norm(r) < 1e-3: r = th.linalg.cross(f, th.tensor([1.0, 0.0, 0.0]))
    r = r / th.linalg.norm(r); u = th.linalg.cross(r, f); return T.mat2quat(th.stack([r, u, -f], dim=1))


def grab(cam_pos, target, path, size=460):
    cam.set_position_orientation(th.as_tensor(cam_pos, dtype=th.float32), look_at(th.as_tensor(cam_pos, dtype=th.float32), th.as_tensor(target, dtype=th.float32)))
    for _ in range(3): og.sim.render()
    rgb = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy(); h, w = rgb.shape[:2]; s = min(h, w)
    Image.fromarray(rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]).resize((size, size), Image.LANCZOS).save(path)


def place(items):
    for k, m in POOL.items(): m.set_position_orientation(th.as_tensor(items[k], dtype=th.float32) if k in items else FAR, th.tensor([0.0, 0.0, 0.0, 1.0]))
    og.sim.step()


done = 0; t0 = time.time()
for key in keys:
    H = json.load(open(f"{A}/handles/{key}.json")); obj = None
    try:
        og.sim.stop(); obj = DatasetObject(name=f"o_{key}", category=H["category"], model=H["model"], fixed_base=True, include_default_states=False); scene.add_object(obj); og.sim.play(); og.sim.step()
        alo, ahi = obj.aabb; rp0, _ = obj.root_link.get_position_orientation()
        obj.set_position_orientation(th.tensor([0.0, 0.0, float(rp0[2] - alo[2])]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        names = list(obj.joints.keys()); q = obj.get_joint_positions().clone()
        for j in H["joints"]:
            if j["joint"] in names: q[names.index(j["joint"])] = j["closed_q"]
        obj.set_joint_positions(q); obj.keep_still(); og.sim.step()
        root = obj.root_link.get_position_orientation()[0].numpy().astype(np.float64)
        for t in [t for t in todo if t["key"] == key]:
            j = next(x for x in H["joints"] if x["joint"] == t["joint"]); n = np.array(j.get("outward_normal", [1, 0, 0]), float)
            lo, hi = np.array(j["link_bbox_min"]), np.array(j["link_bbox_max"]); c = (lo + hi) / 2; ext = hi - lo
            if abs(n[2]) > 0.7:                                                      # top lid: outer face is the top; view from front-above; 'up' in the image = away from the viewer
                view_dir = np.array([1.0, 0, 0]) if ext[0] >= ext[1] else np.array([0, 1.0, 0]); horiz = np.cross(view_dir, [0, 0, 1.0]); up = -view_dir
                face_c = c.copy(); face_c[2] = hi[2]; half_h = np.abs(up) @ (ext / 2); half_w = np.abs(horiz) @ (ext / 2)
                cam_pos = root + face_c + view_dir * (1.3 * np.linalg.norm(ext) + 0.2) + np.array([0, 0, 1.0 * np.linalg.norm(ext) + 0.15])
            else:
                horiz = np.cross([0, 0, 1.0], n); horiz /= np.linalg.norm(horiz); up = np.array([0, 0, 1.0])       # image right = horiz (camera on +n side looking at -n)
                face_c = c + n * (np.abs(n) @ (ext / 2)); half_h = ext[2] / 2; half_w = np.abs(horiz) @ (ext / 2)
                cam_pos = root + face_c + n * (1.4 * np.linalg.norm(ext) + 0.25) + np.array([0, 0, 0.25 * np.linalg.norm(ext)])
            # camera looking along -n: image right is 'horiz' if horiz = cross(up, n)?? compute properly: right = cross(forward, up) with forward = -n
            fwd = -(cam_pos - (root + face_c)); fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0, 0, 1.0]); right /= max(np.linalg.norm(right), 1e-9)
            # corners of the outer face in world: TL = up-left etc. (left = -right)
            sgn_r = 1.0 if np.dot(right, horiz) >= 0 else -1.0
            corners = {"TL": face_c + up * half_h - horiz * sgn_r * half_w, "TR": face_c + up * half_h + horiz * sgn_r * half_w, "BL": face_c - up * half_h - horiz * sgn_r * half_w, "BR": face_c - up * half_h + horiz * sgn_r * half_w, "C": face_c}
            place({k: root + v for k, v in corners.items()})
            json.dump({"key": key, "joint": t["joint"], "corners_obj": {k: [round(float(x), 4) for x in v] for k, v in corners.items()}, "n": n.tolist(), "up": [float(x) for x in up], "right_dir": [float(x) for x in (horiz * sgn_r)], "half_w": float(half_w), "half_h": float(half_h)},
                      open(f"{A}/render2/{key}__{t['joint']}_corners.json", "w"))
            p1 = f"{A}/render2/{key}__{t['joint']}_face.png"; grab(cam_pos, root + face_c, p1); grab(cam_pos, root + face_c, p1)
            side_pos = root + face_c + horiz * sgn_r * (1.2 * np.linalg.norm(ext) + 0.2) + n * 0.4 * np.linalg.norm(ext) + np.array([0, 0, 0.5 * np.linalg.norm(ext) + 0.1])
            p2 = f"{A}/render2/{key}__{t['joint']}_side.png"; grab(side_pos, root + face_c, p2)
            q2 = q.clone(); i = names.index(t["joint"]); q2[i] = j["closed_q"] + j["open_sign"] * 0.6 * (j["upper"] - j["lower"])
            obj.set_joint_positions(q2); obj.keep_still(); og.sim.step()
            p3 = f"{A}/render2/{key}__{t['joint']}_open.png"; grab(cam_pos, root + face_c, p3)
            obj.set_joint_positions(q); obj.keep_still(); og.sim.step()
            S = 460; strip = Image.new("RGB", (3 * S, S + 58), (255, 255, 255)); dr = ImageDraw.Draw(strip)
            dr.text((6, 4), f"{H['category']} / {H['model']} / {t['joint']} ({t['kind']}, {t['type'][:5]})  part size w x h x d = {abs(horiz) @ ext:.2f} x {abs(up) @ ext:.2f} x {abs(n) @ ext:.2f} m   reviewer: {str(t['review_desc'])[:70]}", fill=(0, 0, 0), font=FONT)
            for k, (title, p) in enumerate((("outer face: TL red TR green BL blue BR yellow C grey", p1), ("side / above", p2), ("opened 60 %", p3))):
                strip.paste(Image.open(p).resize((S, S)), (k * S, 30 + 24)); dr.text((k * S + 4, 30), title, fill=(0, 0, 0), font=FONT)
            strip.save(f"{A}/strips2/{key}__{t['joint']}.png"); done += 1
        place({})
        print(f"{key}: done ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAILED {key}: {e!r}", flush=True); traceback.print_exc()
    finally:
        try:
            og.sim.stop()
            if obj is not None: scene.remove_object(obj)
            og.sim.play(); og.sim.step()
        except Exception as e: print("cleanup failed", key, repr(e))
print(f"RENDER2 DONE {done}", flush=True); og.shutdown()
