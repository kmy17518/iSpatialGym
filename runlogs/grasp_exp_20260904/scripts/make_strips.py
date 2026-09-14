"""Regenerate the per-object PNG strips from saved frames + results (tighter crop around the gripper)."""
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "/tmp/grasp_exp"
import sys
RANK = len(sys.argv) > 1 and sys.argv[1] == "rank"
SETS = [("a1", "A1 sampler+filter\nranked by clearance"), ("a1best", "A1 sampler+filter\nranked by CoM heuristic"), ("a2best", "GraspGen (floating object)\nranked by CoM heuristic + conf")] if RANK else [("a1", "A1: antipodal sampler\n+ geometric filter"), ("a2raw", "A2: GraspGen (Panda ckpt)\nraw top-5 by confidence"), ("a2tab", "A2: GraspGen top-5\napproach<=60deg + above table")]
CELL, LABEL_W = 300, 190
CROP = (150, 60, 700, 610)   # x0, y0, x1, y1 in the 720x720 frame
objects = json.load(open(f"{OUT}/objects_exported.json"))
results = json.load(open(f"{OUT}/ag_results_rank.json" if RANK else f"{OUT}/ag_results.json"))
rank_summary = json.load(open(f"{OUT}/rank_summary.json"))
a1 = json.load(open(f"{OUT}/a1_results.json")); a2 = json.load(open(f"{OUT}/a2_franka_panda_results.json")); a2b = json.load(open(f"{OUT}/a2_franka_panda_results_big.json"))
conv = json.load(open(f"{OUT}/a2_convert_summary.json"))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
STRIPS = f"{OUT}/strips_rank" if RANK else f"{OUT}/strips"
os.makedirs(STRIPS, exist_ok=True)
for o in objects:
    name = o["name"]
    strip = Image.new("RGB", (LABEL_W + 5 * CELL, 3 * CELL + 34), (255, 255, 255))
    d = ImageDraw.Draw(strip)
    ext = np.round(np.array(o["coll_bbox_max"]) - np.array(o["coll_bbox_min"]), 3).tolist()
    d.text((8, 8), f"{o['category']} / {o['model']}    collision bbox {ext} m,  mass {o['mass']:.2f} kg", fill=(0, 0, 0), font=font_b)
    for ri, (set_key, set_label) in enumerate(SETS):
        y0 = 34 + ri * CELL
        rs = results[name][set_key]
        if set_key == "a1best":
            info = [f"gen: {a1[name]['t_total_s']*1000:.0f} ms", "score = 5*CoM + antipodal", "+ coverage + clearance + vertical"]
        elif set_key == "a2best":
            info = [f"inference: {a2b[name]['t_inference_s']*1000:.0f} ms", f"{rank_summary[name]['a2_collision_free']} collision-free of 500", "no table constraint"]
        elif set_key == "a1":
            info = [f"gen: {a1[name]['t_total_s']*1000:.0f} ms", f"{a1[name]['n_sampled']} sampled -> {a1[name]['n_after_collision']} pass"]
        elif set_key == "a2raw":
            info = [f"inference: {a2[name]['t_inference_s']*1000:.0f} ms", f"400 sampled -> top 100", f"from above: {conv[name]['raw_top100_from_above']}/100"]
        else:
            info = [f"inference: {a2b[name]['t_inference_s']*1000:.0f} ms", f"2000 sampled -> top 500", f"above+table ok: {conv[name]['big_table_ok']}/500"]
        ok_n = sum(r["ag_attached"] and r["held_after_gravity"] for r in rs)
        d.multiline_text((8, y0 + 8), set_label, fill=(0, 0, 0), font=font, spacing=3)
        for li, t in enumerate(info):
            d.text((8, y0 + 56 + 18 * li), t, fill=(70, 70, 70), font=font_s)
        d.text((8, y0 + 56 + 18 * len(info) + 8), f"AG success {ok_n}/{len(rs)}", fill=(0, 120, 0) if ok_n == len(rs) and rs else (180, 0, 0), font=font_b)
        for ci in range(5):
            x0 = LABEL_W + ci * CELL
            if ci < len(rs):
                im = Image.open(rs[ci]["frame"]).crop(CROP).resize((CELL, CELL), Image.LANCZOS); strip.paste(im, (x0, y0))
                ok = rs[ci]["ag_attached"] and rs[ci]["held_after_gravity"]
                col = (0, 160, 0) if ok else (200, 0, 0)
                d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], outline=col, width=4)
                tilt = rs[ci]["tilt_deg"]; tilt_s = f"  approach tilt {tilt:.0f} deg" if tilt is not None else ""
                d.rectangle([x0 + 4, y0 + 4, x0 + CELL - 5, y0 + 26], fill=(255, 255, 255))
                d.text((x0 + 8, y0 + 6), f"AG {'OK' if ok else 'FAIL'}{tilt_s}", fill=col, font=font)
                com = None
                if RANK and set_key in ("a1best", "a2best") and ci < len(rank_summary[name].get(set_key, [])):
                    com = rank_summary[name][set_key][ci]["com_perp"]
                elif RANK and set_key == "a1" and ci < len(rank_summary[name].get("a1_old_com_perp", [])):
                    com = rank_summary[name]["a1_old_com_perp"][ci]
                if com is not None:
                    d.rectangle([x0 + CELL - 130, y0 + CELL - 26, x0 + CELL - 5, y0 + CELL - 5], fill=(255, 255, 255)); d.text((x0 + CELL - 126, y0 + CELL - 24), f"CoM off {com*1000:.0f} mm", fill=(0, 0, 0), font=font_s)
                if rs[ci]["conf"] is not None and set_key not in ("a1", "a1best"):
                    d.rectangle([x0 + 4, y0 + CELL - 26, x0 + 120, y0 + CELL - 5], fill=(255, 255, 255)); d.text((x0 + 8, y0 + CELL - 24), f"conf {rs[ci]['conf']:.2f}", fill=(0, 0, 0), font=font_s)
            else:
                d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=(235, 235, 235)); d.text((x0 + 95, y0 + CELL // 2 - 8), "no candidate", fill=(120, 120, 120), font=font)
    strip.save(f"{STRIPS}/{name}.png")
# overview contact sheet: one row per object (A1 first grasp | A2 raw first grasp | A2 table first grasp)
print("STRIPS DONE")
