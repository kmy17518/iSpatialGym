"""One PNG strip per model: 5 rows (valid grasps), 4 columns (profile, opposite profile, oblique-below, wrist camera)."""
import json, os, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
OUT = "/tmp/glib"; CELL = 300; LABEL_W = 250
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14); font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12); font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
os.makedirs(f"{OUT}/strips", exist_ok=True)
keys = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob(f"{OUT}/results/*.json"))]
made = 0
for key in keys:
    rp = f"{OUT}/results/{key}.json"; sp = f"{OUT}/strips/{key}.png"
    if not os.path.exists(rp) or (os.path.exists(sp) and os.path.getmtime(sp) > os.path.getmtime(rp)): continue
    r = json.load(open(rp)); valid = [t for t in r["tested"] if t["ag_attached"] and t["held_after_gravity"] and t.get("frames")]
    rows = max(1, min(5, len(valid))); rest = (r.get("candidates_meta") or {}).get("rest_pose", "canonical")
    strip = Image.new("RGB", (LABEL_W + 4 * CELL, 40 + rows * CELL), (255, 255, 255)); d = ImageDraw.Draw(strip)
    scale_s = f"   SCALED x{r['scale']} ({r.get('scale_rule')})" if r.get("scale") else ""
    hdr = f"{r['category']} / {r['model']}   task: {r['task']}   pipeline: {r['pipeline']}{(' mode ' + str(r.get('mode'))) if r.get('mode') else ''}   constraint: {r['constraint']}   rest pose: {rest}{scale_s}   valid {r['n_valid']}/{len(r['tested'])} tested"
    d.text((8, 10), hdr, fill=(0, 0, 0), font=font_b)
    for ci, name in enumerate(("V1 profile (pinch axis L-R)", "V2 opposite profile", "V3 oblique from below", "V4 right wrist camera")):
        d.text((LABEL_W + ci * CELL + 6, 26), name, fill=(80, 80, 80), font=font_s)
    if not valid:
        d.text((LABEL_W + 20, 40 + CELL // 2), f"no valid grasp ({r.get('error') or 'all candidates failed AG'})", fill=(180, 0, 0), font=font_b)
    for ri, t in enumerate(valid[:5]):
        y0 = 40 + ri * CELL; m = t.get("meta") or {}
        lines = [f"grasp #{ri+1} (cand idx {t['idx']})", f"AG ok, attached step {t['attached_step']}", f"tilt {m.get('tilt_deg', float('nan')):.0f} deg" if m.get("tilt_deg") is not None else "",
                 f"CoM offset {m.get('com_perp', 0)*1000:.0f} mm" if "com_perp" in m else "", f"width {m.get('width', 0)*1000:.0f} mm" if "width" in m else "",
                 f"fingers closed at {t['finger_q'][0]*1000:.0f}/{t['finger_q'][1]*1000:.0f} mm" if t.get("finger_q") else "",
                 f"antipodal {m.get('antipodal', 0):.2f}  coverage {m.get('coverage', 0):.2f}" if "antipodal" in m else "",
                 f"GraspGen conf {m.get('graspgen_conf'):.2f}" if m.get("graspgen_conf") is not None else "",
                 f"stem t {m.get('stem_t'):.2f}" if m.get("stem_t") is not None else "", f"attach dist {m.get('attach_dist', 0)*1000:.0f} mm" if m.get("attach_dist") is not None else ""]
        for li, txt in enumerate([x for x in lines if x]):
            d.text((8, y0 + 8 + 18 * li), txt, fill=(0, 0, 0) if li == 0 else (60, 60, 60), font=font if li == 0 else font_s)
        for ci, v in enumerate(("v1", "v2", "v3", "wrist")):
            fp = t["frames"].get(v)
            if fp and os.path.exists(fp):
                strip.paste(Image.open(fp).resize((CELL, CELL), Image.LANCZOS), (LABEL_W + ci * CELL, y0))
            d.rectangle([LABEL_W + ci * CELL, y0, LABEL_W + (ci + 1) * CELL - 1, y0 + CELL - 1], outline=(200, 200, 200), width=1)
    strip.save(sp); made += 1
print(f"STRIPS: {made} written")
