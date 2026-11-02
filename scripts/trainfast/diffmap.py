"""Where two replays of one episode differ: the per-pixel mean |difference| over the episode as a heat map next to a mid-episode frame
of each replay, and the difference per frame over time (deciles).
    diffmap.py <lerobot A> <lerobot B> <episode> <out.png>"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/vision/u/minyeong/projects/isg_home/trainfast/scripts")
from compare_frames import episodes, frames
a, b, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
da, ma = episodes(a); db, mb = episodes(b)
fa, fb = frames(da, ma.iloc[k]), frames(db, mb.iloc[k])
n = min(len(fa), len(fb))
d = np.abs(fa[:n].astype(np.int16) - fb[:n].astype(np.int16)).mean(axis=3)
m = d.mean(axis=0)
per = d.mean(axis=(1, 2))
print("per-frame mean |diff| at deciles of the episode:", [round(float(per[int(q * (n - 1))]), 2) for q in np.linspace(0, 1, 11)])
h, w = m.shape
for name, (y0, y1, x0, x1) in {"top half": (0, h // 2, 0, w), "bottom half": (h // 2, h, 0, w), "left third": (0, h, 0, w // 3),
                               "right third": (0, h, 2 * w // 3, w)}.items():
    print(f"  {name}: mean {m[y0:y1, x0:x1].mean():.2f}")
heat = np.clip(m * (255.0 / max(1e-6, np.percentile(m, 99.5))), 0, 255).astype(np.uint8)
heat = np.stack([heat, np.zeros_like(heat), 255 - heat], axis=2)
i = n // 2
Image.fromarray(np.concatenate([fa[i], fb[i], heat], axis=1)).save(out)
print("saved", out)
