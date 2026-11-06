"""Head-camera sharpness of replays: per episode the median (and 10th / 90th percentile) variance of the Laplacian of the grey image.
    sharpness.py <lerobot root> [<lerobot root> ...]"""
import sys
import numpy as np
sys.path.insert(0, "/vision/u/minyeong/projects/isg_home/trainfast/scripts")
from compare_frames import episodes, frames

def lap_var(img):
    g = img.astype(np.float32).mean(axis=2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())

for root in sys.argv[1:]:
    ds, meta = episodes(root)
    out = []
    for k in range(len(meta)):
        fr = frames(ds, meta.iloc[k])
        v = np.array([lap_var(f) for f in fr[::5]])
        out.append(f"ep{k} median {np.median(v):.1f} (p10 {np.percentile(v, 10):.1f}, p90 {np.percentile(v, 90):.1f}, n {len(fr)})")
    print(f"{root.split('/pv/')[-1]}: " + "; ".join(out))
