"""Head-camera frames of two replays over one episode's first N frames only (before a known trajectory divergence): mean of the
per-frame mean absolute difference (0-255) and the median PSNR.
    cmp_prefix.py <lerobot root A> <lerobot root B> <episode index> <N>"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_frames import episodes, frames

a_root, b_root, ep, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
(da, ma), (db, mb) = episodes(a_root), episodes(b_root)
fa, fb = frames(da, ma.iloc[ep]), frames(db, mb.iloc[ep])
k = min(n, len(fa), len(fb))
mad, psnr = [], []
for i in range(k):
    d = np.abs(fa[i].astype(np.int16) - fb[i].astype(np.int16))
    mad.append(float(d.mean()))
    mse = float((d.astype(np.float32) ** 2).mean())
    psnr.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
print(f"episode {ep} frames 0-{k - 1}: mean abs diff mean {np.mean(mad):.2f} max {np.max(mad):.2f}; PSNR median {np.median(psnr):.1f} dB min {np.min(psnr):.1f}")
