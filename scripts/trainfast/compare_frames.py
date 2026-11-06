"""Partial-scene validation (2026-11-01): head-camera frames of two replays of the same demos (episode k of A vs episode k of B):
per episode the mean / max per-frame mean absolute difference (0-255) and the lowest PSNR, and side-by-side PNGs of a few frames.
    python compare_frames.py <lerobot root A> <lerobot root B> <out png dir> [--frames 0,0.5,1]"""
import glob, json, os, sys
import av
import numpy as np
import pandas as pd

CAM = "observation.rgb.zed_link_camera_0"

def episodes(root):
    ds = glob.glob(os.path.join(root, "*_slot0"))[0]
    meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(ds, "meta", "episodes", "*", "*.parquet")))], ignore_index=True)
    return ds, meta.sort_values("episode_index").reset_index(drop=True)

def frames(ds, row):
    vid = os.path.join(ds, "videos", CAM, f"chunk-{int(row[f'videos/{CAM}/chunk_index']):03d}", f"file-{int(row[f'videos/{CAM}/file_index']):03d}.mp4")
    t0, t1 = float(row[f"videos/{CAM}/from_timestamp"]), float(row[f"videos/{CAM}/to_timestamp"])
    out = []
    with av.open(vid) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t0 - 1e-6 <= t < t1 - 1e-6:
                out.append(fr.to_ndarray(format="rgb24"))
    return np.stack(out) if out else np.zeros((0, 1, 1, 3), np.uint8)

def main():
    a_root, b_root, png = sys.argv[1], sys.argv[2], sys.argv[3]
    picks = [float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[4] == "--frames" else ["0", "0.5", "1"])]
    os.makedirs(png, exist_ok=True)
    da, ma = episodes(a_root)
    db, mb = episodes(b_root)
    from PIL import Image
    for k in range(min(len(ma), len(mb))):
        fa, fb = frames(da, ma.iloc[k]), frames(db, mb.iloc[k])
        n = min(len(fa), len(fb))
        if n == 0:
            print(f"episode {k}: no frames ({len(fa)} / {len(fb)})"); continue
        d = np.abs(fa[:n].astype(np.int16) - fb[:n].astype(np.int16)).mean(axis=(1, 2, 3))
        mse = ((fa[:n].astype(np.float32) - fb[:n].astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
        psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))
        print(f"episode {k}: frames {len(fa)} / {len(fb)}; mean abs diff mean {d.mean():.2f} max {d.max():.2f} (frame {int(d.argmax())}); "
              f"PSNR min {psnr.min():.1f} dB median {np.median(psnr):.1f} dB")
        for p in picks:
            i = min(n - 1, int(round(p * (n - 1))))
            Image.fromarray(np.concatenate([fa[i], fb[i]], axis=1)).save(os.path.join(png, f"ep{k}_f{i:04d}.png"))

if __name__ == "__main__":
    main()
