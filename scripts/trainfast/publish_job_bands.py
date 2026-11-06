#!/usr/bin/env python3
"""Copy the reach bands a job's sampler computed in-process into the shared eval band store (owner 2026-10-12 ~03:00, item 2b).

    python publish_job_bands.py <job band cache> <store> <job start epoch> <job id>

The sampler writes a missing band into the job's private cache (ISPATIALGYM_REACH_BANDS, a copy that dies with the job):
``<name>.npz`` first, ``<name>.json`` last. A band is published when its json is newer than the job start, parses, and its npz
loads (a complete pair); each file goes through a dot-named temporary file and os.replace (atomic), and an existing file in the
store is never overwritten. One line per band (PUBLISHED_BAND / KEPT_BAND / SKIPPED_BAND) for the job log.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np


def main(src: str, dst: str, t0: float, job: str) -> int:
    os.makedirs(dst, exist_ok=True)
    for js in sorted(glob.glob(os.path.join(src, "*.json"))):
        npz = js[:-5] + ".npz"
        name = os.path.basename(npz)
        if os.path.getmtime(js) < t0 or not os.path.exists(npz):
            continue
        try:
            meta = json.load(open(js))
            with np.load(npz) as z:
                _ = z.files
        except Exception as e:  # noqa: BLE001 - an incomplete pair stays in the job's cache
            print(f"SKIPPED_BAND {name}: incomplete ({type(e).__name__}: {e})")
            continue
        if not meta.get("n_reachable_total"):
            # floor band names carry no room and ReachBands.find returns the newest of equally good bands: an empty band would
            # shadow a good one of another room or trunk check
            print(f"SKIPPED_BAND {name}: 0 reachable cells")
            continue
        if all(os.path.exists(os.path.join(dst, os.path.basename(f))) for f in (npz, js)):
            print(f"KEPT_BAND {name}: already in the store")
            continue
        for f in (npz, js):
            out = os.path.join(dst, os.path.basename(f))
            if not os.path.exists(out):
                tmp = os.path.join(dst, f".{os.path.basename(f)}.{job}.tmp")
                shutil.copy2(f, tmp)
                os.replace(tmp, out)
        print(f"PUBLISHED_BAND {name} object={meta.get('object')} grasp_key={meta.get('grasp_key')} arm={meta.get('arm', 'right')} "
              f"scale={meta.get('scale')} cells={meta.get('n_reachable_total')} time_s={meta.get('time_s')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]))
