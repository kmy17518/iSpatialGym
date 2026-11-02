"""COLLECTOR_PREBUILD source choice (trainfast, 2026-11-02): may job_fast.sh prebuild the collectors' world from template copy <dir>?
Only when the copy's scene crop request is the one this job's sampler makes, else the first instance's signature differs, every
collector rebuilds its world (prebuild.mismatch) and the prebuild was wasted (11-01 packing fleet: mr-mug / de03 took full4's
SCENE_CROP=full copies, ~8 min per collector).

The request is (mode, radius in region mode): for this job, scene_crop.requested_mode / crop_radius under this job's switches; for
the copy, its template_meta.json scene_crop when it was cropped (requested_mode or mode, radius_m), else the switches of the job that
wrote it (<dir>/../../../job.env: ISPATIALGYM_FAST, _SCENE_CROP, _SCENE_CROP_RADIUS_M) under today's family defaults, and "full" for a
copy from before the crop (no scene_crop key: train85, tags before 11-01). Same template + same request = same crop plan. With
PARTIAL_SCENE off the collector loads the full template from any copy, so every copy fits.
    prebuild_crop.py <template copy dir>   -> "fits" | "skip <why>"   (exit 0 either way)"""
import json
import os
import re
import sys

from ispatialgym.fastflags import fast_on
from ispatialgym.sampling import scene_crop as SCROP

KEYS = ("ISPATIALGYM_FAST", "ISPATIALGYM_FAST_SCENE_CROP", "ISPATIALGYM_FAST_SCENE_CROP_RADIUS_M")


def request(subtask, env=None):
    """(mode, radius m) the sampler requests for ``subtask`` under ``env`` (None: this process's environment); the radius counts in
    ``region`` mode only."""
    saved = {k: os.environ.get(k) for k in KEYS}
    try:
        if env is not None:
            for k in KEYS:
                if env.get(k) is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = env[k]
        return SCROP.requested_mode(subtask), round(SCROP.crop_radius(), 4)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def job_env(path):
    env = {}
    try:
        for line in open(path):
            m = re.match(r"^(?:export )?(ISPATIALGYM_FAST(?:_SCENE_CROP(?:_RADIUS_M)?)?)=(.*)$", line.strip())
            if m:
                env[m.group(1)] = m.group(2).strip().strip("'\"")
    except OSError:
        pass
    return env


def main(d):
    try:
        meta = json.load(open(os.path.join(d, "template_meta.json")))
    except (OSError, ValueError) as e:
        return f"skip no template_meta.json ({type(e).__name__})"
    if not fast_on("PARTIAL_SCENE"):
        return "fits"
    mode, radius = request(meta.get("subtask"))
    want = (mode, radius if mode == "region" else None)
    sc = meta.get("scene_crop")
    if not isinstance(sc, dict):
        have = ("full", None)
    else:
        env_mode, env_radius = request(meta.get("subtask"), job_env(os.path.join(d, "..", "..", "..", "job.env")))
        if (sc.get("mode") or "full") == "full":
            mode, radius = env_mode, env_radius
        else:
            # a cropped copy records its request: requested_mode when the plan fell back (region -> rooms: no radius_m then)
            mode = sc.get("requested_mode") or sc["mode"]
            radius = round(float(sc["radius_m"]), 4) if sc["mode"] == "region" and sc.get("radius_m") is not None else env_radius
        have = (mode, radius if mode == "region" else None)
    return "fits" if have == want else f"skip crop request {have} != this job's {want}"


if __name__ == "__main__":
    print(main(sys.argv[1]))
