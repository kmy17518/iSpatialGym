"""Quick check that an OmniGibson scene JSON can be opened/loaded.

Usage (inside the `ispatialgym` conda env):
    OMNIGIBSON_HEADLESS=1 python test_scene_loadable.py

It points OmniGibson's data path at the *outdated* assets so the object
models referenced in the scene file resolve correctly, then loads the scene
into an Environment and steps the sim a few times.
"""

import os
import tempfile

# --- Configuration -----------------------------------------------------------
REPO = "/home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/datasets"
OUTDATED_ASSETS = os.path.join(REPO, "behavior-1k-assets-outdated")
SCENE_JSON = os.path.join(
    OUTDATED_ASSETS,
    "scenes/house_single_floor/json/house_single_floor_with_clutter.json",
)
SCENE_MODEL = "house_single_floor"

# --- Build a temp DATA_PATH where behavior-1k-assets -> outdated --------------
data_path = tempfile.mkdtemp(prefix="og_outdated_data_")
os.symlink(OUTDATED_ASSETS, os.path.join(data_path, "behavior-1k-assets"))
os.symlink(os.path.join(REPO, "omnigibson-robot-assets"), os.path.join(data_path, "omnigibson-robot-assets"))
os.symlink(os.path.join(REPO, "omnigibson.key"), os.path.join(data_path, "omnigibson.key"))

# Must be set BEFORE importing omnigibson.
os.environ["OMNIGIBSON_DATA_PATH"] = data_path
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

# The cluttered scene contains a filled container that trips a recipe rule's
# garbage fallback at load, which needs a GPU-dynamics fluid system (sludge).
# Transition rules are irrelevant to "is this scene openable", so disable them.
gm.ENABLE_TRANSITION_RULES = False

cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": SCENE_MODEL,
        "scene_file": SCENE_JSON,
    },
}

env = og.Environment(configs=cfg)
# flush=True: og.shutdown() hard-exits the process, discarding buffered stdout.
print(f"\nScene loaded OK. Object count: {len(env.scene.objects)}", flush=True)

for _ in range(5):
    og.sim.step()

print("Stepped the sim 5 times without error. Scene is openable.", flush=True)
og.shutdown()
