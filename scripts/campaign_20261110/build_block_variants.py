#!/usr/bin/env python3
"""Build isolated Kohs permutations, a red/white cube, and heldout boards.

Run with Blender --background --python THIS -- --output CAMPAIGN_ASSETS.
No source asset is changed. Each model gets its own original-builder namespace.
These are URDF candidates; conversion and simulator readiness remain required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys


NORMALS = ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
FACE_NAMES = ("+z", "-z", "+x", "-x", "+y", "-y")
COLORS = ("red", "blue", "white", "yellow", "blue|yellow", "red|white")


def cube_rotations():
    """24 face permutations induced by proper signed-axis rotations."""
    rotations = []
    for axes in itertools.permutations(range(3)):
        inversions = sum(axes[i] > axes[j] for i in range(3) for j in range(i + 1, 3))
        for signs in itertools.product((-1, 1), repeat=3):
            if (-1) ** inversions * signs[0] * signs[1] * signs[2] != 1:
                continue
            rotations.append(tuple(NORMALS.index(tuple(signs[i] * n[axes[i]] for i in range(3))) for n in NORMALS))
    assert len(set(rotations)) == 24
    return rotations


def canonical_faces(tiles):
    rotations = []
    for rotation in cube_rotations():
        moved = [None] * 6
        for old, new in enumerate(rotation):
            moved[new] = tiles[old]
        rotations.append(tuple(moved))
    return min(rotations)


def permutations(count=10):
    seen = {canonical_faces(tuple(range(6)))}
    out = []
    for tiles in itertools.permutations(range(6)):
        canonical = canonical_faces(tiles)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(tiles)
        if len(out) == count:
            return out
    raise ValueError("requested more distinct face assignments than available")


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(module, recipe, source_script):
    module.main()
    manifest_path = module.OUT / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["campaign_recipe"] = recipe
    manifest["source_builder"] = str(source_script)
    manifest["source_builder_sha256"] = sha256(source_script)
    manifest["ready"] = False
    manifest["readiness_blockers"] = ["OmniGibson conversion", "simulator geometry and collision smoke", "task grasp and replay validation"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files = {str(p.relative_to(module.OUT)): sha256(p) for p in sorted(module.OUT.rglob("*")) if p.is_file()}
    return {"category": module.CATEGORY, "model": module.MODEL, "model_dir": str(module.OUT),
            "build_manifest_sha256": sha256(manifest_path), "files": files, "ready": False, **recipe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    source = root / "ispatialgym/assets/objects"
    block_source = source / "kohs_block/build_kohs_block_blender.py"
    board_source = source / "block_board/build_block_board_blender.py"
    rows = []
    assignments = [(f"isgk{i:02d}", "toy_dice", tiles, "eval_model" if i == 10 else "train")
                   for i, tiles in enumerate(permutations(), 1)]
    assignments.append(("isgwch", "dice", (0, 0, 2, 2, 5, 5), "eval_category"))
    for model, category, tiles, split in assignments:
        module = module_from(block_source, "block_" + model)
        module.HERE = output / model
        module.HERE.mkdir()
        module.MODEL, module.CATEGORY = model, category
        module.OUT = module.HERE / "model" / model
        module.FACES = {name: (normal, tile, corners) for (name, (normal, _, corners)), tile in zip(module.FACES.items(), tiles)}
        module.FACE_COLORS = dict(zip(FACE_NAMES, (COLORS[t] for t in tiles)))
        recipe = {"split_reservation": split, "family": "kohs_face_permutation" if category == "toy_dice" else "wechsler_red_white",
                  "face_tiles": dict(zip(FACE_NAMES, tiles)), "source_atlas_sha256": sha256(module.ATLAS),
                  "physical_geometry": "unchanged one-inch cube; exact existing texture atlas",
                  "identical_blocks_per_instance": 4}
        if category == "toy_dice":
            recipe["rotation_orbit_representative"] = canonical_faces(tiles)
            recipe["distinctness"] = "face-label assignments differ under all 24 proper cube rotations; diagonal orientation can only further distinguish them"
        else:
            recipe["face_count_source"] = "https://wwwn.cdc.gov/nchs/Data/Nhanes3/Manuals/cognitiv.pdf"
            recipe["face_layout_note"] = "Two red, two white and two diagonally divided red/white faces; opposing equal-color face arrangement is an explicit simulation recipe."
        rows.append(finish(module, recipe, block_source))
    # Keep the proven pocket and chamfer geometry; distinct outer-rim widths make
    # actual new board meshes instead of aliases of the held-in board bytes.
    for model, rim_width, split in (("isgbdm", 0.007, "eval_model"), ("isgbdw", 0.008, "eval_category")):
        module = module_from(board_source, "board_" + model)
        module.HERE = output / model
        module.HERE.mkdir()
        module.MODEL = model
        module.OUT = module.HERE / "model" / model
        module.RIM_FLAT = rim_width
        module.PLATE = 2 * module.HALF_POCKETS + 2 * (module.OUTER_CHAMFER_W + rim_width)
        recipe = {"split_reservation": split, "family": "block_board", "rim_flat_m": rim_width,
                  "pocket_geometry": "original seat, pitch, floor, wall, divider and chamfer unchanged",
                  "required_physics_material": {"static_friction": 0.25, "dynamic_friction": 0.25, "friction_combine": "min"}}
        rows.append(finish(module, recipe, board_source))
    (output / "asset_builds.json").write_text(json.dumps({"schema": "isg-campaign-custom-builds-v1", "models": rows}, indent=2, sort_keys=True) + "\n")
    print(f"CAMPAIGN_ASSET_URDF_BUILD_COMPLETE models={len(rows)} output={output}")


if __name__ == "__main__":
    main()
