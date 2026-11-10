#!/usr/bin/env python3
"""Reconstruct ten supplied Shepard-Metzler drawings with the existing wood style.

First ordinary Python --prepare --source EXISTING_MENTAL_SOURCE --output NEWDIR;
then Blender --background --python THIS -- --output NEWDIR.
Recipes record the supplied image hashes and explicitly reconstructed cube cells.
"""
import argparse
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import shutil
import sys
import zipfile

CELLS = {
    3: [(0,-1,0),(0,0,0),(0,0,1),(0,0,2),(1,0,2),(2,0,2),(2,0,3)],
    4: [(0,-1,2),(0,0,2),(0,0,1),(0,0,0),(1,0,0),(2,0,0),(2,-1,0),(2,-2,0)],
    5: [(0,-1,0),(0,0,0),(0,0,1),(0,0,2),(1,0,2),(2,0,2),(2,0,3),(2,0,4)],
    6: [(0,2,0),(0,1,0),(0,0,0),(1,0,0),(2,0,0),(2,0,1),(2,0,2),(2,-1,2)],
    7: [(0,1,0),(0,0,0),(1,0,0),(2,0,0),(2,0,1),(2,0,2),(2,0,3),(1,0,3)],
    8: [(0,-1,3),(0,0,3),(0,0,2),(0,0,1),(0,0,0),(1,0,0),(2,0,0),(2,-1,0)],
    9: [(0,1,0),(0,0,0),(1,0,0),(2,0,0),(2,0,1),(2,0,2),(2,0,3),(3,0,3)],
    10: [(2,0,3),(1,0,3),(0,0,3),(0,0,2),(0,0,1),(0,0,0),(1,0,0),(1,-1,0)],
    11: [(0,0,3),(1,0,3),(2,0,3),(2,0,2),(2,0,1),(2,0,0),(2,-1,0),(2,-2,0),(1,-2,0)],
    12: [(-2,0,3),(-1,0,3),(0,0,3),(0,0,2),(0,0,1),(0,0,0),(1,0,0),(2,0,0),(2,-1,0)],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def normalized(cells):
    lo = [min(c[i] for c in cells) for i in range(3)]
    return tuple(sorted(tuple(c[i]-lo[i] for i in range(3)) for c in cells))


def orbit(cells):
    out = set()
    for axes in itertools.permutations(range(3)):
        parity = (-1) ** sum(axes[i] > axes[j] for i in range(3) for j in range(i+1,3))
        for signs in itertools.product((-1,1), repeat=3):
            if parity * signs[0] * signs[1] * signs[2] == 1:
                out.add(normalized([tuple(signs[i]*c[axes[i]] for i in range(3)) for c in cells]))
    return out


def geometry(source, cells):
    geo = module(source / "figure_cells.py", "figure_cells")
    original_grain = list(geo.GRAIN.values())
    geo.CUBES = {chr(65+i): tuple(c) for i,c in enumerate(cells)}
    geo.CHAIN = list(geo.CUBES)
    geo.GRAIN = {name: original_grain[i % len(original_grain)] for i,name in enumerate(geo.CUBES)}
    return geo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--archive", type=Path, default=Path("/home/minyeongk/Desktop/projects/ispatialgym/object_models_to_generate.zip"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    output = args.output.resolve()
    if args.prepare:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        source = output / "source"
        source.mkdir()
        for name in ("figure_cells.py", "make_texture.py", "build_mental_rotation_block_blender.py"):
            shutil.copyfile(args.source / name, source / name)
        archive = zipfile.ZipFile(args.archive)
        zipname = "object_models_to_generate/mental_rotation_blocks.zip"
        drawings = zipfile.ZipFile(io.BytesIO(archive.read(zipname)))
        seen, recipes = set(), []
        for i, (reference, cells) in enumerate(CELLS.items(), 1):
            rotations = orbit(cells)
            if len(rotations) != 24 or min(rotations) in seen:
                raise ValueError("rotational symmetry or duplicate shape")
            seen.add(min(rotations))
            if len(set(cells)) != len(cells) or any(sum(abs(a-b) for a,b in zip(x,y)) != 1 for x,y in zip(cells,cells[1:])):
                raise ValueError("cell chain is disconnected or repeats a cube")
            # Each shape is a single path, without hidden extra contacts.
            degrees = [sum(sum(abs(a-b) for a,b in zip(c,q)) == 1 for q in cells) for c in cells]
            if sorted(degrees) != [1,1] + [2]*(len(cells)-2):
                raise ValueError("unexpected extra cube adjacency")
            model = f"isgm{i:02d}"
            dest = output / model
            dest.mkdir()
            ref_name = f"All stimuli as jpg /{reference}_0.jpg"
            ref = dest / "reference.jpg"
            ref.write_bytes(drawings.read(ref_name))
            geo = geometry(source, cells)
            texture = module(source / "make_texture.py", "campaign_wood")
            texture.F = geo
            atlas = dest / "mental_rotation_block_wood.png"
            texture.make_atlas(str(atlas), seed=0)
            recipes.append({"model":model,"category":"classroom_mock_up","family":"mental_rotation",
                "split_reservation":"train","cells":cells,"reference_archive_member":zipname+"/"+ref_name,
                "archive_sha256":sha(args.archive),"reference_sha256":sha(ref),"atlas_sha256":sha(atlas),
                "source_sha256":{p.name:sha(p) for p in source.glob("*.py")},"proper_rotation_orbit_size":24,
                "chiral":min(orbit([(-c[0],c[1],c[2]) for c in cells])) != min(rotations),
                "geometry_source":"manual cell reconstruction from supplied item; cube size and wood synthesis copied from established source",
                "ready":False})
        (output / "recipes.json").write_text(json.dumps({"models":recipes},indent=2,sort_keys=True)+"\n")
        print(f"MENTAL_RECIPES_PREPARED {len(recipes)}")
        return
    recipes = json.loads((output / "recipes.json").read_text())["models"]
    source = output / "source"
    rows = []
    for recipe in recipes:
        model = recipe["model"]
        dest = output / model
        if (dest / "model").exists():
            raise FileExistsError(dest / "model")
        geo = geometry(source, recipe["cells"])
        builder = module(source / "build_mental_rotation_block_blender.py", "mental_builder_"+model)
        builder.MODEL, builder.CATEGORY = model, recipe["category"]
        builder.HERE, builder.OUT, builder.ATLAS = dest, dest / "model" / model, dest / "mental_rotation_block_wood.png"
        builder.BOXES = {"cube_"+name.lower():(cell,cell) for name,cell in geo.CUBES.items()}
        builder.main()
        path = builder.OUT / "build_manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update({"campaign_recipe":recipe,"mass_note":f"{len(geo.CUBES)} cubes × established6.9g per25.4mm wooden cube",
            "collision":"one exact convex box per cube; disjoint interiors", "ready":False,
            "symmetry":"no proper rotational symmetry; chiral="+str(recipe["chiral"])})
        path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        rows.append({**recipe,"model_dir":str(builder.OUT),"build_manifest_sha256":sha(path),
            "files":{str(p.relative_to(builder.OUT)):sha(p) for p in sorted(builder.OUT.rglob("*")) if p.is_file()}})
    (output / "asset_builds.json").write_text(json.dumps({"schema":"isg-campaign-custom-builds-v1","models":rows},indent=2,sort_keys=True)+"\n")
    print(f"CAMPAIGN_MENTAL_URDF_BUILD_COMPLETE models={len(rows)}")


if __name__ == "__main__":
    main()
