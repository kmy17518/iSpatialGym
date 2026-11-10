#!/usr/bin/env python3
"""Vectorize ten supplied LEF puzzles and build separable physical line parts.

Ordinary Python --prepare --output NEWDIR; then Blender --background --python
THIS -- --output NEWDIR. Source images remain byte-identical. Only line geometry
is extruded, with the established 12mm bars, 20mm height and matte-gray material.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

ITEMS = (0, 4, 12, 16, 20, 24, 28, 32, 36, 40)
BAR_WIDTH = 0.012
BAR_HEIGHT = 0.020
CLEARANCE = 0.0003
METRES_PER_PIXEL = 0.2 / 300.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def extract_segments(gray):
    """Merge anti-aliased stroke edges, then least-squares fit center lines."""
    import cv2
    import numpy as np
    mask = gray < 210
    raw = cv2.HoughLinesP((mask*255).astype(np.uint8), 1, np.pi/1800, threshold=20,
                         minLineLength=25, maxLineGap=5)
    if raw is None:
        raise ValueError("no drawing strokes")
    clusters = []
    for line in sorted(raw[:,0].tolist(), key=lambda a: -((a[2]-a[0])**2+(a[3]-a[1])**2)):
        p, q = np.array(line[:2],float), np.array(line[2:],float)
        direction = (q-p)/np.linalg.norm(q-p)
        cluster = None
        for c in clusters:
            if abs(np.dot(direction,c["direction"])) > np.cos(np.deg2rad(2.0)):
                normal = np.array([-c["direction"][1],c["direction"][0]])
                if max(abs(np.dot(p-c["point"],normal)), abs(np.dot(q-c["point"],normal))) < 2.5:
                    cluster = c
                    break
        if cluster is None:
            cluster = {"point":p,"direction":direction,"ends":[]}
            clusters.append(cluster)
        cluster["ends"].extend([p,q])
    y,x = np.where(mask)
    pixels = np.stack([x,y],axis=1).astype(float)
    segments = []
    explained = np.zeros(len(pixels),bool)
    for c in clusters:
        p,d = c["point"],c["direction"]
        normal = np.array([-d[1],d[0]])
        ends = np.array(c["ends"])
        projected = (ends-p)@d
        along = (pixels-p)@d
        near = np.abs((pixels-p)@normal)<1.7
        indices = np.flatnonzero(near)
        indices = indices[np.argsort(along[indices])]
        runs = np.split(indices,np.flatnonzero(np.diff(along[indices])>4.0)+1)
        chosen = max(runs,key=lambda ids: sum((along[ids]>=projected.min()-3)&(along[ids]<=projected.max()+3)))
        selected = np.zeros(len(pixels),bool)
        selected[chosen] = True
        points = pixels[selected]
        if len(points)<15:
            continue
        centre = points.mean(axis=0)
        _,_,vt = np.linalg.svd(points-centre,full_matrices=False)
        direction = vt[0]
        coordinates = (points-centre)@direction
        a,b = centre+coordinates.min()*direction, centre+coordinates.max()*direction
        segments.append([a.tolist(),b.tolist()])
        explained |= selected
    coverage = float(explained.mean())
    if coverage < 0.97:
        raise ValueError(f"stroke reconstruction coverage only {coverage}")
    return segments, coverage


def prepare(args, output):
    import cv2
    import numpy as np
    from shapely import constrained_delaunay_triangles
    from shapely.geometry import LineString, box, Polygon
    from shapely.ops import unary_union, linemerge
    from shapely.geometry.polygon import orient
    root = Path(__file__).resolve().parents[2]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for name in ("figure_geometry.py", "build_embedded_figure_blender.py"):
        shutil.copyfile(root / "ispatialgym/assets/objects/embedded_figure" / name, source / name)
    archive = zipfile.ZipFile(args.archive)
    targets = zipfile.ZipFile(io.BytesIO(archive.read("object_models_to_generate/disembedding_puzzle_target.zip")))
    contexts = zipfile.ZipFile(io.BytesIO(archive.read("object_models_to_generate/disembedding_puzzle_context.zip")))
    recipes, figures = [], []
    for index,item in enumerate(ITEMS):
        reference_dir = output / "references" / f"{item:03d}"
        reference_dir.mkdir(parents=True)
        target_path,context_path = reference_dir / "target.png",reference_dir / "context.png"
        target_path.write_bytes(targets.read(f"{item:03d}.png"))
        context_path.write_bytes(contexts.read(f"{item:03d}a.png"))
        target,context = cv2.imread(str(target_path),0),cv2.imread(str(context_path),0)
        target_segments,target_coverage = extract_segments(target)
        context_segments,context_coverage = extract_segments(context)
        mask = target<210
        ys,xs = np.where(mask)
        crop = mask[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.float32)
        distance = cv2.distanceTransform((context>=210).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        costs = cv2.matchTemplate(distance,crop,cv2.TM_CCORR)/crop.sum()
        minimum,_,best,_ = cv2.minMaxLoc(costs)
        shift = np.array([best[0]-xs.min(),best[1]-ys.min()],float)
        if minimum > 0.5:
            raise ValueError(f"item{item}: source target not matched to context, mean distance={minimum}")
        snapped = []
        for aa,bb in target_segments:
            a,b = np.array(aa)+shift,np.array(bb)+shift
            direction = (b-a)/np.linalg.norm(b-a)
            candidates = []
            for cc,dd in context_segments:
                c,d = np.array(cc),np.array(dd)
                tangent = (d-c)/np.linalg.norm(d-c)
                normal = np.array([-tangent[1],tangent[0]])
                error = max(abs(np.dot(a-c,normal)),abs(np.dot(b-c,normal)))
                overlap = min(np.dot(a-c,tangent),np.dot(b-c,tangent)) >= -3 and max(np.dot(a-c,tangent),np.dot(b-c,tangent)) <= np.linalg.norm(d-c)+3
                if abs(np.dot(direction,tangent))>np.cos(np.deg2rad(2.0)) and error<2.5 and overlap:
                    candidates.append((error,c+np.dot(a-c,tangent)*tangent,c+np.dot(b-c,tangent)*tangent))
            if not candidates:
                raise ValueError(f"item{item}: target segment does not match a context stroke")
            _,a,b = min(candidates,key=lambda a:a[0])
            snapped.append([a,b])
        # Join near-identical endpoints before buffering so outline corners use
        # the same mitred geometry as the established triangle model.
        endpoints = [p for segment in snapped for p in segment]
        for i,p in enumerate(endpoints):
            neighbors = [q for q in endpoints if np.linalg.norm(p-q)<3.0]
            endpoints[i] = np.mean(neighbors,axis=0)
        def frame(point):
            return ((float(point[0])-250)*METRES_PER_PIXEL,(250-float(point[1]))*METRES_PER_PIXEL)
        target_lines = [LineString([frame(endpoints[2*i]),frame(endpoints[2*i+1])]) for i in range(len(snapped))]
        # Snap endpoint clusters exactly, avoiding tiny raster-fit disconnections.
        coords = [list(line.coords) for line in target_lines]
        all_points = [p for seg in coords for p in seg]
        fixed = []
        for p in all_points:
            existing = next((q for q in fixed if np.linalg.norm(np.array(p)-q)<0.0015),None)
            fixed.append(existing if existing is not None else p)
        target_lines = [LineString([fixed[2*i],fixed[2*i+1]]) for i in range(len(coords))]
        network = unary_union(target_lines)
        network = linemerge(network) if network.geom_type=="MultiLineString" else network
        target_shape = network.buffer(BAR_WIDTH/2,cap_style=2,join_style=2)
        inner = 0.1-BAR_WIDTH/2-CLEARANCE
        target_shape = target_shape.intersection(box(-inner,-inner,inner,inner))
        if target_shape.geom_type != "Polygon" or not target_shape.is_valid:
            raise ValueError(f"item{item}: target must remain one connected valid part, got{target_shape.geom_type}")
        context_shape = unary_union([LineString([frame(a),frame(b)]).buffer(BAR_WIDTH/2,cap_style=2,join_style=2) for a,b in context_segments])
        context_only = context_shape.difference(target_shape.buffer(CLEARANCE,join_style=2))
        components = [context_only] if context_only.geom_type=="Polygon" else list(context_only.geoms)
        components = sorted([p for p in components if isinstance(p,Polygon) and p.area>=4e-6], key=lambda p:-p.area)
        if not components or min(target_shape.distance(p) for p in components)<CLEARANCE-1e-8:
            raise ValueError("target/context clearance failed")
        split = "eval_model" if index==9 else "train"
        ids = []
        for part_index,poly in enumerate([target_shape]+components):
            model = f"ief{index:02d}" + ("t" if part_index==0 else chr(96+part_index))
            if part_index>19:
                raise ValueError("part model suffix collides with target")
            triangles = [t for t in constrained_delaunay_triangles(poly).geoms if t.area > 1e-12]
            area = sum(t.area for t in triangles)
            if abs(area-poly.area)>1e-10:
                raise ValueError("convex decomposition changed area")
            pieces = [[f"tri{k:03d}",list(orient(t,sign=1).exterior.coords)[:-1]] for k,t in enumerate(triangles)]
            recipe = {"category":"jigsaw_puzzle_piece","model":model,"family":"embedded_figure","split_reservation":split,
                "item":item,"role":"target" if part_index==0 else "context_frame" if part_index==1 else "context_loose",
                "pieces":pieces,"planar_area_m2":poly.area,"target_image_sha256":sha(target_path),"context_image_sha256":sha(context_path),
                "archive_sha256":sha(args.archive),"target_match_translation_px":shift.tolist(),"target_match_mean_distance_px":max(0.,minimum),
                "target_stroke_coverage":target_coverage,"context_stroke_coverage":context_coverage,
                "bar_width_m":BAR_WIDTH,"height_m":BAR_HEIGHT,"clearance_m":CLEARANCE,"ready":False}
            recipes.append(recipe);ids.append(model)
        figures.append({"item":item,"target":ids[0],"models":ids,"split_reservation":split,"ready":False,
                        "minimum_target_context_clearance_m":min(target_shape.distance(p) for p in components)})
    (output / "geometry.json").write_text(json.dumps({"models":recipes,"figures":figures},indent=2,sort_keys=True)+"\n")
    print(f"LEF_GEOMETRY_PREPARED figures={len(figures)} parts={len(recipes)}")


def build(output):
    import bpy
    data = json.loads((output / "geometry.json").read_text())
    source = output / "source"
    module(source / "figure_geometry.py", "figure_geometry")
    builder = module(source / "build_embedded_figure_blender.py", "embedded_builder")
    builder.G.TARGET = "unused_original_triangle_identifier"
    rows = []
    for recipe in data["models"]:
        model = recipe["model"]
        dest = output / model
        if dest.exists():
            raise FileExistsError(dest)
        dest.mkdir()
        builder.clear_scene()
        builder.HERE, builder.OUT = dest, dest / "model"
        builder.ROLES = {model:(recipe["role"],f"Supplied LEF item{recipe['item']:03d}; lines only, target separable from context")}
        material = builder.color_material("bar_ink",builder.INK,0.85)
        manifest = builder.build_part(model,recipe["pieces"],material)
        expected_mass = recipe["planar_area_m2"]*BAR_HEIGHT*builder.G.DENSITY
        if abs(manifest["mass_kg"]-expected_mass)>max(1e-6,expected_mass*0.001):
            raise ValueError(f"{model}: visual union volume changed")
        manifest.update({"figure":f"LEF item{recipe['item']:03d}","campaign_recipe":recipe,"ready":False})
        path = builder.OUT / model / "build_manifest.json"
        path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        bpy.ops.wm.save_as_mainfile(filepath=str(dest / "embedded_figure.blend"))
        model_dir = path.parent
        rows.append({**{k:v for k,v in recipe.items() if k!="pieces"},"model_dir":str(model_dir),"build_manifest_sha256":sha(path),
            "files":{str(p.relative_to(model_dir)):sha(p) for p in sorted(model_dir.rglob("*")) if p.is_file()}})
    index = {"schema":"isg-campaign-custom-builds-v1","models":rows,"figures":data["figures"]}
    add_assemblies(output,index)
    (output / "asset_builds.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
    print(f"CAMPAIGN_LEF_URDF_BUILD_COMPLETE figures={len(data['figures'])} parts={len(rows)}")


def add_assemblies(output,index):
    directory = output / "assemblies"
    directory.mkdir(exist_ok=True)
    for figure in index["figures"]:
        parts = {}
        for model in figure["models"]:
            manifest = json.loads((output/model/"model"/model/"build_manifest.json").read_text())
            parts[model] = {k:manifest[k] for k in ("category","role","origin_in_figure_m","mass_kg","bbox_m")}
        assembly = {**figure,"figure":f"LEF item{figure['item']:03d}","square_m":0.2,
                    "bar_width_m":BAR_WIDTH,"height_m":BAR_HEIGHT,"clearance_m":CLEARANCE,
                    "parts":parts,"figure_frame":"x right, y up drawing, z up; bottom at0; centre of source300px square",
                    "ready":False,"fresh_assembly_settling_and_target_lift_smoke":"required"}
        path = directory / f"item{figure['item']:03d}.json"
        path.write_text(json.dumps(assembly,indent=2,sort_keys=True)+"\n")
        for row in index["models"]:
            if row["model"] in figure["models"]:
                row["assembly_file"] = str(path.relative_to(output))
                row["assembly_sha256"] = sha(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare",action="store_true")
    ap.add_argument("--archive",type=Path,default=Path("/home/minyeongk/Desktop/projects/ispatialgym/object_models_to_generate.zip"))
    ap.add_argument("--output",type=Path,required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    if args.prepare:
        prepare(args,args.output.resolve())
    else:
        build(args.output.resolve())


if __name__ == "__main__":
    main()
