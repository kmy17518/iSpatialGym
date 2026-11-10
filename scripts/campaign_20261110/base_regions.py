"""CPU free-cell manifests using the campaign's frozen room-map convention."""
from pathlib import Path

from scripts.campaign_20261110.contracts import SCENES, file_digest, require
from ispatialgym.sampling.split_seeds import MAP_RESOLUTION_M, NAV_ERODE_PX


def room_base_region(dataset, scene, room):
    import cv2
    import numpy as np

    require(scene in SCENES, "unknown campaign scene")
    root = Path(dataset).resolve()/"behavior-1k-assets"
    categories = root/"metadata/room_categories.txt"
    layout = root/"scenes"/scene/"layout"
    sources = [categories]+[layout/name for name in ("floor_insseg_0.png", "floor_semseg_0.png", "floor_trav_0.png")]
    cats = categories.read_text().splitlines()
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in sources[1:]]
    require(all(image is not None for image in images), "missing room maps")
    ins, sem, trav = images
    require(ins.shape == sem.shape and ins.shape[0] == ins.shape[1], "room maps must be equal squares")
    side = int(ins.shape[0]*.01/MAP_RESOLUTION_M)
    ins = cv2.resize(ins,(side,side),interpolation=cv2.INTER_NEAREST)
    sem = cv2.resize(sem,(side,side),interpolation=cv2.INTER_NEAREST)
    trav = cv2.resize(trav,(side,side))
    free = cv2.erode((trav==255).astype("uint8"),np.ones((NAV_ERODE_PX,NAV_ERODE_PX))) > 0
    by_type, names = {}, {}
    for iid in sorted(set(np.unique(ins).tolist())-{0}):
        rows,cols = np.where(ins==iid)
        semantic = int(sem[rows[0],cols[0]])
        index = len(by_type.setdefault(semantic,[]))
        by_type[semantic].append(iid)
        names[f"{cats[semantic-1]}_{index}"] = iid
    require(room in names, "unknown room instance")
    cells = np.argwhere(free & (ins==names[room]))
    require(len(cells)>0,"room has no eroded free cells")
    # BaseMap.map_to_world flips (row,col) after centering and scaling.
    xy = [[round((int(col)-side/2)*MAP_RESOLUTION_M,8),round((int(row)-side/2)*MAP_RESOLUTION_M,8)] for row,col in cells]
    return {"schema":"isg-room-base-region-v1","scene":scene,"room":room,"dataset":str(Path(dataset).resolve()),
            "map_resolution_m":MAP_RESOLUTION_M,"erosion_square_pixels":NAV_ERODE_PX,
            "map_to_world":"xy = flip((row_col - side/2) * resolution)","free_xy":xy,
            "sources":[{"path":str(path),"sha256":file_digest(path)} for path in sources],
            "validation":"frozen room segmentation and eroded traversal map; loaded-scene collision/IK/path checks remain required"}
