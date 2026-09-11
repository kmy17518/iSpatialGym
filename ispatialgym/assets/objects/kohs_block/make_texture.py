"""Texture atlas for the iSpatialGym Kohs block (``toy_dice/isgblk``).

3 x 2 tiles, 512 px each, one per cube face. Tile order (row-major, row 0 on top of the image):

    0 red      1 blue     2 white
    3 yellow   4 blue|yellow diagonal   5 red|white diagonal

The diagonal tiles are split along the tile diagonal from the top-left to the bottom-right corner
(first colour = upper-right triangle, second colour = lower-left triangle). A faint noise and a thin
darker border per tile keep the paint from looking synthetic. Run with any Python that has Pillow /
numpy (no Blender or OmniGibson needed)::

    python make_texture.py            # writes kohs_block_atlas.png next to this file
"""

from __future__ import annotations

import os

TILE = 512
COLS, ROWS = 3, 2
BORDER = 6  # px of darker rim per tile (painted edge)

COLORS = {
    "red": (196, 30, 36),
    "blue": (28, 64, 168),
    "white": (238, 236, 228),
    "yellow": (244, 196, 30),
}

#: face id -> tile spec. Solid tiles: colour name. Diagonal tiles: (upper-right colour, lower-left colour).
TILES = {
    0: "red",
    1: "blue",
    2: "white",
    3: "yellow",
    4: ("blue", "yellow"),
    5: ("red", "white"),
}


def tile_image(spec, rng):
    import numpy as np

    img = np.zeros((TILE, TILE, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    if isinstance(spec, str):
        img[:] = np.array(COLORS[spec], dtype=np.float32)
    else:
        a, b = spec
        upper_right = (xx + (TILE - 1 - yy)) >= (TILE - 1)  # x + (H-1-y) >= H-1 <=> above the TL->BR diagonal
        img[upper_right] = np.array(COLORS[a], dtype=np.float32)
        img[~upper_right] = np.array(COLORS[b], dtype=np.float32)
    # subtle paint / grain noise (+-3 %)
    noise = rng.normal(0.0, 0.015, size=(TILE, TILE, 1)).astype(np.float32)
    img = img * (1.0 + noise)
    # darker rim
    rim = (xx < BORDER) | (yy < BORDER) | (xx >= TILE - BORDER) | (yy >= TILE - BORDER)
    img[rim] *= 0.72
    return np.clip(img, 0, 255).astype(np.uint8)


def make_atlas(path: str, seed: int = 0) -> str:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    atlas = np.zeros((ROWS * TILE, COLS * TILE, 3), dtype=np.uint8)
    for face, spec in TILES.items():
        r, c = divmod(face, COLS)
        atlas[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile_image(spec, rng)
    Image.fromarray(atlas).save(path)
    return path


def tile_uv_rect(face: int, inset: float = 0.01):
    """UV rectangle (u0, v0, u1, v1) of ``face`` in OBJ convention (v = 0 at the image bottom)."""
    r, c = divmod(face, COLS)
    u0, u1 = c / COLS, (c + 1) / COLS
    v1, v0 = 1.0 - r / ROWS, 1.0 - (r + 1) / ROWS
    du, dv = (u1 - u0) * inset, (v1 - v0) * inset
    return u0 + du, v0 + dv, u1 - du, v1 - dv


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(make_atlas(os.path.join(here, "kohs_block_atlas.png")))
