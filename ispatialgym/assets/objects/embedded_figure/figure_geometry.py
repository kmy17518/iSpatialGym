"""Geometry of the iSpatialGym embedded figure (EFT item 11): the drawn lines as bars and the four parts.

Pure Python (no Blender / numpy), used by ``build_embedded_figure_blender.py``.

Source drawings (owner, 2026-10-10): the context figure ``011a.png`` (a 300 px square with eight lines in a 500 x 500
image) and the target ``011.png`` (a triangle). The lines below are the centre lines of the anti-aliased strokes of
011a.png, least-squares fitted away from the crossings (residual <= 0.03 px, the 30-degree diagonal G 1.6 px). The
target is the triangle bounded by G (its 30-degree long side), C (its short side) and E (its base): angles 30.02 / 75.53 /
74.46 deg against 30.00 / 75.53 / 74.47 deg in 011.png, whose triangle is 0.992 of its size (1.5 px of stroke width).

Owner 2026-10-10 18:07: "ONLY the black lines exist, at same thickness, and white space is just blank. And it should be
split into parts so that the target shape can be lifted (it should not contain lines like the target piece - those
lines should be separate parts)"; 2026-10-11: "it's 4 mm wide, so it can't be assisted grasp. I want it to be 12 mm wide,
but the overall size should be the same". So every drawn line is a black bar ``BAR_W`` (12 mm) wide and ``BAR_H`` tall
(the same for every line and every part) centred on its line of the 20 cm square, and nothing fills the cells. The
figure lies on the standard disembedding source tray (no tray of its own). The parts:

* ``isgefa`` target - the triangle's outline only: its three sides as bars, mitred at the corners, every corner that
  would reach the border bars (the 30-degree corner and the apex) cut ``CLEARANCE`` short of them;
* ``isgefb`` context frame - the square's border and every other bar outside the triangle, one connected piece; where a
  line meets or continues past the target it stops ``CLEARANCE`` short of the target's bars;
* ``isgefc`` / ``isgefd`` - the pieces of lines A and B that cross the triangle's inside, loose parts that stay on the
  tray when the target is lifted.

The empty cells beside the target's bars leave room for the fingers (8.4 mm thick, 14 mm wide over their last 30 mm):
e.g. a pinch across the base bar between the two inner pieces (``grasp_hint``).
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------------------------------------------------------
# the drawing (pixels of the 500 x 500 source images, y down)
# ----------------------------------------------------------------------------------------------------------------------

#: the border square's centre lines
SIDES = {"left": ("x", 0.0, 100.03), "right": ("x", 0.0, 400.03), "top": ("y", 0.0, 100.03), "bottom": ("y", 0.0, 400.03)}
#: ("x", a, b): x = a * y + b (near-vertical); ("y", a, b): y = a * x + b
FITS = {
    "A": ("x", -0.02368, 156.989),
    "B": ("x", 0.0, 224.028),
    "C": ("x", 0.25813, 233.054),  # the target's short side
    "D": ("x", -0.00750, 357.499),
    "E": ("y", 0.0, 209.033),  # the target's base
    "F": ("y", 0.0, 335.500),
    "G": ("y", -0.57775, 269.681),  # the 30-degree diagonal: the target's long side
    "H": ("y", -0.21021, 332.919),
}
#: where each drawn line starts and ends (a side of the square or another line)
ENDS = {
    "A": ("top", "bottom"),
    "B": ("top", "bottom"),
    "C": ("top", "bottom"),
    "D": ("top", "bottom"),
    "E": ("left", "right"),
    "F": ("left", "right"),
    "G": ("left", "top"),
    "H": ("A", "right"),
}
SQUARE_PX = (100.03, 400.03)  # x (and y) of the left (top) and right (bottom) sides
#: the target triangle's corners as line intersections, counter-clockwise in the frame (left, right, apex) and the line of
#: the side that starts at each corner
TARGET_CORNERS = [("G", "E"), ("C", "E"), ("G", "C")]
TARGET_SIDES = ["E", "C", "G"]

TARGET, FRAME, INNER_A, INNER_B = "isgefa", "isgefb", "isgefc", "isgefd"
INNER = {INNER_A: "A", INNER_B: "B"}

# ----------------------------------------------------------------------------------------------------------------------
# physical design (metres)
# ----------------------------------------------------------------------------------------------------------------------

SQUARE_M = 0.200  # the drawing's 300 px square -> 20 cm (2/3 mm per source pixel)
M_PER_PX = SQUARE_M / (SQUARE_PX[1] - SQUARE_PX[0])
BAR_W = 0.012  # every line (owner 2026-10-11: 4 mm bars were too thin for the assisted grasp)
BAR_H = 0.020  # every part: the pads (which end 6.9 mm before the fingertips) close on ~10 mm of a bar
CLEARANCE = 0.0003  # gap where a context bar meets the target (~0.3 px for the head camera: the lines read unbroken)
MIN_PIECE_AREA = 4e-6  # m^2: slivers below this (the stubs of E and G left of the target's 30-degree corner, inside the border bar) are dropped
DENSITY = 421.0  # kg/m^3: the ARAT wooden block (6.5 g at 2.5 cm), as the Kohs block


def _line_point_dir(name):
    kind, a, b = (SIDES.get(name) or FITS[name])
    if kind == "x":
        return (b, 0.0), (a, 1.0)
    return (0.0, b), (1.0, a)


def intersect(n1: str, n2: str):
    """Intersection (px) of two named lines / sides."""
    (x1, y1), (dx1, dy1) = _line_point_dir(n1)
    (x2, y2), (dx2, dy2) = _line_point_dir(n2)
    det = dx1 * (-dy2) - (-dx2) * dy1
    if abs(det) < 1e-12:
        raise ValueError(f"{n1} and {n2} are parallel")
    t = ((x2 - x1) * (-dy2) - (-dx2) * (y2 - y1)) / det
    return (x1 + t * dx1, y1 + t * dy1)


def px_to_frame(p):
    """Source px -> the figure frame (m): origin at the square's centre, +x right, +y up the drawing (away from the robot
    when the figure lies table-aligned in front of it)."""
    c = 0.5 * (SQUARE_PX[0] + SQUARE_PX[1])
    return ((p[0] - c) * M_PER_PX, (c - p[1]) * M_PER_PX)


def segments():
    """Every drawn line except the border as its two end points in the frame (m)."""
    return {name: (px_to_frame(intersect(name, a)), px_to_frame(intersect(name, b))) for name, (a, b) in ENDS.items()}


# ----------------------------------------------------------------------------------------------------------------------
# convex polygon tools (CCW lists of (x, y))
# ----------------------------------------------------------------------------------------------------------------------


def signed_area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def ccw(poly):
    return list(poly) if signed_area(poly) > 0 else list(reversed(poly))


def clean(poly, eps: float = 1e-9):
    """Drop repeated and collinear vertices (clipping leaves them where a cut passes through a corner)."""
    pts = []
    for p in poly:
        if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > eps:
            pts.append(p)
    if len(pts) > 1 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= eps:
        pts.pop()
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > eps * eps:
            out.append(b)
    return out


def offset(poly, d: float):
    """Convex CCW polygon with every edge moved ``d`` inward (outward for d < 0)."""
    poly = clean(poly)
    n = len(poly)
    lines = []
    for i in range(n):
        (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        L = math.hypot(bx - ax, by - ay)
        nx, ny = -(by - ay) / L, (bx - ax) / L  # left normal = inside for CCW
        lines.append(((ax + d * nx, ay + d * ny), (bx - ax, by - ay)))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
        t = ((p2[0] - p1[0]) * (-d2[1]) - (-d2[0]) * (p2[1] - p1[1])) / det
        out.append((p1[0] + t * d1[0], p1[1] + t * d1[1]))
    return out


def keep_left(poly, a, b):
    """The part of a convex polygon left of the directed line a -> b (Sutherland-Hodgman, one edge)."""
    if not poly:
        return []

    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    out = []
    for j in range(len(poly)):
        p, q = poly[j], poly[(j + 1) % len(poly)]
        sp, sq = side(p), side(q)
        if sq >= 0:
            if sp < 0:
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            out.append(q)
        elif sp >= 0:
            t = sp / (sp - sq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    out = clean(out)
    return out if len(out) >= 3 and abs(signed_area(out)) > 1e-14 else []


def clip_convex(subject, clip):
    out = list(subject)
    for i in range(len(clip)):
        out = keep_left(out, clip[i], clip[(i + 1) % len(clip)])
        if not out:
            return []
    return out


def bar(p0, p1, half_w: float):
    """Rectangle of half width ``half_w`` around the segment p0 -> p1 (butt ends), CCW."""
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    nx, ny = -(p1[1] - p0[1]) / L * half_w, (p1[0] - p0[0]) / L * half_w
    return ccw([(p0[0] - nx, p0[1] - ny), (p1[0] - nx, p1[1] - ny), (p1[0] + nx, p1[1] + ny), (p0[0] + nx, p0[1] + ny)])


def touches(p, q, tol: float = 1e-7) -> bool:
    return bool(clip_convex(offset(p, -tol), offset(q, -tol)))


def distance(p, q) -> float:
    """Minimum distance between two convex polygons (0 when they overlap)."""
    if clip_convex(p, q):
        return 0.0

    def point_segment(pt, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / (dx * dx + dy * dy)))
        return math.hypot(pt[0] - a[0] - t * dx, pt[1] - a[1] - t * dy)

    best = math.inf
    for u, v in ((p, q), (q, p)):
        for pt in u:
            for k in range(len(v)):
                best = min(best, point_segment(pt, v[k], v[(k + 1) % len(v)]))
    return best


def triangle_angles(poly):
    out = []
    for i in range(3):
        a, b, c = poly[i], poly[(i + 1) % 3], poly[(i - 1) % 3]
        u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1])
        out.append(math.degrees(math.acos((u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v)))))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# the parts
# ----------------------------------------------------------------------------------------------------------------------


def target_triangle():
    """The target's centre-line triangle in the frame (m), CCW from the 30-degree corner."""
    return ccw([px_to_frame(intersect(a, b)) for a, b in TARGET_CORNERS])


def inner_limit() -> float:
    """The target stays inside |x|, |y| <= this: ``CLEARANCE`` short of the border bars' inner faces (its corners that
    would reach them are cut flat)."""
    return SQUARE_M / 2 - BAR_W / 2 - CLEARANCE


def parts():
    """{model: [(piece name, convex piece as a CCW polygon in the frame, m)]} for the target, the context frame and the
    two inner pieces."""
    h, c = BAR_W / 2, CLEARANCE
    tri = target_triangle()
    outer, inner = offset(tri, -h), offset(tri, h)
    big = 10.0
    q = inner_limit()
    inside_border = [(-q, -q), (q, -q), (q, q), (-q, q)]
    target = []
    for i in range(3):
        trap = [outer[i], outer[(i + 1) % 3], inner[(i + 1) % 3], inner[i]]
        piece = clip_convex(trap, inside_border)
        if piece:
            target.append((f"side_{TARGET_SIDES[i]}", piece))
    # context regions around the target dilated by the clearance: outside each side of the outer triangle (a partition of
    # its outside into three convex regions), and the inside of the inner triangle
    oc, ic = offset(tri, -(h + c)), offset(tri, h + c)
    world = [(-big, -big), (big, -big), (big, big), (-big, big)]

    def outside(region, i):  # right of oc's edge i
        return keep_left(region, oc[(i + 1) % 3], oc[i])

    def inside(region, i):
        return keep_left(region, oc[i], oc[(i + 1) % 3])

    regions = {"outside_E": outside(world, 0), "outside_C": outside(inside(world, 0), 1), "outside_G": outside(inside(inside(world, 0), 1), 2)}
    s = SQUARE_M / 2 + h
    frame = [("border_top", ccw([(-s, SQUARE_M / 2 - h), (s, SQUARE_M / 2 - h), (s, s), (-s, s)])),
             ("border_bottom", ccw([(-s, -s), (s, -s), (s, -SQUARE_M / 2 + h), (-s, -SQUARE_M / 2 + h)])),
             ("border_left", ccw([(-s, -s), (-SQUARE_M / 2 + h, -s), (-SQUARE_M / 2 + h, s), (-s, s)])),
             ("border_right", ccw([(SQUARE_M / 2 - h, -s), (s, -s), (s, s), (SQUARE_M / 2 - h, s)]))]
    out = {TARGET: target, FRAME: frame, INNER_A: [], INNER_B: []}
    inner_line = {line: model for model, line in INNER.items()}
    for name, (p0, p1) in segments().items():
        rect = bar(p0, p1, h)
        for region_name, region in regions.items():
            piece = clip_convex(rect, region)
            if piece and signed_area(piece) >= MIN_PIECE_AREA:
                out[FRAME].append((f"{name}_{region_name}", piece))
        piece = clip_convex(rect, ic)
        if piece and signed_area(piece) >= MIN_PIECE_AREA:
            if name not in inner_line:
                raise RuntimeError(f"line {name} crosses the target's inside but has no part")
            out[inner_line[name]].append((f"{name}_inside", piece))
    return out


def check(parts_):
    """Clearances between parts, the frame's connectivity, convexity; raises on a violation."""
    polys = {m: [p for _, p in ps] for m, ps in parts_.items()}
    h_ok = all(len(p) >= 3 for ps in polys.values() for p in ps)
    models = list(polys)
    min_gap = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            min_gap[f"{models[i]}/{models[j]}"] = round(min(distance(p, q) for p in polys[models[i]] for q in polys[models[j]]), 7)
    frame = polys[FRAME]
    seen, todo = {0}, [0]
    while todo:
        k = todo.pop()
        for m in range(len(frame)):
            if m not in seen and touches(frame[k], frame[m]):
                seen.add(m)
                todo.append(m)
    rep = {"pieces_have_area": h_ok, "min_gap_between_parts_m": min_gap, "frame_connected": len(seen) == len(frame),
           "pieces": {m: [n for n, _ in ps] for m, ps in parts_.items()}}
    rep["ok"] = h_ok and min(min_gap.values()) >= CLEARANCE * (1 - 1e-6) and rep["frame_connected"]
    if not rep["ok"]:
        raise RuntimeError(f"embedded figure parts check failed: {rep}")
    return rep


def grasp_hint(parts_):
    """A pinch across the target's base bar midway between the two inner pieces, closing along y: where, and how much
    free room the cells leave on each side (along y) and along the bar (between the inner pieces / the bars below)."""
    tri = target_triangle()
    a_x = max(x for _, p in parts_[INNER_A] for x, _ in p)
    b_x = min(x for _, p in parts_[INNER_B] for x, _ in p)
    x0 = 0.5 * (a_x + b_x)
    y_base = tri[0][1]

    def spans_at(x):
        out = []
        for model, ps in parts_.items():
            for _, p in ps:
                ys = []
                for k in range(len(p)):
                    (x1, y1), (x2, y2) = p[k], p[(k + 1) % len(p)]
                    if (x1 - x) * (x2 - x) <= 0 and x1 != x2:
                        ys.append(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
                if ys:
                    out.append((min(ys), max(ys), model))
        return sorted(out)

    spans = spans_at(x0)
    k = next(i for i, (lo, hi, m) in enumerate(spans) if m == TARGET and lo <= y_base <= hi)
    below = spans[k][0] - max(hi for lo, hi, m in spans[:k])
    above = min(lo for lo, hi, m in spans[k + 1:]) - spans[k][1]
    return {"pinch_centre_figure_m": [round(x0, 5), round(y_base, 5)], "closing_axis": "y (across the base bar)", "bar_width_m": BAR_W,
            "free_inside_m": round(above, 4), "free_outside_m": round(below, 4), "free_along_bar_m": round(b_x - a_x, 4)}


if __name__ == "__main__":
    import json

    ps = parts()
    print(json.dumps(check(ps), indent=1))
    for m, pieces in ps.items():
        print(m, len(pieces), "pieces, area cm2 %.3f" % (sum(signed_area(p) for _, p in pieces) * 1e4))
    print("target angles", [round(a, 2) for a in triangle_angles(target_triangle())])
    print("grasp", grasp_hint(ps))
