Embedded figure, EFT item 11 (iSpatialGym disembedding) - 12 mm bars, built 2026-10-11

WHAT IT IS
  The context figure of 011a.png made only of its lines: black bars 12 mm wide and 20 mm tall, centred on the lines of
  a 20 cm square (outer size 212 x 212 mm). The target (the triangle of 011.png) is its own part and lifts straight out.
  It has no tray of its own: lay it on the standard disembedding source tray (tray/uekqey, spawned at 1.25 scale);
  the target goes to the standard destination tray.

PARTS (category jigsaw_puzzle_piece, dataset ispatialgym-assets)
  isgefa  target: the triangle outline only               126 x 72 x 20 mm     29.3 g
  isgefb  context frame: the border + every bar outside   212 x 212 x 20 mm   172.8 g
          the triangle, one rigid piece
  isgefc  context: the piece of line A inside the         12 x 8 x 20 mm        0.5 g
          triangle (loose)
  isgefd  context: the piece of line B inside the         12 x 36 x 20 mm       3.3 g
          triangle (loose)

FOLDERS
  ispatialgym-assets/   OmniGibson-ready assets (USD + misc metadata). Merge into
                        <OMNIGIBSON_DATA_PATH>/ispatialgym-assets/ (objects/jigsaw_puzzle_piece/isgef*).
  layout/assembly.json  Where each part goes. origin_in_figure_m = the part's origin (its bounding-box centre in x / y,
                        bottom at z = 0) in the figure frame: origin at the square's centre on the tray floor, +x right,
                        +y up the drawing (away from the robot), z up; every part keeps the figure frame's orientation.
                        Also: the clearance check and a grasp hint for the target.
  source/               Editable source, laid out as in the repo (ispatialgym/assets/objects/embedded_figure/): model/ with
                        per part the URDF, OBJ meshes (visual + convex collision pieces) and build_manifest.json; the
                        Blender file; the scripts that build (Blender 4.5), render and convert (OmniGibson) it.
                        figure_geometry.py holds every dimension and the lines fitted from 011a.png. Rebuild:
                          <blender> --background --python build_embedded_figure_blender.py
                          <blender> --background --python render_previews_blender.py
                          OMNIGIBSON_HEADLESS=1 python convert_to_omnigibson.py   (also copies into the repo's
                          ispatialgym/assets/datasets/ when run from the repo)
  previews/             Renders: blender/ (the asset files on a white sheet), omnigibson/ (on tray/uekqey x1.25,
                        report.json = the physics check), embedded_figure_vs_reference.png (vs 011a.png / 011.png).

LOAD (OmniGibson)
  from omnigibson.objects import DatasetObject
  target = DatasetObject(name="target", category="jigsaw_puzzle_piece", model="isgefa", dataset_name="ispatialgym-assets")
  Place every part at  figure_pose * (x, y, 0)  from layout/assembly.json, where figure_pose is the square's centre on the
  source tray's floor (identity orientation = the drawing upright, +y away from the robot).

GRASP HINT
  Pinch across the target's base bar at (x, y) = (-41.2, 27.3) mm in the figure frame, closing along y.
  Free room: 19.2 mm inside the triangle, 44.1 mm outside, 35.6 mm along the bar (the R1Pro fingers are 8.4 mm thick
  and 14 mm wide over their last 30 mm).

CHECKS (2026-10-11)
  Geometry: every gap between the target and the context parts is 0.3 mm; the frame is one connected piece.
  Meshes: visual meshes closed, collision pieces convex; URDF mass / centre of mass / inertia from the closed mesh
  (wood 421 kg/m^3, painted black).
  OmniGibson on tray/uekqey x1.25: the parts settle (0.5 mm release, <= 0.18 deg tilt); lifting the target 6 cm moves
  the other parts <= 0.01 mm.
