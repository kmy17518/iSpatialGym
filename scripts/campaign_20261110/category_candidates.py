#!/usr/bin/env python3
"""Propose category/room holdouts from actual local ready assets; no release claim."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "tmp/campaign_20261110/inventory"
    inventory = json.loads((out / "category_inventory.json").read_text())
    used = set(inventory["used_standard_categories"]) - {"walls", "floors"}
    by_category = inventory["categories"]
    def obj(category, model):
        matches = [o for o in by_category[category] if o["model"] == model]
        if not matches or not matches[0]["usd_present"]:
            raise ValueError(f"Missing ready source asset: {category}/{model}")
        if not matches[0]["ready"]:
            raise ValueError(f"KB not ready: {category}/{model}")
        if category in used and category != "tray":
            raise ValueError(f"Category leakage: {category}")
        return matches[0]
    support = obj("pedestal_table", "tdpuit")
    common = {"scene": "office_cubicles_right", "room": "conference_hall_0", "support": support,
              "status": "candidate_only", "validation": "not_started"}
    rows = []
    def add(task, target=None, partner=None, note="", support_needed=True, **extra):
        row = {**common, "task": task, "note": note, **extra}
        if not support_needed:
            row.pop("support")
        if target:
            row["target"] = obj(*target)
        if partner:
            row["partner"] = obj(*partner)
        rows.append(row)
    add("camera_relocalization-standard", note="Reserve whole task room; verify 5–10 m path bucket before accepting.", support_needed=False)
    add("camera_relocalization-obstructed", ("footstool", "sghekm"), note="Common household obstacle; reserve footstool category globally.", support_needed=False)
    add("object_scaling", ("tennis_ball", "rgekxe"), note="6.7 cm graspable round object; requires stable grasp validation.", support_needed=False)
    add("mental_rotation", ("hammer", "fcrlbo"), note="Distinct head/handle orientation; check full 3D symmetry and viable grasps.", support_needed=False)
    add("disembedding", ("rubber_eraser", "cungod"), note="Reserve crayon category for distractors; source/destination remain tray per owner exception. Tray models must differ from held-in and model-holdout allocations.",
        distractors=[obj("crayon", "coapeh"), obj("crayon", "csglmn")], tray_model_allocation="pending_global_model_reservation")
    add("alignment-world_vertical", ("poster", "dsrcyt"), note="Readable texture/upright direction must be reviewed; walls exempt but room held out. Current asset 0.62×0.89 m; consider fixed canonical rescaling if required.", support_needed=False,
        structural_support="opaque wall in held-out room; exempt category/model")
    add("alignment-axial", ("calculator", "kwmmty"), note="Button/display orientation establishes a unique top; use ready nightstand/roxnbe if pedestal top is not rectangular.")
    rows[-1]["support"] = obj("nightstand", "roxnbe")
    add("configuration_matching-block_design", note="Owner: Wechsler category, exactly one category-heldout instance; 4 identical blocks and a compatible new board model. Tray/board category exempt, model still held out.",
        generated_target="wechsler_block_design", board_model_allocation="new_compatible_board_required")
    add("configuration_matching-attachment", ("dip_candle", "poexjo"), ("candle_holder", "wiufnv"),
        note="Exact matching dipcandle M/F attachment port in object_inventory.json; tabletop pair, both categories unused.")
    for direction in ("open", "close"):
        add(f"configuration_matching-articulation_{direction}_small_scale", ("hinged_jar", "jnjtrl"), note="One lid joint; 2 category-heldout instances per task.")
        add(f"configuration_matching-articulation_{direction}_large_scale", ("wardrobe", "brimns"), note="Ready 1.9 m tall asset with 5 links; add to heldout room, select exactly one reachable door; 3 instances per task.", support_needed=False)
    prefix = "inter_object_relation_satisfaction-"
    add(prefix + "ontop_small_scale", ("rubber_eraser", "cungod"), ("notepad", "awcyha"), note="Flat eraser on notebook; support geometry check still required.")
    add(prefix + "under_small_scale", ("notepad", "awcyha"), ("calculator", "kwmmty"), note="Explicit BDDL order: notepad under calculator. Check target pickup of 27 mm thick notepad.")
    add(prefix + "inside_small_scale", ("rubber_eraser", "cungod"), ("pencil_holder", "jyetcl"), note="5 cm eraser in 11.3 cm holder; convex hull/interior clearance must be measured.")
    add(prefix + "nextto_small_scale", ("crayon", "coapeh"), ("ruler", "gajhql"), note="Long axes parallel, tips/top same direction where meaningful; random admissible spacing.")
    add(prefix + "touching_small_scale", ("box_of_crackers", "cmdigf"), ("box_of_cookies", "fsdjun"), note="Stable broad faces touch; require geometric goal/size compatibility.")
    add(prefix + "ontop_large_scale", ("alarm_clock", "vqwovi"), ("pedestal_table", "tdpuit"), note="Reuse heldout table as partner; no extra support role.", support_needed=False)
    add(prefix + "under_large_scale", ("shoe_box", "pitwey"), ("pedestal_table", "tdpuit"), note="Check table's actual under-clearance around pedestal; short straight push with broad box face; fixed goal orientation selected before grasping.", support_needed=False)
    add(prefix + "inside_large_scale", ("comic_book", "nekxsh"), ("wardrobe", "brimns"), note="Reuse articulated wardrobe as interior partner; shelf/lean geometry needs probe.", support_needed=False)
    add(prefix + "nextto_large_scale", ("armchair", "pmrcex"), ("pedestal_table", "tdpuit"), note="Straight push, same start/goal orientation for large target; footprint clearance pending.", support_needed=False)
    add(prefix + "touching_large_scale", ("footstool", "sghekm"), ("armchair", "pmrcex"), note="Footstool broad edge against chair front, straight push; reuse heldout obstacle category.", support_needed=False)
    assert len(rows) == 23
    data = {"status": "proposed_not_validated", "decision_basis": "owner structural and tray/board category exceptions",
            "primary_room": {"scene": common["scene"], "room": common["room"], "reason": "Unused standard task room; few existing nonstructural objects; add shared pedestal table and wardrobe."},
            "room_fallbacks": [{"scene": "house_double_floor_upper", "room": "bedroom_2", "note": "Existing nightstand category unused; needs space/collision review."},
                               {"scene": "Rs_int", "room": "kitchen_0", "note": "Unused task room with microwave/oven/top_cabinet categories available for large articulation."}],
            "support_fallbacks": [obj("nightstand", "roxnbe"), obj("pedestal_table", "xcnvsy")], "templates": rows}
    (out / "category_holdout_proposal.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# Category and room holdout proposal", "", "**Candidates only: no geometry, demo, or open-loop replay pass is claimed.**", "",
             "Reserve office_cubicles_right / conference_hall_0. Share pedestal_table/tdpuit as tabletop support and large-relation partner; add wardrobe/brimns for large articulation and inside. Use nightstand/roxnbe for axial alignment. The room has not appeared as a standard task room. Its navigable path lengths and placements still need simulation checks.", "",
             "The ready 1.249 × 0.882 × 0.663 m pedestal table is the first support candidate. Under-clearance and support surface geometry must be checked. The initially considered console tables and all lab tables fail the authoritative KB ready rule, so they are excluded. No native support is assumed to exist in this room.", "",
             "| Task | Target | Partner / other roles |", "|---|---|---|"]
    for row in rows:
        target = row.get("target", {})
        partner = row.get("partner", {})
        t = f"{target['category']}/{target['model']}" if target else row.get("generated_target", "room")
        p = f"{partner['category']}/{partner['model']}" if partner else ("crayon/coapeh,csglmn; reserved trays" if "distractors" in row else "shared support or structural wall")
        lines.append(f"| {row['task']} | {t} | {p} |")
    lines += ["", "Each selected category is absent from all standard task roles. Walls/floors are exempt; trays and boards have only a category exemption and still need unseen models. Every category used here must be excluded from training and model-holdout construction before candidate expansion.", "",
              "Room fallbacks: house_double_floor_upper/bedroom_2 (natural nightstand), Rs_int/kitchen_0 (native microwave/oven/top_cabinet). Using a fallback reserves another room globally."]
    (out / "CATEGORY_HOLDOUT_REVIEW.md").write_text("\n".join(lines) + "\n")
    print(out / "CATEGORY_HOLDOUT_REVIEW.md")


if __name__ == "__main__":
    main()
