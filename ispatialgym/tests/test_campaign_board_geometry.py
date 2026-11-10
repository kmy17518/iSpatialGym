import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import block_design as bd


def original():
    path = Path(bd.__file__).resolve().parents[3] / "assets/objects/block_board/model/isgboard/build_manifest.json"
    return json.loads(path.read_text())


def test_custom_board_geometry_uses_actual_dimensions_and_loaded_scale():
    source = {**original(), "plate_m": .0948}
    geometry = bd.campaign_board_geometry(source, [.9, .9, 1.1])
    assert geometry["plate_m"] == pytest.approx(.0948*.9)
    assert geometry["plate_thickness_m"] == pytest.approx(.008*1.1)
    assert geometry["pocket_centres"][0] == pytest.approx([-.0157*.9, .0157*.9])
    assert geometry["outer_corner_sign"] == source["outer_corner_sign"]
    with pytest.raises(Reject, match="equal x/y"):
        bd.campaign_board_geometry(source, [.9, 1, 1])
    source.pop("pocket_pitch_m")
    with pytest.raises(Reject, match="pocket_pitch"):
        bd.campaign_board_geometry(source, [1, 1, 1])


def test_campaign_board_reads_its_own_metadata_and_pins_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_CONFIG", "campaign")
    monkeypatch.setattr(bd, "get_dataset_path", lambda name: str(tmp_path))
    s = object.__new__(bd.BlockDesign)
    s.choice = NS(models={"board": NS(dataset="private", category="tray", model="custom")})
    s.template_facts = {}
    s.obj = lambda role: NS(scale=th.ones(3))
    with pytest.raises(Reject, match="model-specific"):
        s._board_geometry()
    path = tmp_path / "objects/tray/custom/misc/isg_board.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**original(), "plate_m": .0928}))
    assert s._board_geometry()["plate_m"] == .0928
    evidence = s.template_facts["campaign_board_geometry"]
    assert evidence["path"] == str(path) and len(evidence["sha256"]) == 64
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG")
    assert s._board_geometry()["plate_m"] == pytest.approx(.0908)
