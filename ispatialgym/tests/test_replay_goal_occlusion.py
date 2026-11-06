"""The replay stage's goal-occlusion screen (owner 2026-11-05: the sampler screens mental_rotation / disembedding, the replay stage every
other family): which demos ``multienv.replay`` screens, and the instance record it reads (simulator-free)."""
import json

import pytest

h5py = pytest.importorskip("h5py")

from ispatialgym.multienv import replay as RP


def _rec(summary=None, subtask="inter_object_relation_satisfaction-nextto_small_scale", variant=None):
    return {"summary": summary, "subtask": subtask, "variant": variant}


def test_auto_screens_pending_and_unmeasured_and_skips_what_the_sampler_measured():
    assert RP.needs_occlusion_screen("auto", _rec({"visibility": "pending"}))[0]
    assert RP.needs_occlusion_screen("auto", _rec({}))[0]  # no measured report
    assert RP.needs_occlusion_screen("auto", _rec({"goal_occlusion": {"skipped": "no camera"}}))[0]
    measured = {"goal_occlusion": {"max_occluded": 0.12, "occluded": False}}
    assert RP.needs_occlusion_screen("auto", _rec(measured)) == (False, "measured by the sampler")
    assert RP.needs_occlusion_screen("on", _rec(measured))[0]
    assert not RP.needs_occlusion_screen("off", _rec({"visibility": "pending"}))[0]


def test_camera_relocalization_is_exempt_in_every_mode():
    for mode in ("auto", "on"):
        assert RP.needs_occlusion_screen(mode, _rec({"visibility": "pending"}, subtask="camera_relocalization-obstructed")) == (False, "exempt subtask")
    # the variant names the subtask when the meta has none
    assert not RP.needs_occlusion_screen("on", _rec(None, subtask=None, variant="camera_relocalization-standard"))[0]


def test_instance_record_reads_the_demo_meta_and_the_summary_next_to_the_tro(tmp_path):
    inst = tmp_path / "inst" / "instances"
    inst.mkdir(parents=True)
    tro = inst / "a__0000-tro_state.json"
    tro.write_text("{}")
    (inst / "a__0000.summary.json").write_text(json.dumps({"visibility": "pending"}))
    demo_dir = tmp_path / "demo" / "a__0000"
    demo_dir.mkdir(parents=True)
    (demo_dir / "meta.json").write_text(json.dumps({"variant": "v", "subtask": "s"}))

    class D:
        dir = str(demo_dir)
        tro_path = str(tro)

    rec = RP.instance_record(D())
    assert rec == {"summary": {"visibility": "pending"}, "variant": "v", "subtask": "s"}
    D.tro_path = None
    assert RP.instance_record(D())["summary"] is None
