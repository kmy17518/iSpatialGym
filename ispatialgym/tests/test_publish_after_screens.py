"""i7-attachdef 2026-11-08 (lead decision: sampler-only bug fix; the default since the owner's 5j, 2026-11-08): ``ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS`` - the instance
summary (the collector queue's ready marker, ``run_isg.queue_ready``) is written as ``.pending`` and renamed only after the post-write
screens pass, so a rejected attempt is never claimable and an accepted one is claimable exactly once."""

from __future__ import annotations

import json
import os

import pytest


def _write_attempt(inst_root, iid, CORE):
    """What ``write_instance`` leaves on disk: TRO, sidecar, summary (pending under the switch), template meta."""
    d = os.path.join(inst_root, "variant", "template", "instances")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(inst_root, "variant", "template", "template_meta.json"), "w").write("{}")
    tro = os.path.join(d, f"{iid}-tro_state.json")
    json.dump({"ispatialgym": {}}, open(tro, "w"))
    json.dump({}, open(os.path.join(d, f"{iid}.oracle.json"), "w"))
    sp = CORE.instance_summary_path(tro)
    os.makedirs(os.path.dirname(sp), exist_ok=True)  # as write_instance does (the pending sub-directory)
    json.dump({"instance_id": iid}, open(sp, "w"))
    return tro


@pytest.fixture
def mods(monkeypatch):
    from ispatialgym import fastflags
    from ispatialgym.oracle import run_isg as RI
    from ispatialgym.sampling.v2 import core as CORE

    monkeypatch.setattr(fastflags, "_seen", {}, raising=False)
    return CORE, RI


def test_a_rejected_attempt_is_never_claimable(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "1")
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, "a__300001", CORE)
    assert RI.queue_ready(root) == []  # written, screens not yet run: the summary is pending - not ready
    removed = CORE.remove_instance_files(tro)  # a screen rejects (goal_occluded / repeats_eval)
    assert len(removed) == 3 and RI.queue_ready(root) == []
    assert not CORE.publish_instance(tro)  # nothing left to publish


def test_an_accepted_attempt_is_claimable_exactly_once(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "1")
    root = str(tmp_path / "inst")
    claims = str(tmp_path / "claims")
    os.makedirs(claims)
    tro = _write_attempt(root, "a__300002", CORE)
    assert RI.queue_ready(root) == []
    assert CORE.publish_instance(tro)  # the screens passed
    ready = RI.queue_ready(root)
    assert [i for i, _ in ready] == ["a__300002"] and ready[0][1] == tro
    assert RI._claim(claims, "a__300002") and not RI._claim(claims, "a__300002")


def test_switch_is_on_by_default(monkeypatch, mods):
    CORE, _ = mods
    monkeypatch.delenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", raising=False)
    monkeypatch.delenv("ISPATIALGYM_FAST", raising=False)
    assert CORE.publish_after_screens()  # owner 2026-11-08 (5j): the default
    assert CORE.instance_summary_path("x/a__1-tro_state.json") == os.path.join("x", ".pending", "a__1.summary.json")


def test_switch_off_writes_the_summary_in_place_as_before(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "0")
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, "a__300003", CORE)
    assert [i for i, _ in RI.queue_ready(root)] == ["a__300003"]  # ready at once (the window the switch closes)
    assert not CORE.publish_instance(tro)
    assert CORE.instance_summary_path(tro) == tro[: -len("-tro_state.json")] + ".summary.json"
    assert not os.path.exists(os.path.join(os.path.dirname(tro), CORE.PENDING_DIR))  # nothing pending, no sub-directory


# integration-2 2026-11-08 (lead decision #18): the pending summary in ``instances/.pending/`` under its final name - the old
# ``<iid>.summary.json.pending`` pushed disembedding_07's 254-character summary name to 262 (> 255, OSError Errno 36).
#: disembedding_07 300038 (int2_4, 18385340): 240 characters, its TRO name 255
LONG_IID = ("disembedding-cellulose_tape-cork_1_coaster_1_cap_1-conference_table__office_cubicles_right__support-conference_table_fpkmyw_0-T2"
            "__destination-tray-hvlfig__source-tray-uekqey__target-cellulose_tape-kavsnx__distractors-cap_1-~5cc2e399__300038")


def test_the_longest_instance_id_publishes(tmp_path, monkeypatch, mods):
    """The longest id a TRO can carry (``<iid>-tro_state.json`` <= 255 characters: 240; disembedding_07's is exactly that) - the old
    pending name (``.summary.json.pending``, 261) would not fit; its summary file name in ``.pending/`` is 253."""
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "1")
    iid = (LONG_IID + "x" * 255)[: 255 - len("-tro_state.json")]
    assert len(iid) == 240 and len(iid + ".summary.json.pending") > 255  # the old pending name would not fit
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, iid, CORE)
    assert RI.queue_ready(root) == []
    assert CORE.publish_instance(tro)
    assert [i for i, _ in RI.queue_ready(root)] == [iid]
    assert os.listdir(os.path.join(os.path.dirname(tro), CORE.PENDING_DIR)) == []


def test_the_disembedding_07_instance_publishes(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "1")
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, LONG_IID, CORE)
    assert CORE.publish_instance(tro) and [i for i, _ in RI.queue_ready(root)] == [LONG_IID]


def test_a_rejected_attempt_leaves_nothing_pending(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "1")
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, "a__300004", CORE)
    pend = os.path.join(os.path.dirname(tro), CORE.PENDING_DIR)
    assert os.listdir(pend) == ["a__300004.summary.json"]
    CORE.remove_instance_files(tro)
    assert os.listdir(pend) == [] and os.listdir(os.path.dirname(tro)) == [CORE.PENDING_DIR] and RI.queue_ready(root) == []


def test_an_old_style_pending_summary_is_removed_and_never_ready(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.setenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", "0")
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, "a__300005", CORE)
    old = CORE.instance_summary_path(tro, False) + CORE.PENDING_SUFFIX
    os.replace(CORE.instance_summary_path(tro, False), old)  # a pending summary written by a snapshot before the fix
    assert RI.queue_ready(root) == []
    assert old in CORE.remove_instance_files(tro) and not os.path.exists(old)
