"""i7-attachdef 2026-11-08 (lead decision: sampler-only bug fix, default off): ``ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS`` - the instance
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
    json.dump({"instance_id": iid}, open(CORE.instance_summary_path(tro), "w"))
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


def test_switch_off_writes_the_summary_in_place_as_before(tmp_path, monkeypatch, mods):
    CORE, RI = mods
    monkeypatch.delenv("ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS", raising=False)
    root = str(tmp_path / "inst")
    tro = _write_attempt(root, "a__300003", CORE)
    assert [i for i, _ in RI.queue_ready(root)] == ["a__300003"]  # ready at once (the window the switch closes)
    assert not CORE.publish_instance(tro)
    assert CORE.instance_summary_path(tro).endswith(".summary.json")
