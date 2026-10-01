"""first-demo v3.0 (owner findings on v2.2): each fix switches on only in its situation, detected at run time.

F4 (cork 0928-041525): an attach place keeps the holding arm out of the jam probe from the descent until the attachment exists.
"""
from __future__ import annotations

from types import SimpleNamespace

from ispatialgym.oracle import execute as EX


# ------------------------------------------------------------------------------------------------ F4


def _attach_setup(kind="attach", on="wine_bottle.n.01_1", held=True, kept=()):
    motions = SimpleNamespace(keep_hold_arms=set(kept))
    part = object()
    state = SimpleNamespace(held_object=lambda arm: part if held else None)
    action = SimpleNamespace(kind=kind, on=on, arm="right", object="cork.n.04_1")
    return motions, state, action


def test_f4_attach_place_keeps_the_holding_arm_out_of_the_jam_probe_until_the_attachment():
    motions, state, action = _attach_setup()
    ctx = {}
    with EX.attach_hold_kept(action, motions, state, ctx):
        assert motions.keep_hold_arms == {"right"}
    assert motions.keep_hold_arms == set()
    assert ctx["attach_hold"]["fired"] and ctx["attach_hold"]["arm"] == "right"
    # an arm the executor already keeps (a final hold) stays kept after the place window
    motions, state, action = _attach_setup(kept=("right",))
    with EX.attach_hold_kept(action, motions, state, {}):
        pass
    assert motions.keep_hold_arms == {"right"}


def test_f4_attach_hold_released_when_the_place_fails_inside_the_window():
    motions, state, action = _attach_setup()

    def place():
        with EX.attach_hold_kept(action, motions, state, {}):
            yield 1
            raise RuntimeError("attach refused")

    gen = place()
    next(gen)
    assert motions.keep_hold_arms == {"right"}
    gen.close()  # the executor's rollback closes the action's generator mid-place
    assert motions.keep_hold_arms == set()


def test_f4_every_other_place_is_left_alone():
    for kw, why in (({"kind": "support"}, "not an attach place"), ({"on": None}, "not an attach place"), ({"held": False}, "the part is not in hand")):
        motions, state, action = _attach_setup(**kw)
        ctx = {}
        with EX.attach_hold_kept(action, motions, state, ctx):
            assert motions.keep_hold_arms == set()
        assert ctx["attach_hold"] == {"fired": False, "why": why}
