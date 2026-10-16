import os

import pytest
import yaml

from ispatialgym import run_isg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO, "configs", "eval.yaml")
REQUIRED = ["--task", "alignment_axial", "--policy", "ispatialgym.eval.policy:HoldPolicy"]
EVALUATION_CONDITIONS = (
    "mode",
    "max_steps",
    "wallclock",
    "settle_eval",
    "settle_steps",
    "success_metric",
    "tolerance_scale",
    "image_size",
    "head_aperture",
)


def test_default_config_is_the_standard_track():
    defaults = vars(run_isg.parse_args(REQUIRED))
    configured = vars(run_isg.parse_args(REQUIRED + ["--config", DEFAULT_CONFIG]))
    assert {k: configured[k] for k in EVALUATION_CONDITIONS} == {k: defaults[k] for k in EVALUATION_CONDITIONS}
    assert configured["record_view"] == ("head", "external")


def test_command_line_overrides_the_config(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        yaml.safe_dump({"max_steps": 30, "settle_eval": True, "policy_kwargs": {"steps": 3}, "record_view": ["head"]})
    )
    args = run_isg.parse_args(REQUIRED + ["--config", str(config), "--max-steps", "7"])
    assert (args.max_steps, args.settle_eval, args.policy_kwargs, args.record_view) == (
        7,
        "on",
        {"steps": 3},
        ("head",),
    )


def test_run_options_round_trip_through_a_config(tmp_path):
    args = run_isg.parse_args(
        REQUIRED + ["--record-view", "head,tiled", "--policy-kwargs", '{"steps": 2}', "--wallclock", "9"]
    )
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(run_isg.run_options(args)))
    child = run_isg.parse_args(["--config", str(config), "--instance", "x-tro_state.json"])
    assert run_isg.run_options(child) == run_isg.run_options(args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--policy", "m:p"],
        ["--task", "a", "--instance", "b-tro_state.json", "--policy", "m:p"],
        ["--task", "a"],
        REQUIRED + ["--mode", "replay"],
        REQUIRED + ["--record-view", "head,top"],
    ],
)
def test_invalid_arguments_exit(argv):
    with pytest.raises(SystemExit):
        run_isg.parse_args(argv)


def test_unknown_config_keys_are_rejected(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("max_step: 3\n")
    with pytest.raises(SystemExit, match="unknown option"):
        run_isg.parse_args(REQUIRED + ["--config", str(config)])
