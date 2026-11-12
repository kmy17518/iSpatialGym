from ispatialgym.oracle.campaign_articulated_residual import next_correction


def test_recorded_short_close_can_continue_within_original_budget():
    assert next_correction(.0698, .106, .05, .02, .106, .3, 1, 1.) == .0698
    assert next_correction(.04, .0698, .05, .02, .1758, .3, 2, 1.) is None


def test_stop_at_budget_round_limit_or_no_progress():
    assert next_correction(.0698, .106, .05, .02, .24, .3, 1, 1.) is None
    assert next_correction(.06, .08, .05, .02, .18, .3, 3, 1.) is None
    assert next_correction(.0698, .0701, .05, .02, .106, .3, 1, 1.) is None


def test_never_pull_or_continue_unknown_measurement():
    assert next_correction(-.0698, .106, .05, .02, .106, .3, 1, 1.) is None
    assert next_correction(float('nan'), .106, .05, .02, .106, .3, 1, 1.) is None
    assert next_correction(.0698, .106, None, .02, .106, .3, 1, 1.) is None
    assert next_correction(-.0698, -.106, .05, .02, .106, .3, 1, -1.) == -.0698
