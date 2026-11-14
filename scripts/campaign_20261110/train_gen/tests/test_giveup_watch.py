import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('gw', pathlib.Path(__file__).parents[1] / 'giveup_watch.py')
gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)


def st(members, start=0.0, current=True):
    g = {'started_at': start, 'members': members}
    return {'groups': [], 'open': [], 'current': g} if current else {'groups': [g], 'open': [], 'current': None}


def test_before_giveup_time_keeps():
    assert gw.decide(st([]), 100, 900, 1800)[0] == 'keep'


def test_nothing_published_gives_up():
    assert gw.decide(st([]), 901, 900, 1800)[0] == 'giveup'


def test_accepted_never_gives_up():
    assert gw.decide(st([{'status': 'accepted', 'published_at': 10}]), 5000, 900, 1800)[0] == 'keep'


def test_in_flight_waits_then_gives_up_after_cap():
    m = [{'status': 'pending', 'published_at': 800}]
    assert gw.decide(st(m), 1000, 900, 1800)[0] == 'keep'
    assert gw.decide(st(m), 800 + 1800 + 121, 900, 1800)[0] == 'giveup'


def test_resolved_without_success_gives_up():
    m = [{'status': 'rejected', 'published_at': 100}, {'status': 'expired', 'published_at': 200}]
    assert gw.decide(st(m, current=False), 1000, 900, 1800)[0] == 'giveup'


def test_late_publication_rearms():
    m = [{'status': 'rejected', 'published_at': 950}]
    assert gw.decide(st(m), 1000, 900, 1800)[0] == 'keep'
    assert gw.decide(st(m), 950 + 901, 900, 1800)[0] == 'giveup'
