import pytest
from scripts.campaign_20261110.prepare_consumed_map_routes import registry_route


def test_route_updates_existing_verifier_without_duplicate_or_output_change():
    source = {'path': '/cached/tickets.json', 'sha256': 'one'}
    rows = [{'manifest': source, 'output': '/existing/native-reader'},
            {'manifest': {'path': '/other/tickets.json', 'sha256': 'two'}, 'output': '/other/reader'}]
    registry_route(rows, source, {'path': '/new/route.json', 'sha256': 'route'}, '/unused')
    assert len(rows) == 2
    assert rows[0]['output'] == '/existing/native-reader'
    assert rows[0]['route']['sha256'] == 'route'
    assert rows[1]['output'] == '/other/reader'


def test_route_rejects_changed_source_and_adds_distinct_same_basename():
    rows = [{'manifest': {'path': '/one/tickets.json', 'sha256': 'one'}, 'output': '/one/reader'}]
    with pytest.raises(ValueError, match='source changed'):
        registry_route(rows, {'path': '/one/tickets.json', 'sha256': 'changed'}, {}, '/unused')
    registry_route(rows, {'path': '/two/tickets.json', 'sha256': 'two'}, {}, '/two/reader')
    assert len(rows) == 2
