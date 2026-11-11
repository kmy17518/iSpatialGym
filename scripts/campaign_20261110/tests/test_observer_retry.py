import sys
import pytest
from scripts.campaign_20261110.observer_retry import execute


def test_run_module_preserves_self_referential_controller_adapter(tmp_path,monkeypatch):
    (tmp_path/'test_adapter_entry.py').write_text('import sys\ndef load_seeds(): return "exact-adapter"\nassert sys.modules[__name__].load_seeds()=="exact-adapter"\nassert sys.argv[1:]==["--config","exact.json"]\n')
    monkeypatch.syspath_prepend(str(tmp_path));prior=sys.modules['__main__']
    execute(module='test_adapter_entry',argv=['--config','exact.json'])
    assert sys.modules['__main__'] is prior


def test_real_failures_are_not_relabelled_or_retried_by_entry_runner(tmp_path,monkeypatch):
    (tmp_path/'test_failing_entry.py').write_text('raise ValueError("invalid physical evidence")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ValueError,match='physical evidence'):execute(module='test_failing_entry')
    with pytest.raises(ValueError,match='exactly one'):execute()
