import ast
import hashlib
import os
from pathlib import Path

import pytest


def reader():
    source=Path(__file__).parents[1]/'infra_vast/vastctl.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='library_prebuild_reader')
    namespace={'Path':Path,'os':os}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['library_prebuild_reader']


def test_controller_pinned_reader_ignores_old_import_cache(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import vast_prebuild
    source=tmp_path/'reviewed.py';source.write_text("MARKER='reviewed attachment reader'\n")
    monkeypatch.setenv('ISG_VAST_LIBRARY_READER',str(source))
    monkeypatch.setenv('ISG_VAST_LIBRARY_READER_SHA256',hashlib.sha256(source.read_bytes()).hexdigest())
    assert reader()().MARKER=='reviewed attachment reader'
    assert not hasattr(vast_prebuild,'MARKER')


def test_reader_hash_checked_before_execution(tmp_path,monkeypatch):
    source=tmp_path/'changed.py';source.write_text("raise RuntimeError('must not execute')\n")
    monkeypatch.setenv('ISG_VAST_LIBRARY_READER',str(source))
    monkeypatch.setenv('ISG_VAST_LIBRARY_READER_SHA256','0'*64)
    with pytest.raises(ValueError,match='reader changed'):reader()()


def test_legacy_reader_remains_default(monkeypatch):
    from scripts.campaign_20261110 import vast_prebuild
    monkeypatch.delenv('ISG_VAST_LIBRARY_READER',raising=False)
    assert reader()() is vast_prebuild
