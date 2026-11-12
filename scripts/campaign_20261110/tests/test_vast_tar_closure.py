"""Archive sibling races must not hide mutations or read errors in payloads."""
import ast
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile

import pytest


def packer(tmp_path, monkeypatch, action):
    source = Path(__file__).parents[1] / 'infra_vast/vastctl.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'tar_put')
    namespace = dict(Path=Path, os=os, tempfile=tempfile, subprocess=subprocess, shlex=shlex,
                     STAGE_COMPRESSION_THREADS=1, JOB_S3='s3://test')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    real_tar = shutil.which('tar')
    scripts = {
        'tar': f'#!/bin/bash\n{action}\n',
        'zstd': '#!/bin/bash\ncat\n',
        's5cmd': '#!/bin/bash\ncat > "$ARCHIVE"\n',
    }
    for name, body in scripts.items():
        p = bindir / name
        p.write_text(body.replace('REAL_TAR', shlex.quote(real_tar)))
        p.chmod(0o755)
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('S3_ENDPOINT_URL', 'unused')
    monkeypatch.setenv('ARCHIVE', str(tmp_path / 'archive.tar'))
    parent = tmp_path / 'shared'
    parent.mkdir()
    payload = parent / 'input.json'
    payload.write_text('fixed input\n')
    monkeypatch.setenv('PARENT', str(parent))
    monkeypatch.setenv('PAYLOAD', str(payload))
    listing = tmp_path / 'files'
    listing.write_text(str(parent).lstrip('/') + '\n' + str(payload).lstrip('/') + '\n')
    return namespace['tar_put'], listing, payload


def test_sibling_directory_change_preserves_exact_payload_and_directory(tmp_path, monkeypatch):
    # Reproduce GNU tar's exact directory race diagnostic deterministically.
    action = 'REAL_TAR "$@"\nmkdir "$PARENT/sibling"\nprintf "tar: %s: file changed as we read it\\n" "${PARENT#/}" >&2\nexit 1'
    pack, listing, payload = packer(tmp_path, monkeypatch, action)
    pack(listing, 'job')
    with tarfile.open(tmp_path / 'archive.tar') as archive:
        assert archive.extractfile(str(payload).lstrip('/')).read() == b'fixed input\n'
        assert archive.getmember(str(payload.parent).lstrip('/')).isdir()
        assert str(payload.parent / 'sibling').lstrip('/') not in archive.getnames()


def test_changed_payload_fails_even_with_only_directory_diagnostic(tmp_path, monkeypatch):
    action = 'REAL_TAR "$@"\nprintf "changed payload\\n" > "$PAYLOAD"\nprintf "tar: %s: file changed as we read it\\n" "${PARENT#/}" >&2\nexit 1'
    pack, listing, _ = packer(tmp_path, monkeypatch, action)
    with pytest.raises(RuntimeError, match='archive input changed'):
        pack(listing, 'job')


def test_missing_payload_fails(tmp_path, monkeypatch):
    pack, listing, _ = packer(tmp_path, monkeypatch, 'rm "$PAYLOAD"\nREAL_TAR "$@"')
    with pytest.raises(FileNotFoundError):
        pack(listing, 'job')


@pytest.mark.parametrize('message', ['file changed as we read it', 'Cannot open: Permission denied'])
def test_payload_diagnostics_are_never_ignored(tmp_path, monkeypatch, message):
    action = f'REAL_TAR "$@"\nprintf "tar: %s: {message}\\n" "${{PAYLOAD#/}}" >&2\nexit 1'
    pack, listing, _ = packer(tmp_path, monkeypatch, action)
    with pytest.raises(RuntimeError, match='archive pipeline failed'):
        pack(listing, 'job')


def test_compressor_failure_is_not_hidden_by_directory_race(tmp_path, monkeypatch):
    pack, listing, _ = packer(tmp_path, monkeypatch, 'REAL_TAR "$@"')
    (tmp_path / 'bin/zstd').write_text('#!/bin/bash\ncat\nexit 7\n')
    with pytest.raises(RuntimeError, match='archive pipeline failed'):
        pack(listing, 'job')
