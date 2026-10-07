from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "restore_resum_flex.py"
SPEC = importlib.util.spec_from_file_location("restore_resum_flex", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_member_accepts_regular_relative_file() -> None:
    member = tarfile.TarInfo("core/training.py")
    member.size = 1
    MODULE.validate_member(member)


@pytest.mark.parametrize("name", ["../escape", "/absolute/path"])
def test_validate_member_rejects_escaping_path(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe archive path"):
        MODULE.validate_member(tarfile.TarInfo(name))


def test_validate_member_rejects_link() -> None:
    member = tarfile.TarInfo("link")
    member.type = tarfile.SYMTYPE
    member.linkname = "target"
    with pytest.raises(ValueError, match="Unsupported archive member type"):
        MODULE.validate_member(member)


def test_verify_sources_checks_content(tmp_path: Path) -> None:
    source = tmp_path / "core/training.py"
    source.parent.mkdir()
    source.write_bytes(b"test source\n")
    expected = {"core/training.py": hashlib.sha256(b"test source\n").hexdigest()}
    MODULE.verify_sources(tmp_path, expected)


def test_tar_fixture_contains_no_implicit_link(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("core/training.py")
        payload = b"x"
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    with tarfile.open(archive) as bundle:
        MODULE.validate_member(bundle.getmembers()[0])
