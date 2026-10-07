from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "verify_artifact.py"
SPEC = importlib.util.spec_from_file_location("verify_artifact", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repository_path_resolution_rejects_escape() -> None:
    with pytest.raises(ValueError, match="Path escapes repository"):
        MODULE.resolve_inside_repo("../outside")


def test_portable_release_verifier_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout
