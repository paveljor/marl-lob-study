"""Offline analysis entry points must offer help without reading run artifacts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "agent_divergence.py",
        "fill_skew.py",
        "offset_distribution.py",
        "p2_analysis.py",
        "quote_stats.py",
    ],
)
def test_analysis_help_from_outside_repo(script, tmp_path):
    # Make the source package available without requiring an editable install.
    # Running from an empty directory catches accidental dependence on run files
    # or the caller's working directory, while --help never runs an analysis.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not list(tmp_path.iterdir()), "Help must not create research artifacts"
