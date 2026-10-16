"""Historical exports change machine labels, not research measurements."""

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_historical_data.py"
SPEC = importlib.util.spec_from_file_location("export_historical_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_source(path):
    path.mkdir()
    for name in MODULE.FILES:
        (path / name).write_text("host,value,seed\nprivate-host,0.12345678901234567,17\n")


def test_historical_export_preserves_measurements_and_hides_hosts(tmp_path):
    source, target = tmp_path / "source", tmp_path / "export"
    make_source(source)
    manifest = MODULE.export(source, target, "test-revision")
    assert len(manifest["files"]) == 5
    for record in manifest["files"]:
        path = target / record["path"]
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        assert row == {"host": "runner-1", "value": "0.12345678901234567", "seed": "17"}
        assert record["sha256"] == MODULE.sha256(path)
        assert record["original_sha256"] == MODULE.sha256(source / path.name)
    assert "private-host" not in (target / "manifest.json").read_text()


def test_historical_export_refuses_overwrite(tmp_path):
    source, target = tmp_path / "source", tmp_path / "export"
    make_source(source)
    MODULE.export(source, target, "test-revision")
    with pytest.raises(FileExistsError, match="Refusing"):
        MODULE.export(source, target, "test-revision")


def test_historical_export_checks_inputs_before_writing(tmp_path):
    source, target = tmp_path / "source", tmp_path / "export"
    make_source(source)
    (source / "rescale.csv").write_text("value\n1\n")
    with pytest.raises(ValueError, match="host column"):
        MODULE.export(source, target, "test-revision")
    assert not target.exists()
