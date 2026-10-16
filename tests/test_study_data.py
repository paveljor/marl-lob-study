"""Data publication safety and complete saved-study verification."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_script("export_study_data")
verifier = load_script("verify_study_data")


def test_package_redaction_preserves_versions():
    raw = b"numpy==2.4.4\npackaging @ file:///home/private/build\nother==1.2\n"
    public = exporter.sanitize_packages(raw)
    assert b"/home/" not in public and b"file://" not in public
    assert b"# packaging:" in public
    assert public.startswith(b"numpy==2.4.4\n") and public.endswith(b"other==1.2\n")


def test_unknown_editable_identity_rejected():
    with pytest.raises(ValueError, match="identity"):
        exporter.sanitize_packages(b"-e /home/private/project\n")


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_path_guard(tmp_path, relative):
    with pytest.raises(ValueError, match="path"):
        exporter.guarded_path(tmp_path, relative)


def test_symlink_guard(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        exporter.guarded_path(tmp_path, "link/file")


@pytest.mark.parametrize("dtype", [object, "U4"])
def test_nonnumeric_npz_rejected(tmp_path, dtype):
    arrays = {name: np.zeros(1, dtype=kind) for name, kind in verifier.DTYPES.items()}
    arrays["actions"] = np.array(["bad"], dtype=dtype)
    path = tmp_path / "bad.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        verifier.numeric_arrays(path)


def test_export_collision_preflight_preserves_parent_owned_files(tmp_path, monkeypatch):
    repo = tmp_path / "public"
    data = repo / "data"
    (data / "historical").mkdir(parents=True)
    (data / "historical/README.md").write_text("parent owned")
    (data / "manifest.json").write_text("existing")
    monkeypatch.setattr(exporter, "build_export", lambda *_: {
        "confirmation/seed17/new": b"new", "manifest.json": b"manifest"})
    with pytest.raises(FileExistsError):
        exporter.export(tmp_path / "original", repo)
    assert not (data / "confirmation").exists()
    assert (data / "historical/README.md").read_text() == "parent owned"


def test_manifest_tamper_rejected(tmp_path):
    entries = []
    for original, public, rule in exporter.source_allowlist():
        entries.append({"path": public, "source": original, "sha256": "wrong", "bytes": 0,
                        "transform_rules": [rule] if rule else []})
    generated = {str(Path(r["path"]).with_name("source_members.json")) for r in entries
                 if r["path"].endswith("source_snapshot.zip")}
    generated |= {"THIRD_PARTY_NOTICES.md", "third_party/abides-jpmc-public/LICENSE",
                  "third_party/abides-jpmc-public/LICENSE.txt"}
    entries.extend({"path": path, "sha256": "wrong", "bytes": 0} for path in generated)
    first = tmp_path / entries[0]["path"]
    first.parent.mkdir(parents=True)
    first.write_bytes(b"tampered")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "transform_rules": exporter.RULES, "files": entries}))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        verifier.verify(tmp_path)


def test_complete_published_raw_data():
    data = SCRIPTS.parent / "data"
    if not (data / "manifest.json").exists():
        pytest.skip("Study data export is not present")
    result = verifier.verify(data)
    assert result["npz_files"] == 312
    assert result["npy_files"] == 3
    assert result["confirmation_rows"] == 288
    assert result["checkpoint_zips"] == 12


def test_verify_command():
    data = SCRIPTS.parent / "data"
    if not (data / "manifest.json").exists():
        pytest.skip("Study data export is not present")
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "verify_study_data.py"), "--data", str(data)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["npz_files"] == 312


def test_required_notebook_path_exists():
    data = SCRIPTS.parent / "data"
    if not (data / "manifest.json").exists():
        pytest.skip("Study data export is not present")
    path = data / "confirmation/seed29/eval/sampled_pair/seed40001_rep0/mm_1.npz"
    assert verifier.numeric_arrays(path)["actions"].shape == (599, 6)
