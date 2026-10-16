"""Source provenance tests; no simulator runs or policy training."""
import hashlib
import importlib
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("abides_core")
pytest.importorskip("abides_markets")

# Load the public script explicitly, not the fresh environment's editable snapshot.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "two_trader.py"
spec = importlib.util.spec_from_file_location("two_trader_sources_test", SCRIPT)
two_trader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_trader)

MODULES = {
    "abides_core.utils": "abides-core/abides_core/utils.py",
    "abides_markets.configs.rmsc03": "abides-markets/abides_markets/configs/rmsc03.py",
}


def project_root(tmp_path, monkeypatch):
    root = tmp_path / "public"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "marl_lob").mkdir(parents=True)
    (root / "scripts" / "two_trader.py").write_bytes(b"# project script\n")
    (root / "src" / "marl_lob" / "__init__.py").write_bytes(b"# project package\n")
    (root / "pyproject.toml").write_bytes(b"# project metadata\n")
    monkeypatch.setattr(two_trader, "__file__", str(root / "scripts" / "two_trader.py"))
    monkeypatch.setattr(two_trader.subprocess, "check_output", lambda *a, **kw: "fixture==1\n")
    return root


@pytest.mark.parametrize("layout", ["sibling", "work"])
def test_snapshot_uses_imported_sources(tmp_path, monkeypatch, layout):
    root = project_root(tmp_path, monkeypatch)
    sibling = root.parent / "abides-jpmc-public"
    active = sibling if layout == "sibling" else root / ".work" / "abides-jpmc-public"
    expected = {}
    for name, rel in MODULES.items():
        stale = sibling / rel
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"# stale sibling, not runtime code\n")
        path = active / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"# runtime source for {name}\nVALUE = 42\n".encode()
        path.write_bytes(data)
        module_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
        expected["abides-local/" + rel] = data

    out = tmp_path / "output"
    out.mkdir()
    two_trader.save_sources(out)
    manifest = json.loads((out / "source_manifest.json").read_text())
    with zipfile.ZipFile(out / "source_snapshot.zip") as archive:
        for member, data in expected.items():
            assert archive.read(member) == data
            assert manifest[member] == hashlib.sha256(data).hexdigest()
        for rel in ("scripts/two_trader.py", "src/marl_lob/__init__.py", "pyproject.toml"):
            assert archive.read(rel) == (root / rel).read_bytes()
            assert manifest[rel] == hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert set(archive.namelist()) == set(manifest)
    assert (out / "pip-freeze.txt").read_text() == "fixture==1\n"


@pytest.mark.parametrize("failure", ["missing", "no_source", "unreadable"])
@pytest.mark.parametrize("name", MODULES)
def test_missing_readable_source_has_useful_error(tmp_path, monkeypatch, failure, name):
    project_root(tmp_path, monkeypatch)
    module = types.ModuleType(name)
    path = tmp_path / "missing_utils.py"
    if failure != "no_source":
        module.__file__ = str(path)
    if failure == "unreadable":
        path.write_bytes(b"# inaccessible source\n")
        read_bytes = Path.read_bytes

        def denied(self):
            if self == path:
                raise PermissionError("source permission denied")
            return read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", denied)
    monkeypatch.setitem(sys.modules, name, module)
    out = tmp_path / "output"
    out.mkdir()
    with pytest.raises(RuntimeError, match=f"Cannot snapshot runtime module {name}") as err:
        two_trader.save_sources(out)
    assert "readable Python source files" in str(err.value)
    if failure != "no_source":
        assert str(path) in str(err.value)
    assert not (out / "source_snapshot.zip").exists()
    assert not (out / "source_manifest.json").exists()


def test_snapshot_matches_real_runtime_source_bytes(tmp_path, monkeypatch):
    project_root(tmp_path, monkeypatch)
    expected = {}
    for name, rel in MODULES.items():
        module = importlib.import_module(name)
        expected["abides-local/" + rel] = Path(module.__file__).read_bytes()
    out = tmp_path / "output"
    out.mkdir()
    two_trader.save_sources(out)
    manifest = json.loads((out / "source_manifest.json").read_text())
    with zipfile.ZipFile(out / "source_snapshot.zip") as archive:
        for member, data in expected.items():
            assert archive.read(member) == data
            assert manifest[member] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("layout", ["sibling", "work", "installed", "split"])
def test_git_sha_uses_active_checkout_only(tmp_path, monkeypatch, layout):
    root = tmp_path / "public"
    sibling = tmp_path / "abides-jpmc-public"
    (sibling / ".git").mkdir(parents=True)
    active = sibling if layout == "sibling" else root / ".work" / "abides-jpmc-public"
    if layout in ("work", "split"):
        (active / ".git").mkdir(parents=True)
    sources = {}
    for name, rel in MODULES.items():
        checkout = sibling if layout == "split" and name == "abides_core.utils" else active
        sources[rel] = (checkout / rel, b"# runtime source\n")
    monkeypatch.setattr(two_trader, "simulator_sources", lambda: sources)
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        return "runtime-checkout-sha\n"

    monkeypatch.setattr(two_trader.subprocess, "check_output", git)
    if layout in ("installed", "split"):
        assert two_trader.simulator_git_sha() is None
        assert calls == []
    else:
        assert two_trader.simulator_git_sha() == "runtime-checkout-sha"
        assert calls == [
            ["git", "-C", str(active), "ls-files", "--error-unmatch", "--",
             *MODULES.values()],
            ["git", "-C", str(active), "rev-parse", "HEAD"],
        ]


def test_git_sha_does_not_claim_unrelated_parent_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    sources = {rel: (tmp_path / ".work" / rel, b"# installed source\n")
               for rel in MODULES.values()}
    monkeypatch.setattr(two_trader, "simulator_sources", lambda: sources)

    def untracked(command, **kwargs):
        assert command[3:6] == ["ls-files", "--error-unmatch", "--"]
        raise two_trader.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(two_trader.subprocess, "check_output", untracked)
    assert two_trader.simulator_git_sha() is None
