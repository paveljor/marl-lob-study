"""Dependency-light checks for the generated, read-only research presentation."""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_research_notebook", ROOT / "scripts/build_research_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_notebook_is_deterministic_and_unexecuted():
    builder = load_builder()
    rendered = builder.render_notebook()
    assert rendered == builder.render_notebook()
    committed = json.loads(
        (ROOT / "notebooks/research_walkthrough.ipynb").read_text(encoding="utf-8")
    )
    assert builder.source_signature(committed) == builder.source_signature(json.loads(rendered))
    notebook = json.loads(rendered)
    assert notebook["nbformat"] == 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        assert cell["metadata"]["slideshow"]["slide_type"] in {"slide", "subslide"}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse(cell["source"])


def test_source_check_allows_reviewed_execution_and_nbformat_line_lists():
    builder = load_builder()
    notebook = builder.build_notebook()
    original = builder.source_signature(notebook)
    for cell in notebook["cells"]:
        cell["source"] = cell["source"].splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["execution_count"] = 1
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "ok"}]
            cell["metadata"]["execution"] = {"iopub.status.busy": "reviewed"}
    assert builder.source_signature(notebook) == original
    notebook["cells"][0]["source"] = "Changed research claim"
    assert builder.source_signature(notebook) != original


def test_research_notebook_uses_evidence_without_simulation_or_pickle():
    notebook = load_builder().build_notebook()
    code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module.split(".")[0])
    assert set(imports) <= {"pathlib", "numpy", "pandas", "matplotlib", "IPython"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    load_calls = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
    assert load_calls
    for call in load_calls:
        assert any(k.arg == "allow_pickle" and isinstance(k.value, ast.Constant)
                   and k.value.value is False for k in call.keywords)
    assert "len(confirmation) == 288" in code
    assert "len(raw_audit) == 288" in code
    assert "BOOTSTRAP_SEED = 12345" in code
    assert "BOOTSTRAP_DRAWS = 10000" in code
