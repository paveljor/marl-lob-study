"""Export the finite corrected-study allowlist without executing archived code."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

SEEDS = (17, 29, 43)
SCENARIOS = ("sampled_pair", "marginal_b")
MARKETS = tuple(range(40001, 40013))
CALIBRATION = tuple(range(20001, 20005))
ROLES = ("mm_0", "mm_1")
RULES = {
    "canonical-parent-v1": "Replace evaluation parent_run with runs/concurrent/finite_seed{seed}.",
    "local-package-location-v1": (
        "Replace local file/editable package locators with comments retaining recorded package "
        "identity/version when present; preserve all other lines. No inferred package versions."
    ),
}


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def guarded_path(root, relative):
    """Reject traversal and symlinks, including symlinks in existing ancestors."""
    rel = PurePosixPath(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise ValueError("Unsafe artifact path")
    root = Path(root).absolute()
    target = root.joinpath(*rel.parts)
    for part in (target, *target.parents):
        if part.is_symlink():
            raise ValueError("Symlink artifact path")
    if not target.is_relative_to(root):
        raise ValueError("Artifact escaped root")
    return target


def sanitize_packages(payload):
    lines = []
    for line in payload.decode().splitlines(keepends=True):
        if re.search(r"file://|(?:^|\s)(?:/home/|/Users/|/root/)|^-e\s+\.", line):
            identity = re.match(r"([A-Za-z0-9_.-]+(?:==[^\s]+)?)\s*@\s*file:", line)
            egg = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", line)
            if not identity and not egg:
                raise ValueError("Local package locator lacks a publishable package identity")
            name = identity.group(1) if identity else egg.group(1)
            lines.append(f"# {name}: local source location redacted; version not inferred\n")
        else:
            lines.append(line)
    return "".join(lines).encode()


def source_allowlist():
    """Yield (source path relative to original root, public path, transformation)."""
    for seed in SEEDS:
        base = f"runs/concurrent/finite_seed{seed}_confirmation"
        dest = f"confirmation/seed{seed}"
        for scenario in SCENARIOS:
            for market in MARKETS:
                for rep in (0, 1):
                    folder = f"eval/{scenario}/seed{market}_rep{rep}"
                    for name in ("mm_0.npz", "mm_1.npz", "market.json"):
                        yield f"{base}/{folder}/{name}", f"{dest}/{folder}/{name}", None
        for market in CALIBRATION:
            for name in ("mm_0.npz", "mm_1.npz", "market.json"):
                tail = f"calibration/seed{market}/{name}"
                yield f"{base}/{tail}", f"{dest}/{tail}", None
        for name in ("b_action_pool.npy", "summary.csv", "run_meta.json", "pip-freeze.txt",
                     "source_manifest.json", "source_snapshot.zip"):
            rule = ("canonical-parent-v1" if name == "run_meta.json" else
                    "local-package-location-v1" if name == "pip-freeze.txt" else None)
            yield f"{base}/{name}", f"{dest}/{name}", rule
        base = f"runs/concurrent/finite_seed{seed}"
        dest = f"training/finite_seed{seed}"
        for name in ("run_meta.json", "normalizers.json", "training_diagnostics.json",
                     "protocol.md", "source_manifest.json", "source_snapshot.zip", "pip-freeze.txt",
                     "a/initial.zip", "a/policy.zip", "b/initial.zip", "b/policy.zip"):
            rule = "local-package-location-v1" if name == "pip-freeze.txt" else None
            yield f"{base}/{name}", f"{dest}/{name}", rule


def member_manifest(payload):
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source ZIP member")
        for name in names:
            guarded_path(Path("/unused-study-member-root"), name)
        return {name: {"sha256": digest(archive.read(name)),
                       "bytes": archive.getinfo(name).file_size} for name in sorted(names)}


def build_export(source, repository):
    files, records = {}, []
    for original, public, rule in source_allowlist():
        raw = guarded_path(source, original).read_bytes()
        payload = raw
        if rule == "canonical-parent-v1":
            meta = json.loads(raw)
            seed = int(PurePosixPath(public).parts[1].removeprefix("seed"))
            meta["parent_run"] = f"runs/concurrent/finite_seed{seed}"
            payload = encoded(meta)
        elif rule == "local-package-location-v1":
            payload = sanitize_packages(raw)
        files[public] = payload
        records.append({"path": public, "source": original,
                        "original_sha256": digest(raw), "original_bytes": len(raw),
                        "sha256": digest(payload), "bytes": len(payload),
                        "disposition": "sanitized" if rule else "unchanged",
                        "transform_rules": [rule] if rule else []})
        if public.endswith("source_snapshot.zip"):
            supplemental = str(PurePosixPath(public).with_name("source_members.json"))
            files[supplemental] = encoded(member_manifest(raw))
    for name in ("LICENSE", "LICENSE.txt"):
        relative = f"third_party/abides-jpmc-public/{name}"
        files[relative] = guarded_path(repository, relative).read_bytes()
    notice = (
        "# Study artifact third-party notices\n\n"
        "Project-authored source in the historical snapshots is covered by the repository's "
        "root MIT license. This does not replace upstream terms.\n\n"
        "The immutable source_snapshot.zip archives include ABIDES-derived source, including "
        "abides-local/abides-core/abides_core/utils.py and "
        "abides-local/abides-markets/abides_markets/configs/rmsc03.py. "
        "The recorded upstream revision is "
        "f9cbe51342b7dedd9587e4e069040d68a5c6477f.\n\n"
        "Full upstream copyright, attribution, funding, bundled-library notices, conditions "
        "and disclaimers are retained verbatim alongside this distribution:\n\n"
        "- [J.P. Morgan Chase notice](third_party/abides-jpmc-public/LICENSE)\n"
        "- [Georgia Tech and bundled-library notices]"
        "(third_party/abides-jpmc-public/LICENSE.txt)\n\n"
        "Keep this notice and both full notice files with separately redistributed snapshots. "
        "Original ZIP bytes and historical manifests were not modified; source_members.json "
        "supplements their incomplete upstream-member coverage. Dependencies retain their "
        "own licenses. Checkpoint ZIPs are opaque pickle-based historical artifacts; loading "
        "untrusted models can execute code. Raw analysis never requires model loading.\n"
    )
    files["THIRD_PARTY_NOTICES.md"] = notice.encode()
    recorded = {record["path"] for record in records}
    for path, payload in files.items():
        if path not in recorded:
            records.append({"path": path, "source": None, "original_sha256": None,
                            "original_bytes": None, "sha256": digest(payload),
                            "bytes": len(payload), "disposition": "generated",
                            "transform_rules": []})
    manifest = {"schema_version": 1, "study": "corrected September 26 2026 confirmation",
                "transform_rules": RULES, "files": sorted(records, key=lambda row: row["path"])}
    files["manifest.json"] = encoded(manifest)
    return files


def export(source, repository):
    source, repository = Path(source).absolute(), Path(repository).absolute()
    destination = repository / "data"
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Export destination must not be inside the immutable source")
    files = build_export(source, repository)
    # Preflight every owned file before writing anything; parent-owned data is untouched.
    for relative in files:
        target = guarded_path(destination, relative)
        if target.exists():
            raise FileExistsError(f"Refusing existing export artifact: {relative}")
    for relative, payload in sorted(files.items()):
        target = guarded_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(payload)
    return len(files), sum(map(len, files.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, size = export(args.source, args.repository)
    print(f"Exported {count} files, {size:,} bytes; run scripts/verify_study_data.py next.")


if __name__ == "__main__":
    main()
