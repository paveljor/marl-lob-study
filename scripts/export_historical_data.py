"""Export recovered historical CSVs, pseudonymizing hosts without changing measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

FILES = ("grid.csv", "latency.csv", "p2.csv", "rescale.csv", "retrain.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path, destination: Path, source_revision: str) -> dict:
    targets = [destination / "desktop" / name for name in FILES]
    targets.append(destination / "manifest.json")
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing historical export")
    tables = {}
    for name in FILES:
        with (source / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if not reader.fieldnames or "host" not in reader.fieldnames or not rows:
                raise ValueError(f"Expected a nonempty table with a host column: {name}")
            tables[name] = (reader.fieldnames, rows)
    hosts = sorted({row["host"] for _, rows in tables.values() for row in rows})
    labels = {host: f"runner-{index + 1}" for index, host in enumerate(hosts)}
    manifest = {
        "schema_version": 1,
        "scope": "historical summaries; not scientifically revalidated or pooled with confirmation",
        "collection_source_revision": source_revision,
        "transformation": "Only host labels pseudonymized; CSV serialization may change",
        "files": [],
    }
    for name, (fields, rows) in tables.items():
        path = destination / "desktop" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**row, "host": labels[row["host"]]} for row in rows)
        with path.open(newline="") as handle:
            published = list(csv.DictReader(handle))
        for original, public in zip(rows, published, strict=True):
            if any(original[key] != public[key] for key in fields if key != "host"):
                raise ValueError(f"Measurement changed during historical export: {name}")
        manifest["files"].append({
            "path": str(path.relative_to(destination)),
            "rows": len(rows),
            "original_sha256": sha256(source / name),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        })
    with (destination / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the five recovered CSVs")
    parser.add_argument("--out", type=Path, default=Path("data/historical"))
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest = export(args.source, args.out, args.source_revision)
    print(f"Exported {len(manifest['files'])} historical tables; measurements verified unchanged")


if __name__ == "__main__":
    main()
