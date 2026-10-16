"""Verify published study bytes and numeric accounting; never load models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from export_study_data import (
    CALIBRATION,
    MARKETS,
    ROLES,
    RULES,
    SCENARIOS,
    SEEDS,
    build_export,
    digest,
    guarded_path,
    member_manifest,
    source_allowlist,
)

DTYPES = {"timestamps": "float64", "inventory": "int64", "cash": "int64",
          "mid_price": "int64", "fill_timestamps": "float64", "fill_side": "int64",
          "fill_price": "int64", "fill_quantity": "int64", "actions": "float32",
          "observations": "float32"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(DTYPES), "Unexpected NPZ keys")
        arrays = {key: data[key] for key in data.files}
    for key, value in arrays.items():
        require(str(value.dtype) == DTYPES[key], f"Unexpected dtype: {key}")
        require(np.isfinite(value).all(), f"Nonfinite values: {key}")
    return arrays


def verify_account(path, market, role):
    x = numeric_arrays(path)
    for key in ("timestamps", "inventory", "cash", "mid_price"):
        require(x[key].shape == (600,), f"Invalid account shape: {key}")
    require(x["actions"].shape == (599, 6), "Invalid action shape")
    require(x["observations"].shape == (600, 48), "Invalid observation shape")
    require(np.all(np.diff(x["timestamps"]) > 0), "Unordered account timestamps")
    count = len(x["fill_side"])
    for key in ("fill_timestamps", "fill_side", "fill_price", "fill_quantity"):
        require(x[key].shape == (count,), "Unequal fill lengths")
    require(np.isin(x["fill_side"], [-1, 1]).all(), "Invalid fill side")
    require((x["fill_quantity"] > 0).all() and (x["fill_price"] > 0).all(), "Invalid fill")
    q, c, m = (x[key] for key in ("inventory", "cash", "mid_price"))
    signed = x["fill_side"] * x["fill_quantity"]
    require(int(signed.sum()) == int(q[-1] - q[0]), "Inventory does not reconcile")
    require(int(-(signed * x["fill_price"]).sum()) == int(c[-1] - c[0]),
            "Cash does not reconcile")
    pnl = int(c[-1] + q[-1] * m[-1] - c[0] - q[0] * m[0])
    components = int((q[:-1] * np.diff(m)).sum()
                     + (np.diff(c) + np.diff(q) * m[1:]).sum())
    require(components == pnl, "P&L decomposition does not reconcile")
    executions = market["executions"][role]
    require(np.array_equal(signed, [e["signed_quantity"] for e in executions]),
            "JSON executions disagree with NPZ quantities")
    require(np.array_equal(x["fill_price"], [e["price_cents"] for e in executions]),
            "JSON executions disagree with NPZ prices")
    return x, pnl


def verify(data, source=None, repository=None):
    data = Path(data)
    manifest = json.loads(guarded_path(data, "manifest.json").read_text())
    require(manifest["schema_version"] == 1 and manifest["transform_rules"] == RULES,
            "Unknown manifest schema or transformations")
    entries = manifest["files"]
    paths = [row["path"] for row in entries]
    require(len(paths) == len(set(paths)), "Duplicate manifest paths")
    expected_sources = {public: (original, rule) for original, public, rule in source_allowlist()}
    supplements = {str(Path(public).with_name("source_members.json"))
                   for public in expected_sources if public.endswith("source_snapshot.zip")}
    generated = supplements | {"THIRD_PARTY_NOTICES.md", "third_party/abides-jpmc-public/LICENSE",
                               "third_party/abides-jpmc-public/LICENSE.txt"}
    require(set(paths) == set(expected_sources) | generated, "Manifest allowlist mismatch")
    for row in entries:
        payload = guarded_path(data, row["path"]).read_bytes()
        require(len(payload) == row["bytes"] and digest(payload) == row["sha256"],
                f"Artifact hash/size mismatch: {row['path']}")
        if row["path"] in expected_sources:
            original, rule = expected_sources[row["path"]]
            require(row["source"] == original
                    and row["transform_rules"] == ([rule] if rule else []),
                    "Invalid provenance mapping")
            require(row["disposition"] == ("sanitized" if rule else "unchanged"),
                    "Invalid artifact disposition")
            if not rule:
                require(row["original_sha256"] == row["sha256"]
                        and row["original_bytes"] == row["bytes"], "Raw artifact was transformed")
            if source is not None:
                raw = guarded_path(source, original).read_bytes()
                require(digest(raw) == row["original_sha256"]
                        and len(raw) == row["original_bytes"], "Original provenance mismatch")
        if row["path"].endswith("source_snapshot.zip"):
            members = member_manifest(payload)
            directory = guarded_path(data, row["path"]).parent
            require(members == json.loads((directory / "source_members.json").read_text()),
                    "Supplemental source-member hashes disagree")
            historical = json.loads((directory / "source_manifest.json").read_text())
            require(all(name in members and members[name]["sha256"] == sha
                        for name, sha in historical.items()), "Historical source manifest mismatch")
    for owned in ("confirmation", "training", "third_party"):
        actual = {str(p.relative_to(data)) for p in guarded_path(data, owned).rglob("*")
                  if p.is_file() or p.is_symlink()}
        expected = {p for p in paths if p.startswith(owned + "/")}
        require(actual == expected, "Unexpected files in owned export directories")
    if source is not None:
        require(repository is not None, "Repository required for source-backed verification")
        rebuilt = build_export(Path(source), Path(repository))
        for path, payload in rebuilt.items():
            require(guarded_path(data, path).read_bytes() == payload,
                    f"Deterministic source export mismatch: {path}")
    episodes = npz_count = checkpoint_count = 0
    for seed in SEEDS:
        directory = data / "confirmation" / f"seed{seed}"
        meta = json.loads((directory / "run_meta.json").read_text())
        require(meta["status"] == "complete", "Incomplete confirmation")
        require(meta["parent_run"] == f"runs/concurrent/finite_seed{seed}", "Noncanonical parent")
        require(meta["completed_markets"] == 48, "Incomplete market count")
        require(meta["args"]["market_seeds"] == list(MARKETS)
                and meta["args"]["calibration_seeds"] == list(CALIBRATION)
                and meta["args"]["replicates"] == [0, 1]
                and meta["args"]["scenarios"] == list(SCENARIOS), "Unexpected evaluation design")
        with (directory / "summary.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {(scenario, market, rep, role) for scenario in SCENARIOS
                    for market in MARKETS for rep in (0, 1) for role in ROLES}
        keys = [(r["scenario"], int(r["seed"]), int(r["replicate"]), r["agent"]) for r in rows]
        require(len(keys) == 96 and set(keys) == expected, "Summary Cartesian product mismatch")
        for row, (scenario, market_seed, rep, role) in zip(rows, keys, strict=True):
            folder = directory / "eval" / scenario / f"seed{market_seed}_rep{rep}"
            market = json.loads((folder / "market.json").read_text())
            require(market["seed"] == market_seed
                    and market["environment_config"] == meta["environment_config"],
                    "Market metadata mismatch")
            x, pnl = verify_account(folder / f"{role}.npz", market, role)
            ids = set(map(int, market["agent_ids"]))
            direct = [e for e in market["exchange_matches"] if e["agent_id"] in ids
                      and e["oppos_agent_id"] in ids and e["agent_id"] != e["oppos_agent_id"]]
            values = {"steps": len(x["actions"]), "final_inventory": int(x["inventory"][-1]),
                      "max_abs_inventory": int(np.abs(x["inventory"]).max()),
                      "gross_fills": len(x["fill_side"]),
                      "gross_shares": int(x["fill_quantity"].sum()),
                      "submitted_orders": len(market["submissions"][role]),
                      "direct_pair_matches": len(direct),
                      "direct_pair_shares": sum(e["quantity"] for e in direct),
                      "gross_fill_cash_residual_cents": 0, "pnl_dollars": pnl / 100}
            for name, value in values.items():
                require(np.isclose(float(row[name]), value, rtol=0, atol=1e-8),
                        f"Summary does not reconcile: {name}")
            require(row["risk_terminated"] in ("True", "False"), "Invalid summary boolean")
            require((row["risk_terminated"] == "True") == market["risk_terminated"],
                    "Risk status mismatch")
            episodes += 1
            npz_count += 1
        actions = []
        for market_seed in CALIBRATION:
            folder = directory / "calibration" / f"seed{market_seed}"
            market = json.loads((folder / "market.json").read_text())
            require(market["seed"] == market_seed
                    and market["environment_config"] == meta["environment_config"],
                    "Calibration metadata mismatch")
            for role in ROLES:
                x, _ = verify_account(folder / f"{role}.npz", market, role)
                npz_count += 1
                if role == "mm_1":
                    actions.append(x["actions"])
        pool = np.load(directory / "b_action_pool.npy", allow_pickle=False)
        require(pool.dtype == np.float32 and pool.shape == (2396, 6)
                and np.isfinite(pool).all(), "Invalid action pool")
        require(np.array_equal(pool, np.concatenate(actions)), "Calibration action pool mismatch")
        training = data / "training" / f"finite_seed{seed}"
        train_meta = json.loads((training / "run_meta.json").read_text())
        require(train_meta["status"] == "complete" and train_meta["actual_timesteps"] == 32768
                and train_meta["market_close_bootstrap"] is False
                and train_meta["args"]["legacy_close_bootstrap"] is False
                and train_meta["environment_config"] == meta["environment_config"],
                "Unexpected training metadata")
        require(not (set(train_meta["episode_seeds"]) & (set(MARKETS) | set(CALIBRATION))),
                "Training and held-out market seeds overlap")
        for key, name in (("initial_a", "a/initial.zip"), ("a", "a/policy.zip"),
                          ("initial_b", "b/initial.zip"), ("b", "b/policy.zip")):
            require(digest((training / name).read_bytes()) == meta["checkpoint_sha256"][key],
                    "Checkpoint ZIP hash disagrees with evaluation metadata")
            checkpoint_count += 1
    return {"manifest_files": len(entries), "npz_files": npz_count, "npy_files": 3,
            "confirmation_rows": episodes, "checkpoint_zips": checkpoint_count,
            "artifact_bytes": sum(row["bytes"] for row in entries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.data, args.source, args.repository), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
