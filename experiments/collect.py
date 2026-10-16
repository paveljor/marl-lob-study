#!/usr/bin/env python3
"""Aggregate a finished suite into one CSV plus a printed table.

Reads each run's run_meta.json (args + provenance + wallclock) and, where eval
trajectories exist, computes the realised change in mark-to-market equity per
agent plus the spread-capture / adverse-selection split from the fills.

Also archives each run's raw artifacts into the results dir, because
teardown.sh deletes runs/ and results/ is the only thing that survives being
copied off a borrowed machine.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def equity_change(npz):
    d = np.load(npz)
    eq = (
        d["cash"].astype(np.int64)
        + d["inventory"].astype(np.int64) * d["mid_price"].astype(np.int64)
    )
    return (eq[-1] - eq[0]) / 100.0


def markout(npz):
    """(capture, adverse_to_close) in dollars, or (None, None) without fills."""
    d = np.load(npz)
    if "fill_timestamps" not in d:
        return None, None
    ts, mid = d["timestamps"], d["mid_price"].astype(np.int64)
    ft, fs = d["fill_timestamps"], d["fill_side"].astype(np.int64)
    fp, fq = d["fill_price"].astype(np.int64), d["fill_quantity"].astype(np.int64)
    qi = np.clip(np.searchsorted(ts, ft, side="left") - 1, 0, len(mid) - 1)
    mid_q = mid[qi]
    capture = (fs * fq * (mid_q - fp)).sum() / 100.0
    adverse = (fs * fq * (mid[-1] - mid_q)).sum() / 100.0
    return capture, adverse


ARCHIVE_GLOBS = ("run_meta.json", "progress.csv", "vecnormalize.pkl",
                 "ppo_marl_lob.zip", "eval/*.npz")


def results_dir(repo: Path) -> Path:
    """Where bootstrap.sh told us to write. Falls back to results/ locally."""
    marker = repo / ".work" / "results_dir"
    if marker.is_file() and marker.read_text().strip():
        return Path(marker.read_text().strip())
    return repo / "results"


def archive_run(run_dir: Path, dest_root: Path) -> int:
    """Copy one run's raw artifacts under dest_root, preserving layout.

    Trajectories, checkpoints, normalisation stats and training curves all
    live in runs/, which teardown.sh removes. Copying them here is what makes
    the difference between "collect the numbers again" and "retrain".
    """
    copied = 0
    for pattern in ARCHIVE_GLOBS:
        for src in sorted(run_dir.glob(pattern)):
            dest = dest_root / src.relative_to(run_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--no-archive", action="store_true",
                    help="skip copying raw artifacts into the results dir")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / "runs" / args.suite
    if not root.exists():
        raise SystemExit(f"no runs found at {root.relative_to(repo)}")

    results = results_dir(repo)
    rows = []
    archived = 0
    for meta_path in sorted(root.rglob("run_meta.json")):
        meta = json.loads(meta_path.read_text())
        a = meta["args"]
        run_dir = meta_path.parent
        row = {
            "run": str(run_dir.relative_to(root)),
            "n_agents": a.get("n_agents"),
            "value_agents": meta.get("config_kwargs", {}).get("num_value_agents", "default"),
            "seed": a.get("seed"),
            "min_size": a.get("min_size"),
            "max_offset": a.get("max_offset_cents"),
            "norm_obs": a.get("norm_obs"),
            "inv_penalty": a.get("inventory_penalty"),
            "timesteps": a.get("total_timesteps"),
            "train_min": round(meta.get("train_seconds", 0) / 60, 1),
            "host": meta.get("host"),
            "git_sha": (meta.get("git_sha") or "")[:8],
        }
        # Filenames are <prefix>_<agent_idx>_seed<eval_seed>.npz. Keying only
        # on agent index would let the last eval seed silently overwrite the
        # others, so each seed gets its own column and the headline
        # d_equity_<agent> is the mean across them.
        per_agent: dict[str, dict[str, list[float]]] = {}
        for npz in sorted((run_dir / "eval").glob("*trajectory*.npz")):
            parts = npz.stem.split("_")
            agent, eval_seed = parts[-2], parts[-1].removeprefix("seed")
            acc = per_agent.setdefault(agent, {"d_equity": [], "capture": [], "adverse": []})

            d_eq = equity_change(npz)
            row[f"d_equity_{agent}_s{eval_seed}"] = round(d_eq, 0)
            acc["d_equity"].append(d_eq)

            cap, adv = markout(npz)
            if cap is not None:
                row[f"capture_{agent}_s{eval_seed}"] = round(cap, 0)
                row[f"adverse_{agent}_s{eval_seed}"] = round(adv, 0)
                acc["capture"].append(cap)
                acc["adverse"].append(adv)

        for agent, acc in per_agent.items():
            for metric, vals in acc.items():
                if vals:
                    row[f"{metric}_{agent}"] = round(float(np.mean(vals)), 0)
                    if len(vals) > 1 and metric == "d_equity":
                        # Spread across eval seeds - without it a single mean
                        # reads as far more certain than it is.
                        row[f"d_equity_{agent}_sd"] = round(float(np.std(vals)), 0)
        rows.append(row)

        if not args.no_archive:
            archived += archive_run(
                run_dir, results / "artifacts" / args.suite / row["run"]
            )

    if not rows:
        raise SystemExit("no run_meta.json found - did the suite finish?")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
    out = repo / "results" / f"{args.suite}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    show = [f for f in ("run", "n_agents", "value_agents", "seed", "norm_obs",
                        "train_min", "d_equity_0", "d_equity_1") if f in fields]
    widths = {f: max(len(f), max(len(str(r.get(f, ""))) for r in rows)) for f in show}
    print("  ".join(f.ljust(widths[f]) for f in show))
    print("-" * (sum(widths.values()) + 2 * (len(show) - 1)))
    for r in rows:
        print("  ".join(str(r.get(f, "")).ljust(widths[f]) for f in show))
    print(f"\nfull table ({len(fields)} columns) -> {out.relative_to(repo)}")
    if archived:
        print(f"archived {archived} raw artifacts -> "
              f"{results / 'artifacts' / args.suite}")
        print("these survive teardown.sh; copy results/ off the machine to keep them")


if __name__ == "__main__":
    main()
