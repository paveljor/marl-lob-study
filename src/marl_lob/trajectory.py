"""Trajectory data structure for the metrics harness.

Cents-everywhere unit convention: `cash` and `mid_price` are both stored as
integer cents to avoid float drift across long episodes. Convert to dollars
only at presentation time.

A trajectory row corresponds to one wrapper step:
    (timestamp, inventory, cash, mid_price, fill_qty, fill_price)

where `fill_qty` is a signed integer trade quantity for that step (+N for
buy, -N for sell, 0 for no fill) and `fill_price` is the cents-VWAP of all
fills that landed during the step (or 0 if no fill). The wrapper (Module C)
is the producer; this module is the consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Fill:
    timestamp: float
    side: int          # +1 buy, -1 sell
    price: int         # cents
    quantity: int      # always positive; direction lives in `side`


@dataclass(frozen=True)
class Trajectory:
    timestamps: np.ndarray   # float seconds, shape (T,)
    inventory: np.ndarray    # int signed shares, shape (T,)
    cash: np.ndarray         # int cents, shape (T,)
    mid_price: np.ndarray    # int cents, shape (T,)
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        T = self.timestamps.shape[0]
        for name, arr in (
            ("inventory", self.inventory),
            ("cash", self.cash),
            ("mid_price", self.mid_price),
        ):
            if arr.shape != (T,):
                raise ValueError(f"{name} shape {arr.shape} != ({T},)")
        if self.cash.dtype.kind not in "iu":
            raise TypeError(f"cash must be integer cents, got dtype {self.cash.dtype}")
        if self.mid_price.dtype.kind not in "iu":
            raise TypeError(f"mid_price must be integer cents, got dtype {self.mid_price.dtype}")
        if self.inventory.dtype.kind not in "iu":
            raise TypeError(f"inventory must be integer shares, got dtype {self.inventory.dtype}")

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    def equity(self) -> np.ndarray:
        """Mark-to-market equity in cents: cash + inventory * mid_price."""
        return self.cash + self.inventory * self.mid_price

    @classmethod
    def from_tuples(cls, rows: list[tuple]) -> Trajectory:
        """Build a Trajectory from 6-tuple rows
        ``(timestamp, inventory, cash, mid_price, fill_qty, fill_price)``.

        Non-zero ``fill_qty`` rows are also recorded in the ``fills`` list,
        with ``Fill.price`` taken from the row's ``fill_price`` (the actual
        VWAP of fills, not the mid).
        """
        if not rows:
            return cls(
                timestamps=np.zeros(0, dtype=float),
                inventory=np.zeros(0, dtype=np.int64),
                cash=np.zeros(0, dtype=np.int64),
                mid_price=np.zeros(0, dtype=np.int64),
                fills=[],
            )
        timestamps = np.array([r[0] for r in rows], dtype=float)
        inventory = np.array([r[1] for r in rows], dtype=np.int64)
        cash = np.array([r[2] for r in rows], dtype=np.int64)
        mid_price = np.array([r[3] for r in rows], dtype=np.int64)

        fills: list[Fill] = []
        for ts, _inv, _c, _mid, fill_qty, fill_price in rows:
            if fill_qty:
                fills.append(
                    Fill(
                        timestamp=float(ts),
                        side=1 if fill_qty > 0 else -1,
                        price=int(fill_price),
                        quantity=int(abs(fill_qty)),
                    )
                )
        return cls(
            timestamps=timestamps,
            inventory=inventory,
            cash=cash,
            mid_price=mid_price,
            fills=fills,
        )


# ─────────────────────────────────────────────────────────────────────────────
# npz persistence
#
# One format, three producers (run_baseline.py, eval.py, and any future
# rollout script) and three consumers (eval.py's F loader, scripts/markout.py,
# experiments/collect.py). Fills are stored as four parallel arrays rather than
# a structured dtype so the file stays readable by a bare `np.load` in a
# notebook. `extra` lets a caller attach run-specific arrays (actions,
# observations) without teaching this module about them.
# ─────────────────────────────────────────────────────────────────────────────

FILL_KEYS = ("fill_timestamps", "fill_side", "fill_price", "fill_quantity")


def fill_arrays(fills: list[Fill]) -> dict[str, np.ndarray]:
    """The four parallel fill arrays, correctly typed even when empty.

    An empty `np.array([])` defaults to float64, which would make
    `fill_price.astype(int64)` a lossy round-trip; the explicit dtypes here
    keep a fill-less run loadable by the same code path as a busy one.
    """
    return {
        "fill_timestamps": np.array([f.timestamp for f in fills], dtype=float),
        "fill_side": np.array([f.side for f in fills], dtype=np.int64),
        "fill_price": np.array([f.price for f in fills], dtype=np.int64),
        "fill_quantity": np.array([f.quantity for f in fills], dtype=np.int64),
    }


def save_trajectory(path, traj: Trajectory, **extra: np.ndarray) -> None:
    """Write a Trajectory (including fills) plus any extra arrays to `path`."""
    np.savez(
        path,
        timestamps=traj.timestamps,
        inventory=traj.inventory,
        cash=traj.cash,
        mid_price=traj.mid_price,
        **fill_arrays(traj.fills),
        **extra,
    )


def load_trajectory(path) -> Trajectory:
    """Read back a Trajectory written by `save_trajectory`.

    Tolerates files with no fill arrays — trajectories saved before fills were
    persisted load as a fill-less Trajectory rather than raising.
    """
    d = np.load(path)
    fills: list[Fill] = []
    if "fill_timestamps" in d:
        fills = [
            Fill(
                timestamp=float(d["fill_timestamps"][i]),
                side=int(d["fill_side"][i]),
                price=int(d["fill_price"][i]),
                quantity=int(d["fill_quantity"][i]),
            )
            for i in range(len(d["fill_timestamps"]))
        ]
    return Trajectory(
        timestamps=d["timestamps"],
        inventory=d["inventory"],
        cash=d["cash"],
        mid_price=d["mid_price"],
        fills=fills,
    )
