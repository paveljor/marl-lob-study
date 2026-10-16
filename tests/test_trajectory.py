from __future__ import annotations

import numpy as np
import pytest

from marl_lob.trajectory import (
    Fill,
    Trajectory,
    load_trajectory,
    save_trajectory,
)


def test_from_tuples_records_signed_fills():
    rows = [
        (0.0, 0, 1_000_000, 10_000, 0,  0),
        (1.0, 5,   950_000, 10_010, 5,  10_011),   # buy 5 @ 10011 (mid was 10010)
        (2.0, 3,   970_000, 10_020, -2, 10_019),   # sell 2 @ 10019 (mid was 10020)
        (3.0, 3,   970_000, 10_030, 0,  0),
    ]
    traj = Trajectory.from_tuples(rows)
    assert len(traj) == 4
    assert len(traj.fills) == 2
    assert traj.fills[0] == Fill(timestamp=1.0, side=1, price=10_011, quantity=5)
    assert traj.fills[1] == Fill(timestamp=2.0, side=-1, price=10_019, quantity=2)


def test_from_tuples_fill_price_distinct_from_mid():
    """Pin the contract: Fill.price comes from fill_price, not mid_price."""
    rows = [(0.0, 0, 0, 99_999, 1, 100_005)]   # mid 99999, fill at 100005
    traj = Trajectory.from_tuples(rows)
    assert traj.fills[0].price == 100_005
    assert traj.mid_price[0] == 99_999


def test_from_tuples_empty():
    traj = Trajectory.from_tuples([])
    assert len(traj) == 0
    assert traj.fills == []


def test_equity_marks_to_market():
    traj = Trajectory(
        timestamps=np.array([0.0, 1.0]),
        inventory=np.array([10, 5], dtype=np.int64),
        cash=np.array([100_000, 150_000], dtype=np.int64),
        mid_price=np.array([10_000, 10_000], dtype=np.int64),
    )
    eq = traj.equity()
    # cash + inventory * mid
    assert eq.tolist() == [100_000 + 10 * 10_000, 150_000 + 5 * 10_000]


def test_dtype_guards_reject_floats():
    with pytest.raises(TypeError):
        Trajectory(
            timestamps=np.array([0.0]),
            inventory=np.array([0], dtype=np.int64),
            cash=np.array([1.0]),  # float — should be cents
            mid_price=np.array([10_000], dtype=np.int64),
        )


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Trajectory(
            timestamps=np.array([0.0, 1.0]),
            inventory=np.array([0], dtype=np.int64),
            cash=np.array([0, 0], dtype=np.int64),
            mid_price=np.array([10_000, 10_000], dtype=np.int64),
        )


# ── npz round-trip ───────────────────────────────────────────────────────────
# These exist because eval.py silently dropped fills on save for four months:
# the data was in memory, printed to stdout, and thrown away at np.savez.

def _traj_with_fills():
    rows = [
        (0.0, 0, 100_000, 10_000, 0, 0),
        (1.0, 5, 49_950, 10_010, 5, 9_990),
        (2.0, 0, 100_100, 10_020, -5, 10_030),
    ]
    return Trajectory.from_tuples(rows)


def test_save_load_round_trip_preserves_fills(tmp_path):
    traj = _traj_with_fills()
    assert len(traj.fills) == 2  # guard: the fixture must actually have fills

    path = tmp_path / "t.npz"
    save_trajectory(path, traj)
    back = load_trajectory(path)

    assert len(back.fills) == len(traj.fills)
    for a, b in zip(traj.fills, back.fills, strict=True):
        assert (a.timestamp, a.side, a.price, a.quantity) == (
            b.timestamp, b.side, b.price, b.quantity
        )
    np.testing.assert_array_equal(back.inventory, traj.inventory)
    np.testing.assert_array_equal(back.cash, traj.cash)
    np.testing.assert_array_equal(back.mid_price, traj.mid_price)
    np.testing.assert_array_equal(back.timestamps, traj.timestamps)


def test_round_trip_preserves_integer_cents_dtypes(tmp_path):
    """Trajectory.__post_init__ rejects float cash/mid — so the load path must
    hand back integer arrays, not float64."""
    path = tmp_path / "t.npz"
    save_trajectory(path, _traj_with_fills())
    back = load_trajectory(path)
    assert back.cash.dtype.kind in "iu"
    assert back.mid_price.dtype.kind in "iu"
    assert back.inventory.dtype.kind in "iu"


def test_empty_fills_round_trip(tmp_path):
    """A run with zero fills is a real outcome here — PPO did exactly that —
    so it must not crash the loader or corrupt dtypes."""
    traj = Trajectory.from_tuples([(0.0, 0, 100_000, 10_000, 0, 0)])
    assert traj.fills == []
    path = tmp_path / "t.npz"
    save_trajectory(path, traj)
    back = load_trajectory(path)
    assert back.fills == []
    assert np.load(path)["fill_price"].dtype == np.int64


def test_load_tolerates_legacy_file_without_fill_arrays(tmp_path):
    """Trajectories saved by the old eval.py have only four arrays."""
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        timestamps=np.array([0.0, 1.0]),
        inventory=np.array([0, 5], dtype=np.int64),
        cash=np.array([100_000, 49_950], dtype=np.int64),
        mid_price=np.array([10_000, 10_010], dtype=np.int64),
    )
    back = load_trajectory(path)
    assert back.fills == []
    assert len(back) == 2


def test_extra_arrays_are_stored_alongside(tmp_path):
    """actions/observations ride along without the Trajectory knowing about them."""
    path = tmp_path / "t.npz"
    actions = np.arange(8, dtype=np.float32).reshape(2, 4)
    save_trajectory(path, _traj_with_fills(), actions=actions)

    np.testing.assert_array_equal(np.load(path)["actions"], actions)
    assert len(load_trajectory(path).fills) == 2  # extras don't disturb the load
