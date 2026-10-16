"""Tests for the MARL coordinator + child agents (Module C, chunk 1).

Pure-Python logic only. The ABIDES-derived classes (`MarlChild`,
`MarlCoordinator`) are thin wrappers around these helpers; they are exercised
in the integration test (chunk 6) once the env wiring lands.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from marl_lob.marl_agents import (
    ChildSnapshot,
    ExecutedFill,
    build_per_agent_state,
    distribute_actions,
    snapshot_fills,
)

# ─────────────────────────────────────────────────────────────────────────────
# distribute_actions — coordinator → children plumbing
# ─────────────────────────────────────────────────────────────────────────────

def test_distribute_actions_writes_each_pending_action():
    children = [MagicMock(pending_action=None) for _ in range(3)]
    actions = [
        {"agent_idx": 0, "action_vec": (1.0, 1.0, 5.0, 5.0)},
        {"agent_idx": 1, "action_vec": (2.0, 2.0, 3.0, 3.0)},
        {"agent_idx": 2, "action_vec": (0.0, 0.0, 0.0, 0.0)},
    ]
    distribute_actions(actions, children)
    assert children[0].pending_action == (1.0, 1.0, 5.0, 5.0)
    assert children[1].pending_action == (2.0, 2.0, 3.0, 3.0)
    assert children[2].pending_action == (0.0, 0.0, 0.0, 0.0)


def test_distribute_actions_out_of_order_indices():
    """Coordinator must not assume action_list is sorted by agent_idx."""
    children = [MagicMock(pending_action=None) for _ in range(2)]
    actions = [
        {"agent_idx": 1, "action_vec": (9, 9, 9, 9)},
        {"agent_idx": 0, "action_vec": (1, 1, 1, 1)},
    ]
    distribute_actions(actions, children)
    assert children[0].pending_action == (1, 1, 1, 1)
    assert children[1].pending_action == (9, 9, 9, 9)


def test_distribute_actions_rejects_unknown_index():
    children = [MagicMock(pending_action=None)]
    with pytest.raises(IndexError):
        distribute_actions([{"agent_idx": 5, "action_vec": (0, 0, 0, 0)}], children)


# ─────────────────────────────────────────────────────────────────────────────
# snapshot_fills — drain ABIDES inter-wakeup orders into ExecutedFill records
# ─────────────────────────────────────────────────────────────────────────────

def _fake_order(is_buy: bool, price: int, qty: int):
    """Mock that matches real ABIDES Order: `.side.is_bid()`, `.fill_price`, `.quantity`."""
    side = MagicMock()
    side.is_bid.return_value = is_buy
    o = MagicMock()
    o.side = side
    o.fill_price = price
    o.quantity = qty
    return o


def test_snapshot_fills_signs_buys_positive_sells_negative():
    orders = [
        _fake_order(is_buy=True, price=10_000, qty=5),
        _fake_order(is_buy=False, price=10_002, qty=3),
    ]
    fills = snapshot_fills(orders)
    assert fills == [
        ExecutedFill(signed_qty=5, price_cents=10_000),
        ExecutedFill(signed_qty=-3, price_cents=10_002),
    ]


def test_snapshot_fills_handles_empty():
    assert snapshot_fills([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# build_per_agent_state — the coordinator's wakeup return value
# ─────────────────────────────────────────────────────────────────────────────

def _bids_asks_at(mid: int, spread: int = 2):
    return (
        [(mid - spread // 2, 100), (mid - spread // 2 - 1, 80)],
        [(mid + spread // 2, 100), (mid + spread // 2 + 1, 80)],
    )


def test_build_per_agent_state_shapes_and_keys():
    bids, asks = _bids_asks_at(mid=10_000)
    snaps = [
        ChildSnapshot(inventory=10, cash=10_000_000, fills=[]),
        ChildSnapshot(inventory=-3, cash=9_999_500, fills=[]),
    ]
    out = build_per_agent_state(
        snaps,
        known_bids=bids,
        known_asks=asks,
        current_time_ns=10_000_000_000,
        mkt_open_ns=0,
        mkt_close_ns=3_600_000_000_000,
    )
    assert len(out) == 2
    for entry in out:
        assert set(entry.keys()) == {
            "obs", "inventory", "cash", "mid_cents",
            "fill_signed_qty", "fill_price_cents",
        }
        assert isinstance(entry["obs"], np.ndarray)
        assert entry["obs"].shape == (44,)        # 4*K+4 with K=10
        assert entry["obs"].dtype == np.float32


def test_build_per_agent_state_mid_is_midpoint_of_l1():
    bids, asks = _bids_asks_at(mid=10_000, spread=4)   # bid=9998, ask=10002
    snaps = [ChildSnapshot(inventory=0, cash=10_000_000, fills=[])]
    out = build_per_agent_state(
        snaps, known_bids=bids, known_asks=asks,
        current_time_ns=0, mkt_open_ns=0, mkt_close_ns=3_600_000_000_000,
    )
    assert out[0]["mid_cents"] == 10_000


def test_build_per_agent_state_fill_aggregation_net_buy():
    snaps = [ChildSnapshot(inventory=0, cash=10_000_000, fills=[
        ExecutedFill(signed_qty=10, price_cents=10_000),
        ExecutedFill(signed_qty=5, price_cents=10_002),
    ])]
    bids, asks = _bids_asks_at(mid=10_000)
    out = build_per_agent_state(
        snaps, known_bids=bids, known_asks=asks,
        current_time_ns=0, mkt_open_ns=0, mkt_close_ns=3_600_000_000_000,
    )
    assert out[0]["fill_signed_qty"] == 15
    # vwap weighted by abs(qty): (10*10000 + 5*10002)/15
    assert out[0]["fill_price_cents"] == int(round((10 * 10_000 + 5 * 10_002) / 15))


def test_build_per_agent_state_fill_aggregation_net_sell():
    snaps = [ChildSnapshot(inventory=0, cash=10_000_000, fills=[
        ExecutedFill(signed_qty=-7, price_cents=9_998),
    ])]
    bids, asks = _bids_asks_at(mid=10_000)
    out = build_per_agent_state(
        snaps, known_bids=bids, known_asks=asks,
        current_time_ns=0, mkt_open_ns=0, mkt_close_ns=3_600_000_000_000,
    )
    assert out[0]["fill_signed_qty"] == -7
    assert out[0]["fill_price_cents"] == 9_998


def test_build_per_agent_state_no_fills_zero_price():
    """Convention: when fill_signed_qty is 0, fill_price_cents is 0."""
    snaps = [ChildSnapshot(inventory=0, cash=10_000_000, fills=[])]
    bids, asks = _bids_asks_at(mid=10_000)
    out = build_per_agent_state(
        snaps, known_bids=bids, known_asks=asks,
        current_time_ns=0, mkt_open_ns=0, mkt_close_ns=3_600_000_000_000,
    )
    assert out[0]["fill_signed_qty"] == 0
    assert out[0]["fill_price_cents"] == 0


def test_build_per_agent_state_inventory_cash_passthrough():
    snaps = [
        ChildSnapshot(inventory=42, cash=12_345_678, fills=[]),
        ChildSnapshot(inventory=-13, cash=8_000_000, fills=[]),
    ]
    bids, asks = _bids_asks_at(mid=10_000)
    out = build_per_agent_state(
        snaps, known_bids=bids, known_asks=asks,
        current_time_ns=0, mkt_open_ns=0, mkt_close_ns=3_600_000_000_000,
    )
    assert out[0]["inventory"] == 42 and out[0]["cash"] == 12_345_678
    assert out[1]["inventory"] == -13 and out[1]["cash"] == 8_000_000
