"""Tests for the agent-divergence helpers.

The point of the script is to distinguish "these two agents are the same
policy producing the same output" from "these two agents diverged", so the
tests pin both extremes and the quantisation edge cases in between.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from agent_divergence import (  # noqa: E402
    effective_quotes,
    group_by_seed,
    obs_divergence,
    pairwise_report,
    safe_corr,
)

# ── safe_corr ───────────────────────────────────────────────────────────────

def test_safe_corr_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert safe_corr(x, x) == pytest.approx(1.0)


def test_safe_corr_anticorrelated():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert safe_corr(x, -x) == pytest.approx(-1.0)


def test_safe_corr_constant_is_nan_not_a_crash():
    """A frozen policy emits a constant; that must report, not raise."""
    const = np.zeros(10)
    assert np.isnan(safe_corr(const, np.arange(10.0)))
    assert np.isnan(safe_corr(const, const))


# ── effective_quotes ────────────────────────────────────────────────────────

def test_effective_quotes_subtick_action_becomes_no_quote():
    """The bug-4 case: offsets and sizes round to zero, so nothing is posted."""
    a = np.array([[0.3, 0.3, 0.3, 0.3]])
    assert effective_quotes(a).tolist() == [[0, 0, 0, 0]]


def test_effective_quotes_crossed_voids_both_sides():
    """Sizes survive rounding but both offsets round to 0, so bid == ask."""
    a = np.array([[0.4, 0.4, 10.0, 10.0]])
    assert effective_quotes(a).tolist() == [[0, 0, 0, 0]]


def test_effective_quotes_genuine_two_sided_market_survives():
    a = np.array([[2.0, 3.0, 10.0, 20.0]])
    assert effective_quotes(a).tolist() == [[2, 3, 10, 20]]


def test_effective_quotes_one_sided_is_kept():
    """A bid with no ask is a legal quote and must not be voided."""
    a = np.array([[2.0, 3.0, 10.0, 0.2]])
    assert effective_quotes(a).tolist() == [[2, 0, 10, 0]]


def test_effective_quotes_clips_size_to_max():
    a = np.array([[1.0, 1.0, 500.0, 500.0]])
    assert effective_quotes(a, max_size=100).tolist() == [[1, 1, 100, 100]]


# ── pairwise_report ─────────────────────────────────────────────────────────

def _rand(n=64, seed=0):
    return np.random.default_rng(seed).random((n, 4))


def test_pairwise_report_identical_agents():
    """Parameter-shared clones: 100% identical, zero relative difference."""
    a = _rand()
    r = pairwise_report(a, a.copy())
    assert r["identical_pct"] == pytest.approx(100.0)
    assert r["n_steps"] == 64
    for m in r["per_dim"].values():
        assert m["corr"] == pytest.approx(1.0)
        assert m["mean_abs_diff"] == pytest.approx(0.0)
        assert m["rel_diff"] == pytest.approx(0.0)


def test_pairwise_report_independent_agents():
    a, b = _rand(seed=0), _rand(seed=1)
    r = pairwise_report(a, b)
    assert r["identical_pct"] == pytest.approx(0.0)
    for m in r["per_dim"].values():
        assert abs(m["corr"]) < 0.5
        assert m["rel_diff"] > 0.5


def test_pairwise_report_truncates_to_shorter_agent():
    a, b = _rand(n=100), _rand(n=60)
    assert pairwise_report(a, b)["n_steps"] == 60


def test_pairwise_report_rel_diff_zero_when_both_constant():
    """Pooled std of 0 must not divide by zero."""
    a = np.zeros((10, 4))
    r = pairwise_report(a, a.copy())
    for m in r["per_dim"].values():
        assert m["rel_diff"] == 0.0


# ── obs_divergence ──────────────────────────────────────────────────────────

def test_obs_divergence_reports_only_differing_dims():
    names = ["f0", "f1", "f2"]
    o_a = np.zeros((5, 3))
    o_b = np.zeros((5, 3))
    o_b[:, 1] = 0.5
    out = obs_divergence(o_a, o_b, names)
    assert [n for n, _, _ in out] == ["f1"]
    assert out[0][1] == pytest.approx(0.5)


def test_obs_divergence_empty_when_identical():
    o = np.arange(15.0).reshape(5, 3)
    assert obs_divergence(o, o.copy(), ["a", "b", "c"]) == []


# ── group_by_seed ───────────────────────────────────────────────────────────

def test_group_by_seed_buckets_agents_within_a_seed():
    paths = [
        "runs/x/eval/ppo_trajectory_0_seed42.npz",
        "runs/x/eval/ppo_trajectory_1_seed42.npz",
        "runs/x/eval/ppo_trajectory_0_seed43.npz",
    ]
    g = group_by_seed(paths)
    assert sorted(g) == ["42", "43"]
    assert sorted(g["42"]) == [0, 1]
    assert sorted(g["43"]) == [0]


def test_group_by_seed_ignores_unrelated_files():
    assert group_by_seed(["runs/x/eval/vecnormalize.pkl", "notes.npz"]) == {}
