"""Tests for the saved-action replay helpers.

Two action layouts coexist on disk — 4-column trajectories written before the
gated space, 6-column ones after — so the central risk these tests guard is a
*silent* misread: a 6-wide array interpreted as the legacy layout reports the
gates as offsets and produces plausible, wrong numbers.
"""
import numpy as np
import pytest

from marl_lob.quote_analysis import (
    GATED_ACTION_NAMES,
    LEGACY_ACTION_NAMES,
    action_names,
    effective_quotes,
    is_gated,
    participation_stats,
)

# ── layout dispatch ─────────────────────────────────────────────────────────

def test_action_names_dispatch_on_width():
    assert action_names(4) == LEGACY_ACTION_NAMES
    assert action_names(6) == GATED_ACTION_NAMES


@pytest.mark.parametrize("width", [0, 1, 3, 5, 7])
def test_unrecognised_width_raises_rather_than_guessing(width):
    with pytest.raises(ValueError, match="unrecognised action width"):
        action_names(width)


def test_is_gated_reads_the_width():
    assert is_gated(np.zeros((3, 6)))
    assert not is_gated(np.zeros((3, 4)))


# ── effective_quotes, legacy layout ─────────────────────────────────────────

def test_legacy_subtick_action_posts_nothing():
    assert effective_quotes(np.array([[0.3, 0.3, 0.3, 0.3]])).tolist() == [[0, 0, 0, 0]]


def test_legacy_crossed_offsets_void_both_sides():
    assert effective_quotes(np.array([[0.4, 0.4, 10.0, 10.0]])).tolist() == [[0, 0, 0, 0]]


def test_legacy_genuine_market_survives():
    assert effective_quotes(np.array([[2.0, 3.0, 10.0, 20.0]])).tolist() == [[2, 3, 10, 20]]


# ── effective_quotes, gated layout ──────────────────────────────────────────

def test_gated_subtick_action_becomes_a_real_quote():
    """The same numbers that post nothing under the legacy layout."""
    assert effective_quotes(
        np.array([[1.0, 1.0, 0.3, 0.3, 0.3, 0.3]])).tolist() == [[1, 1, 1, 1]]


def test_gated_closed_gates_post_nothing():
    assert effective_quotes(
        np.array([[-1.0, -1.0, 9.0, 9.0, 50.0, 50.0]])).tolist() == [[0, 0, 0, 0]]


def test_gated_one_sided_keeps_the_open_side():
    assert effective_quotes(
        np.array([[1.0, -1.0, 3.0, 3.0, 20.0, 20.0]])).tolist() == [[3, 0, 20, 0]]


def test_gated_never_crosses_at_any_offset():
    rows = np.array([[1.0, 1.0, off, off, 10.0, 10.0]
                     for off in (0.0, 0.1, 0.4, 0.6, 1.0, 7.3)])
    q = effective_quotes(rows)
    assert (q[:, 2] > 0).all() and (q[:, 3] > 0).all()


def test_reading_a_gated_array_as_legacy_would_be_wrong():
    """Pins the misread this module exists to prevent: the first two gated
    columns are gates, and treating them as offsets changes the answer."""
    gated_row = np.array([[1.0, 1.0, 4.0, 4.0, 10.0, 10.0]])
    correct = effective_quotes(gated_row)
    misread = effective_quotes(gated_row[:, :4])   # gates read as offsets
    assert correct.tolist() == [[4, 4, 10, 10]]
    assert misread.tolist() != correct.tolist()


# ── participation_stats ─────────────────────────────────────────────────────

def test_participation_of_a_dead_legacy_policy():
    """The measured 2026-09 grid behaviour: never quotes, nothing crosses
    because the sizes round away first."""
    m = participation_stats(np.array([[0.3, 0.3, 0.3, 0.3]] * 100))
    assert m["two_sided_pct"] == 0.0
    assert m["withdrawn_pct"] == 100.0
    assert m["gated"] is False


def test_participation_counts_legacy_self_voids_as_crossed():
    m = participation_stats(np.array([[0.4, 0.4, 10.0, 10.0]] * 10))
    assert m["crossed_pct"] == 100.0
    assert m["two_sided_pct"] == 0.0


def test_participation_of_a_half_withdrawn_gated_policy():
    rows = np.array([[1.0, 1.0, 2.0, 2.0, 5.0, 5.0]] * 50
                    + [[-1.0, -1.0, 2.0, 2.0, 5.0, 5.0]] * 50)
    m = participation_stats(rows)
    assert m["two_sided_pct"] == 50.0
    assert m["withdrawn_pct"] == 50.0
    assert m["spread_cents"] == 4.0     # 2c each side
    assert m["quote_size"] == 5.0
    assert m["gated"] is True


def test_gated_policy_never_reports_crossings():
    """Structural, not incidental — worth pinning so a future change to the
    floors cannot silently reintroduce self-voiding."""
    rng = np.random.default_rng(0)
    rows = np.column_stack([
        rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500),
        rng.uniform(0, 0.9, 500), rng.uniform(0, 0.9, 500),
        rng.uniform(0, 0.9, 500), rng.uniform(0, 0.9, 500),
    ])
    assert participation_stats(rows)["crossed_pct"] == 0.0


def test_withdrawal_is_measurable_across_the_gate_range():
    """Withdrawal rate is the dependent variable the gated space exists to
    expose, so it must track the gates rather than the offsets or sizes."""
    for frac in (0.0, 0.25, 0.75, 1.0):
        n_off = int(100 * frac)
        rows = np.array([[-1.0, -1.0, 2.0, 2.0, 5.0, 5.0]] * n_off
                        + [[1.0, 1.0, 2.0, 2.0, 5.0, 5.0]] * (100 - n_off))
        assert participation_stats(rows)["withdrawn_pct"] == pytest.approx(frac * 100)
