"""Tests for the agent-identity one-hot appended to the observation.

Measured across 4,350 agent pairs, the parameter-shared agents posted
identical quotes on 97-99.9% of steps: of the 4*K+4 features, only inventory
and cash could differ between two agents watching the same book, and that was
not enough to make them distinguishable
(`notes/agent_divergence_results.md`). The one-hot is what lets a shared
network condition on which agent it is.
"""
import numpy as np
import pytest

from marl_lob.observation_extractor import (
    describe_obs,
    extract_obs,
    obs_vector_size,
)

BOOK = dict(known_bids=[(9_990, 50), (9_980, 30)],
            known_asks=[(10_010, 40), (10_020, 20)])


def _obs(**kw):
    return extract_obs(inventory=0, cash=10_000_000, current_time=5,
                       mkt_open=0, mkt_close=10, **BOOK, **kw)


# ── sizing ──────────────────────────────────────────────────────────────────

def test_default_layout_is_unchanged():
    """Width 0 must reproduce the pre-2026-09 vector exactly, so legacy
    checkpoints keep evaluating against the contract they were trained on."""
    assert obs_vector_size() == 44
    assert _obs().shape == (44,)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_width_extends_the_vector(width):
    assert obs_vector_size(10, width) == 44 + width
    assert _obs(agent_index=0, agent_id_width=width).shape == (44 + width,)


def test_negative_width_is_treated_as_absent():
    assert obs_vector_size(10, -3) == 44


# ── contents ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_one_hot_marks_the_right_agent(idx):
    tail = _obs(agent_index=idx, agent_id_width=4)[-4:]
    assert tail.tolist() == [1.0 if i == idx else 0.0 for i in range(4)]


def test_agents_differ_even_with_identical_book_and_holdings():
    """The point of the change: two agents seeing the same book and holding
    the same position previously produced bit-identical observations."""
    a = _obs(agent_index=0, agent_id_width=2)
    b = _obs(agent_index=1, agent_id_width=2)
    assert not np.array_equal(a, b)
    assert np.array_equal(a[:-2], b[:-2]), "only the identity tail may differ"


def test_without_the_one_hot_they_are_identical():
    """The control for the test above — this is the measured failure."""
    assert np.array_equal(_obs(agent_index=0), _obs(agent_index=1))


def test_body_is_untouched_by_the_suffix():
    assert np.array_equal(_obs(), _obs(agent_index=1, agent_id_width=3)[:-3])


def test_index_outside_the_pinned_width_is_all_zeros():
    """A pinned width smaller than n_agents leaves the extra agents
    unidentifiable rather than raising or corrupting a neighbour's bit."""
    assert _obs(agent_index=9, agent_id_width=4)[-4:].tolist() == [0.0] * 4


def test_missing_index_is_all_zeros():
    assert _obs(agent_id_width=3)[-3:].tolist() == [0.0] * 3


def test_one_hot_survives_the_final_clip():
    """extract_obs clips to [-1, 1]; a hot bit of 1.0 must come through."""
    assert _obs(agent_index=1, agent_id_width=2)[-1] == 1.0


# ── names ───────────────────────────────────────────────────────────────────

def test_describe_obs_tracks_the_width():
    names = describe_obs(10, 3)
    assert len(names) == obs_vector_size(10, 3)
    assert names[-3:] == ["agent_id_0", "agent_id_1", "agent_id_2"]


def test_describe_obs_default_is_unchanged():
    assert len(describe_obs()) == 44
    assert describe_obs()[-1] == "time_to_close_norm"


# ─────────────────────────────────────────────────────────────────────────────
# Order-flow imbalance
#
# Everything else in the observation is a snapshot, so the agent cannot see the
# book *change* — and adverse selection is a statement about flow. grid2's P1
# (no widening against informed flow) may be a fact about that gap rather than
# about market making, which is what these features exist to test.
# ─────────────────────────────────────────────────────────────────────────────

from marl_lob.observation_extractor import (  # noqa: E402
    compute_ofi,
    squash_ofi,
    squash_ofi_ewma,
)

PB, PA = [(9_990, 50)], [(10_010, 40)]


def test_no_change_is_zero_flow():
    assert compute_ofi(PB, PA, PB, PA) == 0.0


def test_bid_queue_growth_is_buying_pressure():
    assert compute_ofi(PB, PA, [(9_990, 80)], PA) == +30.0


def test_ask_queue_growth_is_selling_pressure():
    assert compute_ofi(PB, PA, PB, [(10_010, 70)]) == -30.0


def test_bid_price_improving_is_buying_pressure():
    """Bid steps up: the new queue is added, the old one is not removed."""
    assert compute_ofi(PB, PA, [(10_000, 60)], PA) == +60.0


def test_ask_price_improving_is_selling_pressure():
    """An ask improves by moving DOWN, so the sign flips."""
    assert compute_ofi(PB, PA, PB, [(10_000, 40)]) == -40.0


def test_bid_pulled_away_is_selling_pressure():
    assert compute_ofi(PB, PA, [(9_980, 20)], PA) == -50.0


def test_symmetric_pressure_cancels():
    """Both queues grow by the same amount — no net imbalance."""
    assert compute_ofi(PB, PA, [(9_990, 80)], [(10_010, 70)]) == 0.0


@pytest.mark.parametrize("pb,pa,b,a", [
    ([], [], PB, PA),          # no previous book
    (PB, PA, [], PA),          # bid side vanished
    (PB, [], PB, PA),          # previous ask missing
])
def test_one_sided_or_empty_books_report_no_flow(pb, pa, b, a):
    """There is no flow to infer, and guessing would inject a fake signal."""
    assert compute_ofi(pb, pa, b, a) == 0.0


def test_the_two_channels_are_squashed_differently_on_purpose():
    """Theory said the EWMA would be ~0.16x the raw channel; measurement says
    its median is 3.7x LARGER, because OFI is fat-tailed and autocorrelated.
    So the two use different transforms, and this pins that they differ."""
    assert squash_ofi(1000.0, 100) != squash_ofi_ewma(1000.0, 100)


def test_raw_channel_keeps_resolution_across_the_measured_common_range():
    """Measured |OFI|: p50 40, p75 69, p90 107, p95 140. Those must map to
    clearly distinct values, or the feature cannot inform a quote."""
    vals = [squash_ofi(x, 100) for x in (40, 69, 107, 140)]
    assert vals == sorted(vals)
    assert vals[0] > 0.3 and vals[-1] < 0.95
    assert min(b - a for a, b in zip(vals, vals[1:], strict=False)) > 0.05


def test_ewma_channel_spans_its_much_wider_measured_range():
    """Measured |EWMA|: p50 148, p75 1012, p90 2299, p95 3210. tanh saturated
    12% of steps at its best scale; asinh keeps them separated."""
    vals = [squash_ofi_ewma(x, 100) for x in (148, 1012, 2299, 3210)]
    assert vals == sorted(vals)
    assert vals[0] > 0.15 and vals[-1] < 0.95
    assert min(b - a for a, b in zip(vals, vals[1:], strict=False)) > 0.05


@pytest.mark.parametrize("fn", [squash_ofi, squash_ofi_ewma])
def test_squashes_are_odd_and_monotone(fn):
    xs = [0, 1, 10, 100, 1_000, 100_000]
    ys = [fn(x, 100) for x in xs]
    assert ys == sorted(ys)
    for x in xs:
        assert fn(-x, 100) == pytest.approx(-fn(x, 100))
    assert all(-1.0 <= y <= 1.0 for y in ys)


def test_ofi_features_are_clipped():
    o = _obs(max_size=100, ofi=1e9, ofi_ewma=-1e9)
    assert o[-2] == 1.0 and o[-1] == -1.0


def test_ofi_absent_unless_asked_for():
    assert _obs().shape == (44,)
    assert obs_vector_size(10, 0, False) == 44


def test_ofi_adds_exactly_two_features_before_the_identity_block():
    assert obs_vector_size(10, 0, True) == 46
    assert obs_vector_size(10, 4, True) == 50
    names = describe_obs(10, 2, True)
    assert names[-4:] == ["ofi_norm", "ofi_ewma_norm", "agent_id_0", "agent_id_1"]
