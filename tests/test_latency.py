"""Tests for the latency edge.

Speed is the canonical market-maker advantage and grid2 makes it the natural
next question: with no viable region anywhere, does an edge create one? These
pin the mechanics so that a result can be attributed to the edge rather than to
a wiring mistake.
"""
import numpy as np
import pytest

from marl_lob.latency import (
    EXCHANGE_ID,
    apply_latency_edge,
    describe_edge,
    median_latency_to_exchange,
)


def _matrix(latencies_to_exchange):
    """Square pairwise matrix; index 0 is the exchange."""
    n = len(latencies_to_exchange) + 1
    m = np.zeros((n, n), dtype=int)
    for i, lat in enumerate(latencies_to_exchange, start=1):
        m[i, 0] = m[0, i] = lat
    return m


def test_median_ignores_the_exchange_itself():
    m = _matrix([100, 200, 300])
    assert median_latency_to_exchange(m) == 200


def test_median_excludes_our_own_agents():
    """Otherwise the yardstick moves as we change the thing being measured."""
    m = _matrix([100, 200, 300, 1])          # agent 4 is ours, absurdly fast
    assert median_latency_to_exchange(m, exclude=(4,)) == 200


def test_factor_one_is_the_population_median_not_the_random_draw():
    """The point of pinning: 'no edge' must mean 'typical participant', not
    'wherever generate_latency_model happened to put us'."""
    m = _matrix([100, 200, 300, 9_999])
    out = apply_latency_edge(m, our_ids=(4,), factor=1.0)
    assert out[4, 0] == 200 and out[0, 4] == 200


@pytest.mark.parametrize("factor,expected", [
    (0.0, 0), (0.1, 20), (0.5, 100), (1.0, 200), (2.0, 400),
])
def test_factor_scales_the_median(factor, expected):
    m = _matrix([100, 200, 300, 555])
    out = apply_latency_edge(m, our_ids=(4,), factor=factor)
    assert out[4, 0] == expected


def test_edge_is_symmetric():
    m = _matrix([100, 200, 300, 555])
    out = apply_latency_edge(m, our_ids=(4,), factor=0.25)
    assert out[4, 0] == out[0, 4] == 50


def test_all_our_agents_get_the_same_edge():
    """A latency asymmetry *between* our own agents would confound the
    competition axis with the speed axis."""
    m = _matrix([100, 200, 300, 111, 222])
    out = apply_latency_edge(m, our_ids=(4, 5), factor=0.5)
    assert out[4, 0] == out[5, 0] == 100


def test_background_pairs_are_untouched():
    """Only our link to the exchange changes; the population's own structure
    has to survive or the market itself is a different market."""
    m = _matrix([100, 200, 300, 555])
    m[1, 2] = m[2, 1] = 77
    out = apply_latency_edge(m, our_ids=(4,), factor=0.0)
    assert out[1, 2] == out[2, 1] == 77
    for i in (1, 2, 3):
        assert out[i, 0] == m[i, 0]


def test_input_is_not_mutated():
    m = _matrix([100, 200, 300, 555])
    before = m.copy()
    apply_latency_edge(m, our_ids=(4,), factor=0.0)
    assert np.array_equal(m, before)


def test_negative_factor_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        apply_latency_edge(_matrix([100, 200]), our_ids=(2,), factor=-1.0)


def test_exchange_cannot_have_an_edge_over_itself():
    with pytest.raises(ValueError, match="over itself"):
        apply_latency_edge(_matrix([100, 200]), our_ids=(EXCHANGE_ID,), factor=0.5)


def test_describe_edge_reports_position_in_the_population():
    m = _matrix([100, 200, 300, 400, 50])
    d = describe_edge(m, our_ids=(5,))
    assert d["ours_ns"] == 50
    assert d["median_ns"] == 250
    assert d["percentile"] == 100.0, "faster than every background agent"


def test_describe_edge_median_participant_sits_mid_pack():
    m = _matrix([100, 200, 300, 400, 250])
    d = describe_edge(m, our_ids=(5,))
    assert d["percentile"] == 50.0
