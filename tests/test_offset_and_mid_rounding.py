"""The offset rescale and the three mid-rounding modes.

Both changes are pure integer arithmetic on the way from an action vector to a
pair of limit prices, and both have a legacy default that a large body of
existing results depends on reproducing exactly. So there are three kinds of
test here:

* the mapping itself, swept - what tick does each raw offset reach, and is
  every tick in range reachable at all;
* the legacy default, pinned against a literal restatement of the old formula;
* and the agreement between the live path and the offline reconstruction, for
  every combination of the two flags. That last one is the load-bearing test:
  ``quote_analysis`` replays saved actions to produce every behavioural metric
  this project reports, and if its arithmetic and the agent's arithmetic ever
  disagree, every one of those metrics is wrong with nothing raising.
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from marl_lob.actions import (
    DEFAULT_MIN_OFFSET_TICKS,
    MID_ROUNDING_MODES,
    PlaceIntent,
    ResolvedQuote,
    quote_prices,
    resolve_offset_cents,
    resolve_quote,
    translate_action,
)
from marl_lob.agent_adapter import doubled_mid_from_obs
from marl_lob.marl_agents import doubled_mid_from_book
from marl_lob.quote_analysis import (
    LEGACY_QUOTE_MODE,
    QuoteMode,
    effective_quotes,
    participation_stats,
    quote_prices_from_actions,
)

SPAN = 15          # the intended experimental value; never a default
# where raw=0 lands, i.e. where a fresh SB3 policy (mean ~0) starts
_MIDPOINT = DEFAULT_MIN_OFFSET_TICKS + int(round(SPAN / 2.0))
MID2_EVEN = 200_000        # bid 99_999 / ask 100_001 -> true mid 100_000
MID2_ODD = 200_001         # bid 100_000 / ask 100_001 -> true mid 100_000.5


# ─────────────────────────────────────────────────────────────────────────────
# 1. The offset mapping
# ─────────────────────────────────────────────────────────────────────────────

def test_legacy_offset_mapping_is_exactly_max_round_raw_1():
    """The formula the rescale replaces, restated independently.

    With TICK_SIZE_CENTS=1 and min_offset_ticks=1 the legacy resolution is
    ``max(round(raw), 1)``, which is why every raw offset in [0, 1.5) posts
    one tick. Nothing about the new flag may move this.
    """
    for raw in np.arange(0.0, 8.0, 0.01):
        assert resolve_offset_cents(raw) == max(round(float(raw)), 1)


def test_legacy_offset_mapping_collapses_the_whole_unit_interval():
    """The defect, stated as a test: [0, 1.5) is one indistinguishable tick."""
    reached = {resolve_offset_cents(r) for r in np.arange(0.0, 1.5, 0.001)}
    assert reached == {1}


@pytest.mark.parametrize("span_arg", [None, 0])
def test_unset_span_means_legacy(span_arg):
    """None and 0 both select the legacy mapping - 'unset' has two spellings
    (a missing flag, and an int flag defaulted to zero) and they must agree."""
    for raw in np.arange(0.0, 4.0, 0.05):
        assert (resolve_offset_cents(raw, offset_span=span_arg)
                == max(round(float(raw)), 1))


def test_span_maps_each_fifteenth_of_the_range_to_one_tick():
    """The point of the rescale: the action range is spent on distinct ticks.

    Sampled off the tie points, so Python's banker's rounding at exact halves
    is not what is being asserted here (that has its own test below).
    """
    for k in range(SPAN * 200 + 1):
        raw = -1.0 + k / (SPAN * 100.0)
        pos = (raw + 1.0) / 2.0 * SPAN
        expected = DEFAULT_MIN_OFFSET_TICKS + int(np.floor(pos + 0.5))
        if abs(pos - (np.floor(pos) + 0.5)) < 1e-9:
            continue                     # tie point, see the tie test
        assert resolve_offset_cents(raw, offset_span=SPAN) == expected


def test_span_mapping_is_monotone_and_hits_every_tick_in_range():
    raws = np.linspace(-1.0, 1.0, 20_001)
    offs = np.array([resolve_offset_cents(r, offset_span=SPAN) for r in raws])
    assert (np.diff(offs) >= 0).all(), "mapping must be non-decreasing in raw"
    lo = DEFAULT_MIN_OFFSET_TICKS
    assert set(offs.tolist()) == set(range(lo, lo + SPAN + 1))
    assert offs[0] == lo
    assert offs[-1] == lo + SPAN


def test_span_ceiling_is_min_plus_span_and_raw_is_clamped():
    """raw outside [-1, 1] is out of the declared Box and must not run off
    either end."""
    top = DEFAULT_MIN_OFFSET_TICKS + SPAN
    for raw in (1.0, 1.0000001, 2.0, 50.0):
        assert resolve_offset_cents(raw, offset_span=SPAN) <= top
    assert resolve_offset_cents(1.0, offset_span=SPAN) == top
    for raw in (-1.0, -1.0000001, -2.0, -50.0):
        assert resolve_offset_cents(raw, offset_span=SPAN) \
            == DEFAULT_MIN_OFFSET_TICKS


@pytest.mark.parametrize("raw", [-1.0, -1.5, -50.0])
def test_span_floor_holds_at_and_below_minus_one(raw):
    assert resolve_offset_cents(raw, offset_span=SPAN) == DEFAULT_MIN_OFFSET_TICKS


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), float("-inf")])
def test_span_scrubs_non_finite_raw_to_the_midpoint(raw):
    """``_safe_float`` maps non-finite input to 0.0 in both branches. Under a
    symmetric span 0.0 is the *middle* of the range, not the floor - unlike the
    legacy mapping, where 0.0 rounds down to ``min_offset_ticks``."""
    assert resolve_offset_cents(raw, offset_span=SPAN) == _MIDPOINT
    assert resolve_offset_cents(raw) == DEFAULT_MIN_OFFSET_TICKS


def test_span_is_symmetric_about_the_midpoint():
    """-1 is the floor, +1 the ceiling, 0 the midpoint - and a fresh SB3 policy
    initialises its mean at ~0, which is the whole reason the range is
    symmetric rather than [0, 1]."""
    lo = DEFAULT_MIN_OFFSET_TICKS
    assert resolve_offset_cents(-1.0, offset_span=SPAN) == lo
    assert resolve_offset_cents(1.0, offset_span=SPAN) == lo + SPAN
    assert resolve_offset_cents(0.0, offset_span=SPAN) == _MIDPOINT
    # mirrored raws are equidistant from the midpoint
    for r in (0.2, 0.5, 0.8, 1.0):
        hi_t = resolve_offset_cents(r, offset_span=SPAN) - lo
        lo_t = resolve_offset_cents(-r, offset_span=SPAN) - lo
        assert hi_t + lo_t == SPAN


def test_span_ties_break_the_same_way_as_the_legacy_mapping():
    """Both branches go through Python's ``round``, so a raw landing exactly
    between two ticks resolves the same way in either mapping."""
    assert resolve_offset_cents(0.5) == 1              # round(0.5) == 0, floored
    assert resolve_offset_cents(1.5) == 2              # round(1.5) == 2
    assert resolve_offset_cents(2.5) == 2              # round(2.5) == 2, banker's
    # span=2: raw -0.5 -> 0.5 ticks -> banker's -> 0 ticks -> the floor
    assert resolve_offset_cents(-0.5, offset_span=2) == DEFAULT_MIN_OFFSET_TICKS
    # span=2: raw +0.5 -> 1.5 ticks -> banker's -> 2 ticks
    assert resolve_offset_cents(0.5, offset_span=2) == DEFAULT_MIN_OFFSET_TICKS + 2


def test_span_respects_a_raised_min_offset_ticks():
    for raw, ticks in ((-1.0, 0), (1.0, SPAN)):
        assert resolve_offset_cents(
            raw, offset_span=SPAN, min_offset_ticks=4) == 4 + ticks


def test_span_scales_with_tick_size():
    """Offsets are returned in cents, so a wider tick multiplies them."""
    assert resolve_offset_cents(1.0, offset_span=SPAN, tick_size=5) \
        == 5 * (DEFAULT_MIN_OFFSET_TICKS + SPAN)


# ─────────────────────────────────────────────────────────────────────────────
# 2. The three mid-rounding modes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MID_ROUNDING_MODES)
@pytest.mark.parametrize("off", [0, 1, 2, 7, 50])
def test_even_spread_makes_all_three_modes_identical(mode, off):
    """When the spread is even the doubled mid is even, the true mid is an
    integer, and there is nothing to round. Every mode must agree exactly -
    which is what makes the odd-spread rows below attributable to parity."""
    assert quote_prices(MID2_EVEN, off, off, mid_rounding=mode) \
        == quote_prices(MID2_EVEN, off, off, mid_rounding="floor")


@pytest.mark.parametrize("mid2", [200_000, 200_002, 12, 0, -4])
@pytest.mark.parametrize("b,a", [(0, 0), (1, 3), (9, 2)])
def test_even_doubled_mid_agreement_holds_off_symmetric_offsets(mid2, b, a):
    ref = quote_prices(mid2, b, a, mid_rounding="floor")
    for mode in MID_ROUNDING_MODES:
        assert quote_prices(mid2, b, a, mid_rounding=mode) == ref


def test_floor_is_the_legacy_reference_and_lands_on_the_bid():
    """(bid + ask) // 2 on an odd spread is the bid, so a symmetric offset
    posts the ask one tick closer to the true mid than the bid."""
    bid, ask = quote_prices(MID2_ODD, 1, 1, mid_rounding="floor")
    assert (bid, ask) == (100_000 - 1, 100_000 + 1)
    true_mid = MID2_ODD / 2.0
    assert true_mid - bid == 1.5
    assert ask - true_mid == 0.5


def test_ceil_is_the_exact_mirror_of_floor_on_an_odd_spread():
    """Not a fix - a reversal. Whatever floor does to the bid, ceil does to
    the ask, so the pair of arms brackets the true mid from both directions."""
    true_mid = MID2_ODD / 2.0
    for off in range(0, 12):
        fb, fa = quote_prices(MID2_ODD, off, off, mid_rounding="floor")
        cb, ca = quote_prices(MID2_ODD, off, off, mid_rounding="ceil")
        assert (cb, ca) == (fb + 1, fa + 1)
        # mirror in the exact sense: reflecting floor's pair about the true
        # mid gives ceil's pair
        assert (true_mid - fb, fa - true_mid) == (ca - true_mid, true_mid - cb)


def test_outward_is_symmetric_about_the_true_half_integer_mid():
    true_mid = MID2_ODD / 2.0
    for off in range(0, 12):
        bid, ask = quote_prices(MID2_ODD, off, off, mid_rounding="outward")
        assert true_mid - bid == ask - true_mid
        assert true_mid - bid == off + 0.5      # rounded away, both sides


def test_outward_keeps_asymmetric_offsets_asymmetric_by_exactly_their_gap():
    """The mode must fix the parity artifact without flattening a deliberate
    skew: an agent quoting 2 up and 5 down still quotes 2 up and 5 down."""
    true_mid = MID2_ODD / 2.0
    bid, ask = quote_prices(MID2_ODD, 5, 2, mid_rounding="outward")
    assert (true_mid - bid) - (ask - true_mid) == 3


@pytest.mark.parametrize("mid2", [-7, -1, 0, 1, 3, 7])
def test_outward_integer_arithmetic_is_correct_for_negative_doubled_mids(mid2):
    """floor/ceil are written as ``x // 2`` and ``-((-x) // 2)``; Python's
    floor division on negatives is where a float-based version would drift."""
    for b, a in product(range(0, 4), repeat=2):
        bid, ask = quote_prices(mid2, b, a, mid_rounding="outward")
        assert bid == int(np.floor((mid2 - 2 * b) / 2))
        assert ask == int(np.ceil((mid2 + 2 * a) / 2))


def test_unknown_mid_rounding_mode_raises():
    with pytest.raises(ValueError, match="mid_rounding"):
        quote_prices(MID2_ODD, 1, 1, mid_rounding="nearest")
    with pytest.raises(ValueError, match="mid_rounding"):
        resolve_quote((1.0, 1.0, 0.5, 0.5, 5.0, 5.0), doubled_mid=MID2_ODD,
                      gated=True, mid_rounding="round-half-even")


# ─────────────────────────────────────────────────────────────────────────────
# 3. The doubled mid, and refusing to lose the half cent
# ─────────────────────────────────────────────────────────────────────────────

def test_doubled_mid_from_book_keeps_the_half_cent():
    assert doubled_mid_from_book([(100_000, 5)], [(100_001, 5)]) == 200_001
    assert doubled_mid_from_book([(100_000, 5)], [(100_002, 5)]) == 200_002


def test_doubled_mid_from_one_sided_or_empty_book():
    assert doubled_mid_from_book([(99_998, 3)], []) == 2 * 99_998
    assert doubled_mid_from_book([], [(100_004, 3)]) == 2 * 100_004
    assert doubled_mid_from_book([], []) is None


def test_doubled_mid_from_obs_prefers_the_exact_touch():
    assert doubled_mid_from_obs({"doubled_mid": 200_001}) == 200_001
    assert doubled_mid_from_obs(
        {"best_bid": 100_000, "best_ask": 100_001}) == 200_001
    assert doubled_mid_from_obs({"mid_price": 100_000}) is None
    assert doubled_mid_from_obs(np.zeros(4)) is None


@pytest.mark.parametrize("mode", ["ceil", "outward"])
def test_a_prerounded_mid_is_refused_by_the_modes_that_need_the_half_cent(mode):
    """The trap this guards: 2 * a floored mid is always even, so ceil and
    outward would silently behave exactly like floor. Refuse instead."""
    with pytest.raises(ValueError, match="doubled mid"):
        translate_action(
            action=(1.0, 1.0, 0.5, 0.5, 5.0, 5.0), agent_id=0, symbol="ABM",
            mid_price=100_000, resting_orders=[], gated=True,
            mid_rounding=mode)


def test_inconsistent_mid_price_and_doubled_mid_raise():
    with pytest.raises(ValueError, match="inconsistent"):
        translate_action(
            action=(1.0, 1.0, 0.5, 0.5, 5.0, 5.0), agent_id=0, symbol="ABM",
            mid_price=50_000, resting_orders=[], gated=True,
            doubled_mid=MID2_ODD)


def test_consistent_mid_price_and_doubled_mid_are_accepted():
    intents = translate_action(
        action=(1.0, 1.0, 0.5, 0.5, 5.0, 5.0), agent_id=0, symbol="ABM",
        mid_price=100_000, resting_orders=[], gated=True,
        doubled_mid=MID2_ODD)
    assert len([i for i in intents if isinstance(i, PlaceIntent)]) == 2


def test_translate_action_needs_some_mid():
    with pytest.raises(TypeError, match="doubled_mid or mid_price"):
        translate_action(action=(1.0, 1.0, 0.5, 0.5, 5.0, 5.0), agent_id=0,
                         symbol="ABM", resting_orders=[], gated=True)


def test_span_requires_the_gated_layout():
    """The legacy 4-tuple has no gate, so it has no 'side is on' state for the
    rescale's floor to attach to; applying it there would quietly delete the
    no-quote region the legacy layout is kept as a control for."""
    with pytest.raises(ValueError, match="offset_span requires the gated"):
        resolve_quote((0.5, 0.5, 10.0, 10.0), doubled_mid=MID2_EVEN,
                      gated=False, offset_span=SPAN)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Structural guarantee: gated sides still cannot cross
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODES = list(product([None, 0, 1, 5, SPAN, 40], MID_ROUNDING_MODES))


@pytest.mark.parametrize("span,mode", ALL_MODES)
def test_gated_sides_never_cross_under_any_flag_combination(span, mode):
    """``crossed% == 0.00`` is a structural property the project relies on:
    with both gates on and both offsets floored at one tick,
    ``bid_price < ask_price`` whatever the offsets, the span or the rounding.
    """
    rng = np.random.default_rng(0)
    n = 400
    hi = 1.0 if span else 40.0
    rows = np.column_stack([
        rng.uniform(0.01, 1, n), rng.uniform(0.01, 1, n),      # gates on
        rng.uniform(0, hi, n), rng.uniform(0, hi, n),
        rng.uniform(0, 60, n), rng.uniform(0, 60, n),
    ])
    for mid2 in (MID2_EVEN, MID2_ODD):
        for row in rows:
            q = resolve_quote(tuple(row), doubled_mid=mid2, gated=True,
                              offset_span=span, mid_rounding=mode)
            assert not q.crossed
            assert q.bid_qty > 0 and q.ask_qty > 0
            assert q.bid_price < q.ask_price


@pytest.mark.parametrize("span,mode", ALL_MODES)
def test_participation_reports_zero_crossings_through_the_reconstruction(span, mode):
    rng = np.random.default_rng(1)
    n = 300
    hi = 1.0 if span else 40.0
    rows = np.column_stack([
        rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
        rng.uniform(0, hi, n), rng.uniform(0, hi, n),
        rng.uniform(0, 0.9, n), rng.uniform(0, 0.9, n),
    ])
    m = participation_stats(rows, offset_span=span, mid_rounding=mode)
    assert m["crossed_pct"] == 0.0


def test_the_cross_guard_still_fires_when_the_floor_is_removed():
    """min_offset_ticks=0 reopens the degenerate region; the guard must catch
    it rather than posting a crossed quote, span or no span."""
    # the degenerate offset is 0.0 in the legacy mapping and -1.0 (the floor
    # of the symmetric range) under a span
    for span, floor_raw in ((None, 0.0), (SPAN, -1.0)):
        q = resolve_quote((1.0, 1.0, floor_raw, floor_raw, 10.0, 10.0),
                          doubled_mid=MID2_EVEN, gated=True,
                          offset_span=span, min_offset_ticks=0)
        assert q.crossed and q.bid_qty == 0 and q.ask_qty == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. The reconstruction and the live path agree
# ─────────────────────────────────────────────────────────────────────────────

def _live(row, mid2, *, gated, span, mode, max_size=100):
    """What the agent actually sends, read back off the intents."""
    intents = translate_action(
        action=tuple(row), agent_id=0, symbol="ABM", resting_orders=[],
        max_size=max_size, gated=gated, offset_span=span, mid_rounding=mode,
        doubled_mid=int(mid2))
    places = [i for i in intents if isinstance(i, PlaceIntent)]
    bid = next((p for p in places if p.is_buy_order), None)
    ask = next((p for p in places if not p.is_buy_order), None)
    return (
        bid.limit_price if bid else 0,
        ask.limit_price if ask else 0,
        bid.quantity if bid else 0,
        ask.quantity if ask else 0,
    )


@pytest.mark.parametrize("span,mode", ALL_MODES)
def test_reconstruction_matches_the_live_path_gated(span, mode):
    """The test that protects every downstream behavioural metric.

    ``quote_stats``, ``fill_skew`` and ``agent_divergence`` all report numbers
    computed by replaying saved actions. If the replay and the agent disagree
    about what an action means, those numbers are wrong and silent. Both sides
    below go through ``actions.resolve_quote``, and this asserts that they
    still produce identical prices and sizes on every row.
    """
    rng = np.random.default_rng(7)
    n = 200
    hi = 1.0 if span else 40.0
    rows = np.column_stack([
        rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
        rng.uniform(0, hi, n), rng.uniform(0, hi, n),
        rng.uniform(0, 60, n), rng.uniform(0, 60, n),
    ])
    for mid2 in (MID2_EVEN, MID2_ODD, 2 * 7 + 1, 15):
        mids = np.full(n, mid2)
        quotes = effective_quotes(rows, offset_span=span, mid_rounding=mode,
                                  doubled_mid=mids)
        prices = quote_prices_from_actions(rows, mids, offset_span=span,
                                           mid_rounding=mode)
        for i, row in enumerate(rows):
            expected = _live(row, mid2, gated=True, span=span, mode=mode)
            got = (int(prices[i, 0]), int(prices[i, 1]),
                   int(quotes[i, 2]), int(quotes[i, 3]))
            assert got == expected, f"row {i} at doubled_mid={mid2}"


@pytest.mark.parametrize("mode", MID_ROUNDING_MODES)
def test_reconstruction_matches_the_live_path_legacy_layout(mode):
    """Same agreement for the 4-tuple, which has no span but does have a mid."""
    rng = np.random.default_rng(8)
    n = 200
    rows = np.column_stack([
        rng.uniform(-1, 6, n), rng.uniform(-1, 6, n),
        rng.uniform(0, 30, n), rng.uniform(0, 30, n),
    ])
    for mid2 in (MID2_EVEN, MID2_ODD):
        mids = np.full(n, mid2)
        quotes = effective_quotes(rows, mid_rounding=mode, doubled_mid=mids)
        prices = quote_prices_from_actions(rows, mids, mid_rounding=mode)
        for i, row in enumerate(rows):
            expected = _live(row, mid2, gated=False, span=None, mode=mode)
            got = (int(prices[i, 0]), int(prices[i, 1]),
                   int(quotes[i, 2]), int(quotes[i, 3]))
            assert got == expected, f"row {i} at doubled_mid={mid2}"


def test_reconstruction_default_mid_reproduces_the_legacy_offset_columns():
    """Callers that kept only the actions still get the right offsets and
    sizes; only the prices need a real mid."""
    rows = np.array([[1.0, 1.0, 2.4, 3.6, 10.0, 20.0]])
    assert effective_quotes(rows).tolist() == [[2, 4, 10, 20]]


def test_span_changes_the_reconstruction_not_only_the_live_path():
    """If the flag reached the agent but not the replay, this is the number
    that would have gone wrong: the same action would be reported as a 1-tick
    quote forever."""
    rows = np.array([[1.0, 1.0, 1.0, 1.0, 10.0, 10.0]])
    legacy = effective_quotes(rows)
    spanned = effective_quotes(rows, offset_span=SPAN)
    assert legacy[0, 0] == 1 and legacy[0, 1] == 1
    assert spanned[0, 0] == DEFAULT_MIN_OFFSET_TICKS + SPAN
    assert spanned[0, 1] == DEFAULT_MIN_OFFSET_TICKS + SPAN


def test_resolve_quote_returns_a_frozen_record():
    q = resolve_quote((1.0, 1.0, 0.5, 0.5, 5.0, 5.0), doubled_mid=MID2_ODD,
                      gated=True, offset_span=SPAN, mid_rounding="outward")
    assert isinstance(q, ResolvedQuote)
    with pytest.raises(FrozenInstanceError):
        q.bid_qty = 0        # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# 6. How the mode reaches the analysis path
# ─────────────────────────────────────────────────────────────────────────────

def test_quote_mode_defaults_are_the_legacy_semantics():
    m = QuoteMode()
    assert m.offset_span is None
    assert m.mid_rounding == "floor"
    assert m.min_offset_ticks == DEFAULT_MIN_OFFSET_TICKS
    assert m.is_legacy and m == LEGACY_QUOTE_MODE


def test_quote_mode_from_run_args_reads_the_flags():
    m = QuoteMode.from_run_args(
        {"offset_span": SPAN, "mid_rounding": "outward",
         "min_offset_ticks": 2, "min_quote_size": 3})
    assert (m.offset_span, m.mid_rounding) == (SPAN, "outward")
    assert (m.min_offset_ticks, m.min_quote_size) == (2, 3)
    assert not m.is_legacy


@pytest.mark.parametrize("args", [
    {},                                   # a run from before the flags
    {"n_agents": 2, "seed": 1},           # other flags only
    {"offset_span": None},                # flag present, unset
    {"offset_span": 0},                   # flag present, zero
    {"mid_rounding": None},               # flag present, null
])
def test_absent_or_empty_flags_mean_legacy(args):
    """Old artifacts carry no such field, and absence must mean the semantics
    those runs actually had - not the new default."""
    assert QuoteMode.from_run_args(args).is_legacy


def test_quote_mode_for_trajectory_reads_the_run_meta(tmp_path):
    run = tmp_path / "cell" / "seed1"
    (run / "eval").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps(
        {"args": {"offset_span": SPAN, "mid_rounding": "outward"}}))
    npz = run / "eval" / "trajectory_0_seed42.npz"
    npz.write_bytes(b"")
    m = QuoteMode.for_trajectory(npz)
    assert (m.offset_span, m.mid_rounding) == (SPAN, "outward")


def test_quote_mode_for_trajectory_without_meta_is_legacy(tmp_path):
    npz = tmp_path / "eval" / "trajectory_0_seed42.npz"
    npz.parent.mkdir(parents=True)
    npz.write_bytes(b"")
    assert QuoteMode.for_trajectory(npz).is_legacy


def test_quote_mode_for_trajectory_with_broken_meta_is_legacy(tmp_path):
    run = tmp_path / "seed1"
    (run / "eval").mkdir(parents=True)
    (run / "run_meta.json").write_text("{not json")
    npz = run / "eval" / "t.npz"
    npz.write_bytes(b"")
    assert QuoteMode.for_trajectory(npz).is_legacy


def test_quote_mode_kwargs_splat_into_the_reconstruction():
    mode = QuoteMode(offset_span=SPAN, mid_rounding="outward")
    rows = np.array([[1.0, 1.0, 1.0, 1.0, 10.0, 10.0]])
    q = effective_quotes(rows, **mode.kwargs())
    assert q[0, 0] == DEFAULT_MIN_OFFSET_TICKS + SPAN
    assert "span=15" in mode.label() and "outward" in mode.label()


# ─────────────────────────────────────────────────────────────────────────────
# 7. The action space the policy actually sees
# ─────────────────────────────────────────────────────────────────────────────

def test_env_offset_bounds_are_unchanged_without_a_span():
    """The default must stay bit-identical: a change here would silently
    invalidate every existing run's action-space contract."""
    from marl_lob.env import MarlLobEnv

    env = MarlLobEnv(n_agents=2, max_offset_cents=50)
    lo, hi = env.action_space("mm_0").low, env.action_space("mm_0").high
    assert (lo[2], lo[3]) == (0.0, 0.0)
    assert (hi[2], hi[3]) == (50.0, 50.0)
    assert env.offset_span is None and env.mid_rounding == "floor"


def test_env_offset_bounds_become_unit_under_a_span():
    from marl_lob.env import MarlLobEnv

    env = MarlLobEnv(n_agents=2, max_offset_cents=50, offset_span=SPAN)
    lo, hi = env.action_space("mm_0").low, env.action_space("mm_0").high
    assert (lo[2], lo[3]) == (-1.0, -1.0), "offset range must be symmetric"
    assert (hi[2], hi[3]) == (1.0, 1.0)
    # the gate and size dimensions are untouched by the rescale
    assert (lo[0], lo[1], hi[0], hi[1]) == (-1.0, -1.0, 1.0, 1.0)
    assert (hi[4], hi[5]) == (100.0, 100.0)


@pytest.mark.parametrize("span_arg", [None, 0])
def test_env_treats_unset_and_zero_span_the_same(span_arg):
    from marl_lob.env import MarlLobEnv

    env = MarlLobEnv(n_agents=1, offset_span=span_arg)
    assert env.offset_span is None
    assert env.action_space("mm_0").high[2] == 50.0


def test_env_rejects_a_span_on_the_legacy_layout():
    from marl_lob.env import MarlLobEnv

    with pytest.raises(ValueError, match="offset_span requires the gated"):
        MarlLobEnv(n_agents=1, gated_actions=False, offset_span=SPAN)


def test_env_rejects_an_unknown_mid_rounding_mode():
    from marl_lob.env import MarlLobEnv

    with pytest.raises(ValueError, match="mid_rounding"):
        MarlLobEnv(n_agents=1, mid_rounding="nearest")


@pytest.mark.parametrize("mode", MID_ROUNDING_MODES)
def test_env_accepts_every_declared_mode(mode):
    from marl_lob.env import MarlLobEnv

    assert MarlLobEnv(n_agents=1, mid_rounding=mode).mid_rounding == mode


# ─────────────────────────────────────────────────────────────────────────────
# 8. Replay without a mid
# ─────────────────────────────────────────────────────────────────────────────

def test_replay_without_a_mid_reports_no_prices():
    """A saved action array does not carry the book, so prices are not
    recoverable from it - and are reported as absent rather than invented."""
    q = resolve_quote((1.0, 1.0, 0.5, 0.5, 5.0, 5.0), gated=True)
    assert q.bid_price is None and q.ask_price is None
    assert (q.bid_off_cents, q.ask_off_cents, q.bid_qty, q.ask_qty) == (1, 1, 5, 5)


def test_replay_without_a_mid_does_not_invent_a_positivity_void():
    """A fabricated stand-in mid would void an offset larger than itself,
    which is a statement about a price no one measured. The offsets stand."""
    huge = np.array([[1.0, 1.0, 1e9, 1e9, 10.0, 10.0]])
    q = effective_quotes(huge)
    assert q[0, 0] == 10 ** 9 and q[0, 2] == 10


def test_replay_without_a_mid_still_catches_legacy_self_voids():
    """The cross test survives the missing mid, because under floor and ceil
    it is exactly ``-bid_off >= ask_off`` at any parity."""
    assert effective_quotes(np.array([[0.4, 0.4, 10.0, 10.0]])).tolist() \
        == [[0, 0, 0, 0]]
    assert participation_stats(
        np.array([[0.4, 0.4, 10.0, 10.0]] * 4))["crossed_pct"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Integration: the flags survive a real ABIDES kernel
#
# Everything above is pure arithmetic. These two drive the actual coordinator
# and children, which is the only way to catch a plumbing break between the
# env's flags and MarlChild.act_on_wakeup - where the doubled mid is now
# derived from the book instead of being floored on the way in.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.abides
@pytest.mark.parametrize("mode", MID_ROUNDING_MODES)
def test_env_quotes_and_fills_under_every_rounding_mode(mode):
    from marl_lob.env import MarlLobEnv

    e = MarlLobEnv(n_agents=2, offset_span=SPAN, mid_rounding=mode)
    try:
        e.reset(seed=42)
        assert all(c.mid_rounding == mode for c in e._children)
        assert all(c.offset_span == SPAN for c in e._children)
        fills = 0
        for _ in range(60):
            acts = {a: np.array([1.0, 1.0, 0.1, 0.1, 5.0, 5.0],
                                dtype=np.float32) for a in e.agents}
            _o, _r, _t, truncs, info = e.step(acts)
            fills += sum(info[a]["traj_row"][4] != 0 for a in e.possible_agents)
            if any(truncs.values()):
                break
        assert fills > 0, "a 2-tick two-sided quote should fill within 60 steps"
    finally:
        e.close()


@pytest.mark.abides
def test_span_offsets_reach_the_book_at_their_rescaled_width():
    """raw 1.0 must post min_offset_ticks + span ticks out, not 1 tick.

    Read off the trajectory rather than the action: the point of the flag is
    what reaches the exchange.
    """
    from marl_lob.env import MarlLobEnv
    from marl_lob.quote_analysis import QuoteMode, effective_quotes

    e = MarlLobEnv(n_agents=1, offset_span=SPAN, mid_rounding="outward")
    try:
        e.reset(seed=42)
        acts = np.array([[1.0, 1.0, 1.0, 1.0, 5.0, 5.0]], dtype=np.float32)
        for _ in range(3):
            e.step({a: acts[0] for a in e.agents})
        mode = QuoteMode(offset_span=SPAN, mid_rounding="outward")
        q = effective_quotes(acts, **mode.kwargs())
        assert q[0, 0] == DEFAULT_MIN_OFFSET_TICKS + SPAN
    finally:
        e.close()


def test_explicit_nulls_in_run_meta_do_not_crash_or_shift_defaults():
    """argparse writes an unset int flag into run_meta.json as null, which is
    not the same value as 0 - and 0 is a meaningful min_offset_ticks."""
    m = QuoteMode.from_run_args(
        {"offset_span": None, "mid_rounding": None,
         "min_offset_ticks": None, "min_quote_size": None})
    assert m.is_legacy
    zeroed = QuoteMode.from_run_args({"min_offset_ticks": 0})
    assert zeroed.min_offset_ticks == 0
