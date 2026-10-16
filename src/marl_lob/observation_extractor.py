"""
Task A — Observation Extractor
================================
Converts the ABIDES order book state + agent state into a fixed-size 1D numpy
vector that can be fed directly into a PettingZoo / SB3 policy network.

Observation vector layout (total: 4*K + 4 elements, default K=10 → 44 dims)
─────────────────────────────────────────────────────────────────────────────
Index       Feature
─────────────────────────────────────────────────────────────────────────────
0 .. K-1    Bid prices  (top K levels), normalised by mid-price, descending
K .. 2K-1   Bid sizes   (top K levels), normalised by max_size
2K .. 3K-1  Ask prices  (top K levels), normalised by mid-price, ascending
3K .. 4K-1  Ask sizes   (top K levels), normalised by max_size
4K          Inventory   (signed shares), normalised by max_inventory
4K+1        Cash        (cents),         normalised by starting_cash
4K+2        Spread      (cents / mid),   dimensionless
4K+3        Time-to-close (seconds),     normalised by session_duration_s
4K+4        Order-flow imbalance, last step   (optional, `include_ofi`), tanh
4K+5        Order-flow imbalance, EWMA        (optional, `include_ofi`), tanh
4K+4/6 ..   Agent identity one-hot, width `agent_id_width` (0 = omitted)

Why order-flow imbalance
────────────────────────
Everything else here is a *snapshot*. The agent sees the book as it stands and
has no memory, so it cannot see the book *change* - and adverse selection is a
statement about flow, not about levels. Order-flow imbalance is the single
most-cited predictive feature in microstructure and it was absent, which meant
"the agent does not widen its spread against informed flow" (grid2, P1) was a
claim about the observation rather than about market making: an agent with no
signal for adverse selection cannot condition on it, and would look exactly
like one that has decided not to.

OFI follows Cont, Kukanov & Stoikov (2014), on L1, between consecutive
snapshots. Positive means net buying pressure. Because it is a difference it
needs the previous book, which the caller owns - `extract_obs` stays pure and
takes the already-computed values.
─────────────────────────────────────────────────────────────────────────────

Why the identity one-hot exists
───────────────────────────────
SuperSuit presents the N PettingZoo agents to SB3 as N parallel copies of a
single-agent env, so one policy network is trained on the pooled experience and
every agent evaluates the same weights. Of the 4K+4 features above, only
`inventory` and `cash` can differ between two agents watching the same book -
and measured across 4,350 agent pairs, that was not enough: the agents' quotes
were identical on 97-99.9% of steps once quantised to whole ticks
(`notes/agent_divergence_results.md`). Varying `n_agents` therefore varied the
number of *copies* of one policy, not the amount of competition.

Appending an identity one-hot lets the shared network condition on which agent
it is, so the agents can differentiate without paying for N separate networks.
`agent_id_width` is separate from `n_agents` on purpose: pinning it to a fixed
width (e.g. 4) keeps the observation dimension constant across a grid that
varies `n_agents`, so cells stay comparable.

All values are clipped to [-1, 1] before returning.

Units / conventions inherited from ABIDES
─────────────────────────────────────────
- Prices are in CENTS  (e.g. $1000.00 → 100_000)
- known_bids / known_asks: list of (price_cents, quantity) tuples, best first
- Holdings cash ('CASH' key) is in cents
- NanosecondTime timestamps are int64 nanoseconds since epoch
"""

from __future__ import annotations

import math

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases (mirror ABIDES internals; no import needed here)
# ──────────────────────────────────────────────────────────────────────────────
PriceLevels = list[tuple[int, int]]   # [(price_cents, qty), ...]
NanosecondTime = int

# ──────────────────────────────────────────────────────────────────────────────
# Constants — change here to resize the obs vector (update PettingZoo spaces too)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_K            = 10        # number of price levels each side
DEFAULT_MAX_INVENTORY = 1_000    # shares  — used for normalisation
DEFAULT_MAX_SIZE      = 1_000    # shares  — used for normalising queue sizes
DEFAULT_STARTING_CASH = 10_000_000  # cents ($100 000)
# EWMA smoothing for the OFI channel. At the default 1s wake-up this is a
# half-life of ~13.5s, long enough to carry a persistent flow signal through
# the step-to-step noise and short enough to still be a *current* reading.
DEFAULT_OFI_ALPHA    = 0.05
NS_PER_SECOND         = 1_000_000_000


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

# Two features: last-step OFI and its EWMA.
OFI_FEATURES = 2


def obs_vector_size(
    k: int = DEFAULT_K,
    agent_id_width: int = 0,
    include_ofi: bool = False,
) -> int:
    """Returns the length of the observation vector.

    ``agent_id_width`` is the width of the trailing identity one-hot; 0 omits
    it entirely, which is the pre-2026-09 layout. ``include_ofi`` adds the two
    order-flow features ahead of that one-hot.
    """
    return (4 * k + 4
            + (OFI_FEATURES if include_ofi else 0)
            + max(int(agent_id_width), 0))


def compute_ofi(
    prev_bids: PriceLevels,
    prev_asks: PriceLevels,
    bids: PriceLevels,
    asks: PriceLevels,
) -> float:
    """L1 order-flow imbalance between two book snapshots, in shares.

    Cont, Kukanov & Stoikov (2014):

        e = 1(Pb_n >= Pb_p)*qb_n - 1(Pb_n <= Pb_p)*qb_p
          - 1(Pa_n <= Pa_p)*qa_n + 1(Pa_n >= Pa_p)*qa_p

    Read it side by side. On the bid, a price that rose or held *adds* the new
    queue (buy interest arriving) and a price that fell or held *removes* the
    old one (buy interest leaving); when the price is unchanged both fire and
    the term collapses to the change in queue size. The ask mirrors it with the
    sign flipped, since an ask improving means its price *falls*.

    Positive is net buying pressure. Returns 0.0 whenever either book is
    one-sided or empty in either snapshot - there is no flow to infer.
    """
    if not (prev_bids and prev_asks and bids and asks):
        return 0.0

    pb_p, qb_p = prev_bids[0]
    pa_p, qa_p = prev_asks[0]
    pb_n, qb_n = bids[0]
    pa_n, qa_n = asks[0]

    e = 0.0
    if pb_n >= pb_p:
        e += qb_n
    if pb_n <= pb_p:
        e -= qb_p
    if pa_n <= pa_p:
        e -= qa_n
    if pa_n >= pa_p:
        e += qa_p
    return float(e)


# Reference for the EWMA channel's asinh compression: asinh(50) ~ 4.6, chosen
# so a measured p95 EWMA lands near 0.9 rather than saturating.
OFI_EWMA_ASINH_REF = math.asinh(50.0)


def squash_ofi(ofi: float, max_size: int) -> float:
    """Compress a raw single-step OFI into [-1, 1].

    Measured over a 900-step rollout, |OFI| has median 40, p75 69, p95 140 -
    and then p99 49,240 with a max near 100,000. tanh at a scale of `max_size`
    keeps full resolution across that common range (p50 -> 0.38, p95 -> 0.89)
    while treating the rare enormous prints as one saturating signal. Measured
    saturation: 1.8% of steps, against 12.3% for the linear clip it replaced.
    """
    return float(np.tanh(float(ofi) / max(int(max_size), 1)))


def squash_ofi_ewma(ofi_ewma: float, max_size: int) -> float:
    """Compress the smoothed OFI channel into [-1, 1].

    The EWMA needs a *different* transform from the raw channel, which is not
    obvious and was got wrong first time. The theoretical shrinkage for an
    EWMA of an i.i.d. zero-mean series is sqrt(a/(2-a)) = 0.16, so the smoothed
    channel was initially scaled down by that. Measurement says the opposite:
    the EWMA's median is **3.7x larger** than the raw median, because OFI is
    fat-tailed and autocorrelated - a single 100,000-share print contributes
    ~5,000 to the EWMA and then decays over a ~13.5s half-life, so the smoothed
    series sits persistently high while the median raw step stays small. The
    i.i.d. finite-variance assumption simply does not hold here.

    It also spans a much wider range than the raw channel (p50 148, p90 2,299,
    max 5,350), so tanh either crushes the low end or saturates the high one -
    at its best scale it still saturated 12% of steps. asinh compresses that
    logarithmically instead: p50 -> 0.26, p75 -> 0.65, p90 -> 0.83, p95 -> 0.90.
    """
    x = float(ofi_ewma) / max(int(max_size), 1)
    return float(np.clip(math.asinh(x) / OFI_EWMA_ASINH_REF, -1.0, 1.0))


def extract_obs(
    known_bids: PriceLevels,
    known_asks: PriceLevels,
    inventory: int,
    cash: int,
    current_time: NanosecondTime,
    mkt_open: NanosecondTime,
    mkt_close: NanosecondTime,
    k: int = DEFAULT_K,
    max_inventory: int = DEFAULT_MAX_INVENTORY,
    max_size: int = DEFAULT_MAX_SIZE,
    starting_cash: int = DEFAULT_STARTING_CASH,
    agent_index: int | None = None,
    agent_id_width: int = 0,
    ofi: float | None = None,
    ofi_ewma: float | None = None,
    ofi_alpha: float = DEFAULT_OFI_ALPHA,
) -> np.ndarray:
    """
    Convert ABIDES book state + agent state → 1D float32 numpy array.

    Parameters
    ----------
    known_bids : list of (price_cents, qty) — best bid first, from agent.known_bids[symbol]
    known_asks : list of (price_cents, qty) — best ask first, from agent.known_asks[symbol]
    inventory  : signed share count (positive = long), from agent.holdings[symbol]
    cash       : cash in cents, from agent.holdings['CASH']
    current_time, mkt_open, mkt_close : NanosecondTime ints from ABIDES kernel
    k          : number of price levels to include each side
    max_inventory, max_size, starting_cash : normalisation constants
    agent_index : which agent this observation is for; sets the hot element of
        the identity one-hot. Ignored when ``agent_id_width`` is 0.
    agent_id_width : width of the trailing identity one-hot (0 = omitted)
    ofi, ofi_ewma : order-flow imbalance in shares, last step and smoothed.
        Pass both to include the two OFI features; pass neither to omit them.
        They are a property of the market, not of the agent, so the caller
        computes them once per wake-up via ``compute_ofi`` and shares them.
    ofi_alpha : the EWMA smoothing the caller used. Recorded for provenance;
        the smoothed channel's compression is fixed by measurement, not by
        this value (see squash_ofi_ewma)

    Returns
    -------
    obs : np.ndarray, shape (4*k+4+agent_id_width,), float32, values in [-1, 1]
    """
    # ── mid-price (cents) ────────────────────────────────────────────────────
    best_bid = known_bids[0][0] if known_bids else None
    best_ask = known_asks[0][0] if known_asks else None

    if best_bid is not None and best_ask is not None:
        mid = (best_bid + best_ask) / 2.0
    elif best_bid is not None:
        mid = float(best_bid)
    elif best_ask is not None:
        mid = float(best_ask)
    else:
        mid = 1.0   # fallback — avoids division by zero at open

    # ── book features ────────────────────────────────────────────────────────
    bid_prices = _pad_prices(known_bids, k, mid, side="bid")
    bid_sizes  = _pad_sizes(known_bids, k, max_size)
    ask_prices = _pad_prices(known_asks, k, mid, side="ask")
    ask_sizes  = _pad_sizes(known_asks, k, max_size)

    # ── scalar features ──────────────────────────────────────────────────────
    inv_norm    = np.clip(inventory / max_inventory, -1.0, 1.0)
    cash_norm   = np.clip(cash / starting_cash - 1.0, -1.0, 1.0)  # 0 at start

    spread_norm = 0.0
    if best_bid is not None and best_ask is not None and mid > 0:
        spread_norm = np.clip((best_ask - best_bid) / mid, 0.0, 1.0)

    session_ns  = max(mkt_close - mkt_open, 1)
    elapsed_ns  = max(current_time - mkt_open, 0)
    ttc_norm    = np.clip(1.0 - elapsed_ns / session_ns, 0.0, 1.0)  # 1=open, 0=close

    # ── assemble ─────────────────────────────────────────────────────────────
    parts = [
        bid_prices, bid_sizes,
        ask_prices, ask_sizes,
        [inv_norm, cash_norm, spread_norm, ttc_norm],
    ]

    if ofi is not None or ofi_ewma is not None:
        # The two channels are squashed differently, and deliberately so -
        # see squash_ofi / squash_ofi_ewma. Both are monotone, so ordering
        # survives into the tail.
        parts.append([
            squash_ofi(ofi or 0.0, max_size),
            squash_ofi_ewma(ofi_ewma or 0.0, max_size),
        ])

    width = max(int(agent_id_width), 0)
    if width:
        one_hot = np.zeros(width, dtype=np.float64)
        # Out-of-range indices leave the one-hot all zeros rather than raising:
        # an agent beyond the pinned width is unidentifiable, not invalid.
        if agent_index is not None and 0 <= agent_index < width:
            one_hot[agent_index] = 1.0
        parts.append(one_hot)

    obs = np.concatenate(parts).astype(np.float32)
    return np.clip(obs, -1.0, 1.0)


def describe_obs(
    k: int = DEFAULT_K,
    agent_id_width: int = 0,
    include_ofi: bool = False,
) -> list[str]:
    """
    Returns a list of human-readable feature names in the same order as
    extract_obs(). Useful for debugging and documentation.
    """
    names = []
    for i in range(k):
        names.append(f"bid_price_L{i+1}_norm")
    for i in range(k):
        names.append(f"bid_size_L{i+1}_norm")
    for i in range(k):
        names.append(f"ask_price_L{i+1}_norm")
    for i in range(k):
        names.append(f"ask_size_L{i+1}_norm")
    names += ["inventory_norm", "cash_norm", "spread_norm", "time_to_close_norm"]
    if include_ofi:
        names += ["ofi_norm", "ofi_ewma_norm"]
    names += [f"agent_id_{i}" for i in range(max(int(agent_id_width), 0))]
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pad_prices(
    levels: PriceLevels,
    k: int,
    mid: float,
    side: str,
) -> np.ndarray:
    """
    Extract up to k price levels, normalise relative to mid, pad missing levels.

    For bids: (price - mid) / mid  → negative values (below mid)
    For asks: (price - mid) / mid  → positive values (above mid)
    Missing levels are filled with ±1 (far away from mid).
    """
    out = np.zeros(k, dtype=np.float64)
    fill = -1.0 if side == "bid" else 1.0

    for i in range(k):
        if i < len(levels) and mid > 0:
            price = levels[i][0]
            out[i] = np.clip((price - mid) / mid, -1.0, 1.0)
        else:
            out[i] = fill

    return out


def _pad_sizes(
    levels: PriceLevels,
    k: int,
    max_size: int,
) -> np.ndarray:
    """Extract up to k queue sizes, normalise by max_size, pad missing with 0."""
    out = np.zeros(k, dtype=np.float64)

    for i in range(k):
        if i < len(levels):
            out[i] = np.clip(levels[i][1] / max_size, 0.0, 1.0)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic book: 3 bid levels, 3 ask levels
    mock_bids = [(99_950, 100), (99_900, 200), (99_850, 150)]
    mock_asks = [(100_050, 80),  (100_100, 300), (100_150, 120)]

    mkt_open  = 0
    mkt_close = 3_600 * NS_PER_SECOND     # 1 hour
    current   = 1_800 * NS_PER_SECOND     # halfway through

    obs = extract_obs(
        known_bids=mock_bids,
        known_asks=mock_asks,
        inventory=50,
        cash=9_900_000,
        current_time=current,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
    )

    names = describe_obs()
    print(f"Observation vector — shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"{'Feature':<30} {'Value':>10}")
    print("─" * 42)
    for name, val in zip(names, obs, strict=False):
        print(f"{name:<30} {val:>10.4f}")

    assert obs.shape == (obs_vector_size(),), "Shape mismatch!"
    assert obs.dtype == np.float32, "Wrong dtype!"
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Values out of [-1, 1]!"
    print("\n✓ All assertions passed.")
