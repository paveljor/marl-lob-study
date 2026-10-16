"""Replay saved action vectors through the live quantisation.

Delta-equity says how much money a policy made; it does not say whether the
policy was making markets, or whether two agents were doing anything
different from each other. Both questions are answered by pushing the saved
``actions`` array through the same rounding, flooring and voiding that
``actions.translate_action`` applies before orders reach ABIDES.

An action vector is not a quote. Offsets are continuous cents rounded to whole
ticks and sizes are rounded to integers, so a policy emitting sub-tick offsets
and sub-unit sizes posts nothing at all while still producing a full-looking
action array.

**Two layouts coexist on disk.** Trajectories written before the gated action
space have 4 columns; those written after have 6. Every reader here dispatches
on width, because a 6-wide array read as the legacy layout reports the gates as
offsets - wrong, and silently so.

**And two offset mappings, and three mid-rounding modes.** Layout is
recoverable from the array itself; these are not. They are recorded in the
run's ``run_meta.json`` and read back by ``QuoteMode.for_trajectory``, whose
defaults are the legacy semantics - so an artifact written before the flags
existed is replayed the way it actually ran. Every function here delegates the
arithmetic to ``actions.resolve_quote``, the same function the live agent
calls; the point is that there is no second copy of it to drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import (
    DEFAULT_MID_ROUNDING,
    DEFAULT_MIN_OFFSET_TICKS,
    DEFAULT_MIN_QUOTE_SIZE,
    DEFAULT_OFFSET_SPAN,
    TICK_SIZE_CENTS,
    ResolvedQuote,
    resolve_quote,
)

LEGACY_ACTION_NAMES = ("bid_offset", "ask_offset", "bid_size", "ask_size")
GATED_ACTION_NAMES = ("bid_gate", "ask_gate", "bid_offset", "ask_offset",
                      "bid_size", "ask_size")
QUOTE_NAMES = LEGACY_ACTION_NAMES

@dataclass(frozen=True)
class QuoteMode:
    """The action semantics one run was trained and evaluated under.

    Reconstructing quotes from a saved ``actions`` array requires knowing
    which mapping produced them. Old artifacts have no such record, and every
    field here defaults to the pre-2026-09-05 behaviour, so absence reads as
    legacy - which is what those runs actually were.
    """
    offset_span: int | None = DEFAULT_OFFSET_SPAN
    mid_rounding: str = DEFAULT_MID_ROUNDING
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE

    def kwargs(self) -> dict:
        """Splat into ``effective_quotes`` / ``participation_stats``."""
        return {
            "offset_span": self.offset_span,
            "mid_rounding": self.mid_rounding,
            "min_offset_ticks": self.min_offset_ticks,
            "min_quote_size": self.min_quote_size,
        }

    @property
    def is_legacy(self) -> bool:
        return self == QuoteMode()

    def label(self) -> str:
        """Short human tag for tables - the mode must never be invisible."""
        span = "off" if not self.offset_span else str(self.offset_span)
        return f"span={span},mid={self.mid_rounding}"

    @classmethod
    def from_run_args(cls, args: dict | None) -> QuoteMode:
        """Read the mode out of ``run_meta.json``'s ``args`` block.

        ``offset_span`` of 0 and a missing key mean the same thing: the legacy
        mapping. A missing ``mid_rounding`` means ``floor``, because that is
        what every run written before the flag existed did.
        """
        args = args or {}

        def _int(key: str, default: int) -> int:
            # An explicit null is not the same as a zero, so this cannot be
            # written with ``or``: min_offset_ticks=0 is a meaningful setting
            # (it reopens the degenerate region) and must survive the read.
            value = args.get(key)
            return default if value is None else int(value)

        span = args.get("offset_span")
        return cls(
            offset_span=int(span) if span else None,
            mid_rounding=args.get("mid_rounding") or DEFAULT_MID_ROUNDING,
            min_offset_ticks=_int("min_offset_ticks", DEFAULT_MIN_OFFSET_TICKS),
            min_quote_size=_int("min_quote_size", DEFAULT_MIN_QUOTE_SIZE),
        )

    @classmethod
    def from_run_meta(cls, meta: dict) -> QuoteMode:
        return cls.from_run_args(meta.get("args"))

    @classmethod
    def for_trajectory(cls, npz_path, max_up: int = 4) -> QuoteMode:
        """Find the ``run_meta.json`` covering one eval trajectory.

        Layout is ``<run_dir>/eval/<name>.npz`` both in ``runs/`` and in the
        archived ``results/artifacts/`` copy, so the meta is one level up;
        the walk is kept general so a re-arranged tree still resolves. No meta
        found means an artifact from before the flags existed, which is the
        legacy mode.
        """
        here = Path(npz_path).resolve().parent
        for _ in range(max_up + 1):
            candidate = here / "run_meta.json"
            if candidate.is_file():
                try:
                    return cls.from_run_meta(json.loads(candidate.read_text()))
                except (ValueError, OSError):
                    return cls()
            if here.parent == here:
                break
            here = here.parent
        return cls()


LEGACY_QUOTE_MODE = QuoteMode()


def action_names(width: int) -> tuple[str, ...]:
    """Column labels for an action array, chosen by its width."""
    if width == 6:
        return GATED_ACTION_NAMES
    if width == 4:
        return LEGACY_ACTION_NAMES
    raise ValueError(f"unrecognised action width {width}; expected 4 or 6")


def is_gated(actions: np.ndarray) -> bool:
    """True when the array uses the 6-column gated layout."""
    actions = np.asarray(actions)
    width = actions.shape[1] if actions.ndim == 2 else 0
    action_names(width)
    return width == 6


def resolve_rows(
    actions: np.ndarray,
    max_size: int = 100,
    *,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
    doubled_mid: np.ndarray | int | None = None,
) -> list[ResolvedQuote]:
    """Replay every saved action row through ``actions.resolve_quote``.

    This is the reconstruction side of the one shared path. It calls the same
    function the live agent calls, so an action-semantics change cannot reach
    the book without also reaching every metric computed from these rows.

    ``doubled_mid`` is ``best_bid + best_ask`` per row, and is what makes the
    reconstructed prices exact. Omitting it - the usual case, since a saved
    action array is often all there is - leaves the prices as None and the
    offsets and quantities exact; see ``actions.resolve_quote`` for how the
    two mid-dependent voids degrade.
    """
    actions = np.asarray(actions, dtype=float)
    gated = is_gated(actions)
    if doubled_mid is None:
        mids: list[int | None] = [None] * len(actions)
    else:
        mids = [int(m) for m in np.broadcast_to(
            np.asarray(doubled_mid, dtype=np.int64), (len(actions),))]
    return [
        resolve_quote(
            tuple(row),
            doubled_mid=mid,
            gated=gated,
            max_size=max_size,
            tick_size=TICK_SIZE_CENTS,
            min_offset_ticks=min_offset_ticks,
            min_quote_size=min_quote_size,
            offset_span=offset_span,
            mid_rounding=mid_rounding,
        )
        for row, mid in zip(actions, mids, strict=False)
    ]


def effective_quotes(
    actions: np.ndarray,
    max_size: int = 100,
    *,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
    doubled_mid: np.ndarray | int | None = None,
) -> np.ndarray:
    """Quantise an action array into the quotes the exchange would see.

    Returns an (n, 4) int array of ``(bid_off, ask_off, bid_qty, ask_qty)``.
    None of those four depend on the mid, so ``doubled_mid`` is optional here;
    use ``quote_prices_from_actions`` when the posted prices themselves are
    the quantity of interest.
    """
    rows = resolve_rows(
        actions, max_size, min_offset_ticks=min_offset_ticks,
        min_quote_size=min_quote_size, offset_span=offset_span,
        mid_rounding=mid_rounding, doubled_mid=doubled_mid)
    out = np.zeros((len(rows), 4), dtype=int)
    for i, q in enumerate(rows):
        out[i] = (q.bid_off_cents, q.ask_off_cents, q.bid_qty, q.ask_qty)
    return out


def quote_prices_from_actions(
    actions: np.ndarray,
    doubled_mid: np.ndarray | int,
    max_size: int = 100,
    *,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
) -> np.ndarray:
    """The prices the agent actually posted: an (n, 2) int array.

    Zero on a side that posted nothing. Reconstructing these as
    ``mid_floor -/+ offset`` at the call site is only correct under
    ``mid_rounding="floor"``; under ``outward`` the ask sits a cent higher
    whenever the spread is odd, which is the whole point of the mode, so
    every caller that wants prices must come through here.
    """
    rows = resolve_rows(
        actions, max_size, min_offset_ticks=min_offset_ticks,
        min_quote_size=min_quote_size, offset_span=offset_span,
        mid_rounding=mid_rounding, doubled_mid=doubled_mid)
    out = np.zeros((len(rows), 2), dtype=np.int64)
    for i, q in enumerate(rows):
        out[i] = (q.bid_price or 0, q.ask_price or 0)
    return out


def participation_stats(
    actions: np.ndarray,
    max_size: int = 100,
    *,
    min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
    min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
    offset_span: int | None = DEFAULT_OFFSET_SPAN,
    mid_rounding: str = DEFAULT_MID_ROUNDING,
    doubled_mid: np.ndarray | int | None = None,
) -> dict:
    """How often did this policy actually put a market on the book?

    ``two_sided_pct`` is the headline: the share of steps carrying a genuine
    two-sided quote. ``crossed_pct`` is the legacy failure mode - both sides
    voided because the rounded offsets crossed - and is 0 by construction under
    gating with ``min_offset_ticks >= 1``.
    """
    rows = resolve_rows(
        actions, max_size, min_offset_ticks=min_offset_ticks,
        min_quote_size=min_quote_size, offset_span=offset_span,
        mid_rounding=mid_rounding, doubled_mid=doubled_mid)
    bid_off = np.array([q.bid_off_cents for q in rows])
    ask_off = np.array([q.ask_off_cents for q in rows])
    bid_qty = np.array([q.bid_qty for q in rows])
    ask_qty = np.array([q.ask_qty for q in rows])
    # A voided pair leaves zero quantity on both sides, so the crossing that
    # caused it is unrecoverable from the result - resolve_quote reports it.
    crossed = np.array([q.crossed for q in rows], dtype=bool)

    raw = np.asarray(actions, dtype=float)
    gated = is_gated(raw)

    bid_ok, ask_ok = bid_qty > 0, ask_qty > 0
    two_sided = bid_ok & ask_ok
    any_quote = bid_ok | ask_ok

    if gated:
        # Under gating, not quoting is a decision the gates express, not an
        # artifact of rounding - so it is read off the gates directly.
        withdrawn = ~((raw[:, 0] > 0) | (raw[:, 1] > 0))
    else:
        withdrawn = ~any_quote & ~crossed

    return {
        "two_sided_pct": float(two_sided.mean() * 100),
        "any_quote_pct": float(any_quote.mean() * 100),
        "crossed_pct": float(crossed.mean() * 100),
        "withdrawn_pct": float(withdrawn.mean() * 100),
        "spread_cents": (float((bid_off + ask_off)[two_sided].mean())
                         if two_sided.any() else 0.0),
        "quote_size": (float((bid_qty + ask_qty)[two_sided].mean() / 2.0)
                       if two_sided.any() else 0.0),
        "gated": gated,
    }
