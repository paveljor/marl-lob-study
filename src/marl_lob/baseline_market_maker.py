"""
Task F — Constant-Spread Baseline Market Maker
================================================
A native ABIDES TradingAgent that quotes a fixed symmetric spread around the
current mid-price, waking up every `wake_up_freq` nanoseconds.

Strategy
─────────
Every wake-up:
  1. Cancel all resting orders.
  2. Query the current best bid / ask.
  3. On receiving the spread response, compute mid-price.
  4. Post one limit buy  at (mid - half_spread_ticks) for `quote_size` shares.
  5. Post one limit sell at (mid + half_spread_ticks) for `quote_size` shares.

This agent does NOT use PettingZoo — it runs inside an ABIDES simulation
alongside the noise/value/momentum agents from rmsc03_simple.

Parameters
──────────
spread_ticks  : int   — total spread in cents (e.g. 10 = $0.10 spread)
quote_size    : int   — shares to quote on each side
wake_up_freq  : int   — nanoseconds between wakeups (default 10s)
symbol        : str   — ticker to trade (must match ExchangeAgent symbol)

How to add this agent to rmsc03_simple
───────────────────────────────────────
    from abides_markets.agents.baseline_market_maker import ConstantSpreadMarketMaker
    from abides_core.utils import str_to_ns

    mm = ConstantSpreadMarketMaker(
        id=agent_count,
        symbol="ABM",
        starting_cash=10_000_000,
        spread_ticks=10,
        quote_size=100,
        wake_up_freq=str_to_ns("10s"),
        log_orders=True,
    )
    agents.append(mm)
    agent_count += 1
"""

from __future__ import annotations

import logging

import numpy as np
from abides_core import Message, NanosecondTime
from abides_markets.agents.trading_agent import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side

logger = logging.getLogger(__name__)


class ConstantSpreadMarketMaker(TradingAgent):
    """
    Baseline market maker: posts a fixed spread around mid every wake-up.

    This is Task F — the benchmark that our trained MARL agents must beat.
    It is a native ABIDES agent with no PettingZoo or RL dependency.
    """

    def __init__(
        self,
        id: int,
        symbol: str,
        starting_cash: int = 10_000_000,
        name: str | None = None,
        type: str | None = None,
        random_state: np.random.RandomState | None = None,
        spread_ticks: int = 10,          # total spread in cents
        quote_size: int = 100,           # shares per side
        wake_up_freq: NanosecondTime = 10_000_000_000,  # 10 seconds
        log_orders: bool = False,
    ) -> None:
        super().__init__(
            id=id,
            name=name or f"ConstantSpreadMM_{id}",
            type=type or "ConstantSpreadMarketMaker",
            random_state=random_state or np.random.RandomState(),
            starting_cash=starting_cash,
            log_orders=log_orders,
        )
        self.symbol       = symbol
        self.spread_ticks = spread_ticks      # cents, e.g. 10 = $0.10 full spread
        self.quote_size   = quote_size
        self.wake_up_freq = wake_up_freq

        # Internal state machine
        self._awaiting_spread: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # ABIDES lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def wakeup(self, current_time: NanosecondTime) -> None:
        """Called by the ABIDES kernel every wake_up_freq nanoseconds."""
        can_trade = super().wakeup(current_time)

        if not can_trade:
            return

        # Step 1: cancel all resting orders
        self.cancel_all_orders()

        # Step 2: request best bid/ask from exchange
        self._awaiting_spread = True
        self.get_current_spread(self.symbol, depth=1)

        logger.debug(
            f"[{self.name}] Woke up at {current_time}. Cancelled orders, querying spread."
        )

    def receive_message(
        self,
        current_time: NanosecondTime,
        sender_id: int,
        message: Message,
    ) -> None:
        """Handle messages from the exchange."""
        super().receive_message(current_time, sender_id, message)

        if isinstance(message, QuerySpreadResponseMsg) and self._awaiting_spread:
            self._awaiting_spread = False
            self._place_quotes(current_time)

    # ──────────────────────────────────────────────────────────────────────────
    # Core quoting logic
    # ──────────────────────────────────────────────────────────────────────────

    def _place_quotes(self, current_time: NanosecondTime) -> None:
        """Compute mid-price and post symmetric limit orders."""
        bid, _, ask, _ = self.get_known_bid_ask(self.symbol)

        if bid is None or ask is None:
            logger.debug(f"[{self.name}] No spread available at {current_time}, skipping.")
            self._schedule_next_wakeup(current_time)
            return

        mid = int((bid + ask) / 2)
        half = self.spread_ticks // 2

        bid_price = mid - half
        ask_price = mid + half

        # Sanity: ensure bid < ask (always true for spread_ticks >= 2)
        if bid_price >= ask_price:
            ask_price = bid_price + 1

        # Step 4: post bid
        self.place_limit_order(
            symbol=self.symbol,
            quantity=self.quote_size,
            side=Side.BID,
            limit_price=bid_price,
        )

        # Step 5: post ask
        self.place_limit_order(
            symbol=self.symbol,
            quantity=self.quote_size,
            side=Side.ASK,
            limit_price=ask_price,
        )

        logger.debug(
            f"[{self.name}] Posted bid@{bid_price} ask@{ask_price} "
            f"(mid={mid}, spread={self.spread_ticks}¢, size={self.quote_size})"
        )

        self._schedule_next_wakeup(current_time)

    def _schedule_next_wakeup(self, current_time: NanosecondTime) -> None:
        """Schedule the next periodic wakeup."""
        self.set_wakeup(current_time + self.wake_up_freq)

    def get_wake_frequency(self) -> NanosecondTime:
        """Required by TradingAgent base class."""
        return self.wake_up_freq

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def get_pnl(self) -> float:
        """
        Approximate mark-to-market P&L in dollars.

        P&L = (cash - starting_cash) / 100  +  inventory * mid_price / 100

        Returns 0.0 if no market data is available yet.
        """
        cash_pnl = (self.holdings.get("CASH", 0) - self.starting_cash) / 100.0
        inventory = self.holdings.get(self.symbol, 0)

        try:
            bid, _, ask, _ = self.get_known_bid_ask(self.symbol)
            if bid is not None and ask is not None:
                mid_dollars = (bid + ask) / 200.0
                inventory_pnl = inventory * mid_dollars
            else:
                inventory_pnl = 0.0
        except KeyError:
            inventory_pnl = 0.0

        return cash_pnl + inventory_pnl

    def get_inventory(self) -> int:
        """Return signed inventory (shares). Positive = long, negative = short."""
        return self.holdings.get(self.symbol, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build a full RMSC03 config that includes the baseline MM
# ──────────────────────────────────────────────────────────────────────────────

def build_config_with_baseline(
    spread_ticks: int = 10,
    quote_size: int = 100,
    wake_up_freq_s: int = 10,
    **rmsc03_kwargs,
) -> dict:
    """
    Build an rmsc03_simple config that includes the ConstantSpreadMarketMaker.

    All keyword arguments are forwarded to rmsc03_simple.build_config().

    Example
    -------
        from abides_core import abides
        from abides_markets.agents.baseline_market_maker import build_config_with_baseline

        config = build_config_with_baseline(spread_ticks=10, quote_size=100, seed=42)
        end_state = abides.run(config)

        # Retrieve the baseline agent (it's always the last agent added)
        baseline_agent = end_state["agents"][-1]
        print(f"Final P&L: ${baseline_agent.get_pnl():.2f}")
        print(f"Final inventory: {baseline_agent.get_inventory()} shares")
    """
    from abides_markets.configs import rmsc03_simple

    config = rmsc03_simple.build_config(**rmsc03_kwargs)

    baseline = ConstantSpreadMarketMaker(
        id=len(config["agents"]),
        symbol="ABM",
        starting_cash=10_000_000,
        spread_ticks=spread_ticks,
        quote_size=quote_size,
        wake_up_freq=wake_up_freq_s * 1_000_000_000,
        log_orders=True,
    )
    config["agents"].append(baseline)

    # Regenerate latency model to include the new agent
    from abides_markets.utils import generate_latency_model
    config["agent_latency_model"] = generate_latency_model(len(config["agents"]))

    return config


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test — run this file directly
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from abides_core import abides

    print("Building config with ConstantSpreadMarketMaker...")
    config = build_config_with_baseline(
        spread_ticks=10,
        quote_size=100,
        wake_up_freq_s=10,
        seed=42,
        start_time="09:30:00",
        end_time="10:30:00",
        stdout_log_level="WARNING",
    )

    print(f"Total agents: {len(config['agents'])}")
    print("Running simulation (this takes ~1–3 min)...")
    end_state = abides.run(config)

    baseline = end_state["agents"][-1]
    print("\n=== Baseline Market Maker Results ===")
    print(f"  Spread       : {baseline.spread_ticks}¢")
    print(f"  Quote size   : {baseline.quote_size} shares")
    print(f"  Final P&L    : ${baseline.get_pnl():.2f}")
    print(f"  Inventory    : {baseline.get_inventory()} shares")
    print(f"  Executed orders: {len(baseline.executed_orders)}")
    print("\n✓ Baseline agent ran successfully.")
