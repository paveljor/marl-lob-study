"""Smaller RMSC03 variant for fast local runs.

Wraps upstream `abides_markets.configs.rmsc03` with reduced agent counts
that still produce meaningful order-book activity:

  1 Exchange | 1 Adaptive MM | 50 Value | 10 Momentum | 2000 Noise

Noise count is the volume driver — drop below ~1000 and the L1 barely
moves. Values were calibrated on a 1h sim (seed=42) producing ~140
post-dropna L1 events, ~3s wallclock.

Note: upstream rmsc03 imports POVExecutionAgent which is missing from the
current JPMC fork. See README install section for the small patch.

Background order and book logging default to OFF here, unlike upstream
(2026-09-05). rmsc03 turns on `exchange_log_orders`, `log_orders` and
`book_logging` for every background agent, which makes ~2060 agents append a
record per order and the exchange snapshot the book to depth 10 on every
change. **Nothing in this project reads any of it** - observations come from
each agent's market-data subscription (`parsed_mkt_data`) and metrics from the
`traj_row` contract, so it is pure overhead.

Measured on the desktop, same seed and the same action stream both ways:
**1.28x at 300 steps and 1.35x at 1800**, and the gain grows with episode
length as the logs accumulate. Verified passive: observations and inventories
come back bit-identical (max |delta| 0.0), which is the only thing that makes
this safe to change rather than a silent alteration of results.

Pass `log_orders=True` (etc.) through `config_kwargs` to turn them back on for
a run that needs the ABIDES-side logs.
"""
from abides_markets.configs import rmsc03


def build_config(start_time="09:30:00", end_time="10:30:00", seed=1, **kwargs):
    return rmsc03.build_config(
        start_time=start_time,
        end_time=end_time,
        seed=seed,
        num_noise_agents=kwargs.pop("num_noise_agents", 2000),
        num_value_agents=kwargs.pop("num_value_agents", 50),
        num_momentum_agents=kwargs.pop("num_momentum_agents", 10),
        execution_agents=kwargs.pop("execution_agents", False),
        # Off by default - see the module docstring. Overridable per run.
        exchange_log_orders=kwargs.pop("exchange_log_orders", False),
        log_orders=kwargs.pop("log_orders", False),
        book_logging=kwargs.pop("book_logging", False),
        **kwargs,
    )
