"""MarlLobEnv — PettingZoo ParallelEnv wrapping ABIDES via MarlCoordinator.

Reset spins up an RMSC03_simple sim with one
coordinator + N children injected; step calls into the kernel one wrapper
tick at a time and unpacks per-agent state into PettingZoo dicts.

Rewards use marked-account changes minus an absolute-inventory penalty.
Exceeding the inventory cap terminates an agent; exchange close truncates
the finite trading session. See ``_compute_reward`` for units and penalties.

Pattern adapted from abides-gym/abides_gym/envs/core_environment.py, with
the multi-agent multiplexing handled by MarlCoordinator.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from abides_core import Kernel
from abides_core.generators import ConstantTimeGenerator
from abides_core.utils import datetime_str_to_ns, str_to_ns, subdict
from abides_markets.utils import config_add_agents
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .actions import (
    DEFAULT_MID_ROUNDING,
    DEFAULT_MIN_OFFSET_TICKS,
    DEFAULT_MIN_QUOTE_SIZE,
    DEFAULT_OFFSET_SPAN,
    MID_ROUNDING_MODES,
)
from .configs import rmsc03_simple
from .latency import apply_latency_edge, describe_edge
from .marl_agents import MarlChild, MarlCoordinator
from .observation_extractor import (
    DEFAULT_K,
    DEFAULT_MAX_INVENTORY,
    DEFAULT_OFI_ALPHA,
    DEFAULT_STARTING_CASH,
    obs_vector_size,
)


class MarlLobEnv(ParallelEnv):
    """N-market-maker PettingZoo environment over ABIDES RMSC03.

    Parameters
    ----------
    n_agents : int
        Number of learning market makers. Each gets its own MarlChild.
    starting_cash : int
        Per-agent starting CASH (cents). Also used for obs normalisation.
    k : int
        LOB depth in the observation vector. The base has 4*K + 4 features;
        optional agent identity and order-flow imbalance append features.
    max_size : int
        Cap on per-side order size; also the action_space upper bound.
    max_offset_cents : int
        Cap on bid/ask offset from mid (cents); the action_space upper bound
        for the two offset dimensions.
    offset_span : int | None
        None (the default) keeps the legacy offset mapping, in which the raw
        action is read as cents. An int rescales the offset dimensions so the
        [-1, 1] action range maps to min_offset_ticks + [0, span] ticks.
        Gated layout only; see actions.resolve_quote for rounding details.
    mid_rounding : {"floor", "ceil", "outward"}
        How the quoting reference is rounded off the touch. "floor" is the
        legacy default and biases the ask toward the true mid on odd spreads;
        "ceil" mirrors that bias; "outward" is symmetric. See actions.py.
    tick_size : int
        ABIDES tick (cents). Defaults to 1.
    historical_date : str
        YYYYMMDD passed through to RMSC03 — fixes the oracle's date.
    start_time, end_time : str
        HH:MM:SS market window.
    config_kwargs : dict | None
        Extra kwargs forwarded to rmsc03_simple.build_config (agent counts,
        etc.). seed is handled via reset(seed=...) and overrides this.
    """

    metadata = {"name": "marl_lob_v0", "is_parallelizable": True}

    def __init__(
        self,
        n_agents: int = 2,
        *,
        starting_cash: int = DEFAULT_STARTING_CASH,
        k: int = DEFAULT_K,
        max_size: int = 100,
        min_size: int = 0,
        max_offset_cents: int = 50,
        tick_size: int = 1,
        max_inventory: int = DEFAULT_MAX_INVENTORY,
        symbol: str = "ABM",
        historical_date: str = "20200603",
        start_time: str = "09:30:00",
        end_time: str = "10:30:00",
        wakeup_interval: str = "1s",
        inventory_penalty: float = 1e-4,
        termination_penalty: float = -1.0,
        gated_actions: bool = True,
        min_offset_ticks: int = DEFAULT_MIN_OFFSET_TICKS,
        min_quote_size: int = DEFAULT_MIN_QUOTE_SIZE,
        offset_span: int | None = DEFAULT_OFFSET_SPAN,
        mid_rounding: str = DEFAULT_MID_ROUNDING,
        agent_id_obs: bool = True,
        agent_id_width: int | None = None,
        include_ofi: bool = False,
        ofi_alpha: float = DEFAULT_OFI_ALPHA,
        latency_edge: float | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.n_agents = n_agents
        self.starting_cash = starting_cash
        self.k = k
        self.max_size = max_size
        self.min_size = min_size
        self.max_offset_cents = max_offset_cents
        self.tick_size = tick_size
        self.max_inventory = max_inventory
        self.gated_actions = gated_actions
        self.min_offset_ticks = min_offset_ticks
        self.min_quote_size = min_quote_size
        # None/0 keeps the legacy offset mapping (raw value read as cents),
        # under which every raw offset in [0, 1.5) posts one tick and ~99% of
        # the policy's offset range is a no-op. A span rescales [0, 1] onto
        # min_offset_ticks + [0, span] ticks. See actions.py.
        self.offset_span = int(offset_span) if offset_span else None
        if self.offset_span and not gated_actions:
            raise ValueError(
                "offset_span requires the gated action layout; the legacy "
                "4-tuple has no gate for the rescale to floor against"
            )
        if mid_rounding not in MID_ROUNDING_MODES:
            raise ValueError(
                f"mid_rounding must be one of {MID_ROUNDING_MODES}, "
                f"got {mid_rounding!r}"
            )
        self.mid_rounding = mid_rounding
        # Pinned rather than derived from n_agents so that a grid varying
        # n_agents keeps a constant observation dimension across its cells.
        self.agent_id_width = (
            (n_agents if agent_id_width is None else int(agent_id_width))
            if agent_id_obs else 0
        )
        self.include_ofi = include_ofi
        self.ofi_alpha = ofi_alpha
        # None leaves ABIDES' random placement alone (the historical
        # behaviour); a float pins our latency to the exchange at that
        # multiple of the background median. 0.0 is co-located.
        self.latency_edge = latency_edge
        self._latency_summary: dict | None = None
        self.symbol = symbol
        self.historical_date = historical_date
        self.start_time = start_time
        self.end_time = end_time
        self.wakeup_interval = wakeup_interval
        self.inventory_penalty = float(inventory_penalty)
        self.termination_penalty = float(termination_penalty)
        self.config_kwargs = dict(config_kwargs or {})

        self.possible_agents = [f"mm_{i}" for i in range(n_agents)]
        self.agents: list[str] = []
        self.render_mode = None  # required by SuperSuit's MarkovVectorEnv

        obs_dim = obs_vector_size(k, self.agent_id_width, self.include_ofi)
        self._obs_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        if gated_actions:
            # (bid_gate, ask_gate, bid_offset, ask_offset, bid_size, ask_size).
            # Gates span [-1, 1] and quote when > 0, so a policy initialised at
            # mean ~0 sits on the decision boundary instead of inside a
            # zero-gradient no-quote region. See actions.py for the mechanism.
            #
            # The offset dimensions run to max_offset_cents in the legacy
            # mapping (the raw value IS the offset in cents). Under a span they
            # span [-1, 1], for the same reason the gates do: SB3's Gaussian
            # policy initialises its mean at ~0 in action-space units, so a
            # [0, 1] box would start the policy pinned at min_offset_ticks with
            # half its probability mass clipped into a zero-gradient region -
            # measured at init, 51.8% of steps posted one tick under [0, 1]
            # against 17.5% under [-1, 1]. A symmetric box puts the initial
            # mean at the middle of the reachable range with gradient in both
            # directions, which is the entire point of the rescale.
            off_low = -1.0 if self.offset_span else 0.0
            off_high = 1.0 if self.offset_span else float(max_offset_cents)
            self._act_space = spaces.Box(
                low=np.array(
                    [-1.0, -1.0, off_low, off_low, float(min_size), float(min_size)],
                    dtype=np.float32),
                high=np.array(
                    [1.0, 1.0, off_high, off_high, max_size, max_size],
                    dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self._act_space = spaces.Box(
                low=np.array(
                    [0.0, 0.0, float(min_size), float(min_size)], dtype=np.float32),
                high=np.array(
                    [max_offset_cents, max_offset_cents, max_size, max_size],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )

        self.kernel: Kernel | None = None
        self._coord: MarlCoordinator | None = None
        self._children: list[MarlChild] = []
        self._mkt_open_ns: int = 0
        self._mkt_close_ns: int = 0
        # Per-agent equity from the previous step, for ΔPnL reward.
        # Indexed by possible_agents[i] (we keep all N entries even after
        # an agent terminates; lookups for terminated agents stop happening).
        self._prev_equity: dict[str, int] = {}
        self._terminated: dict[str, bool] = {}

    # ── PettingZoo space accessors ──────────────────────────────────────────
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Box:
        return self._act_space

    # ── Lifecycle ───────────────────────────────────────────────────────────
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is None:
            seed = np.random.randint(low=0, high=2**32, dtype=np.uint64)

        # Build background config
        cfg_kwargs = dict(self.config_kwargs)
        cfg_kwargs.setdefault("start_time", self.start_time)
        cfg_kwargs.setdefault("end_time", self.end_time)
        cfg_kwargs["seed"] = int(seed) % (2**32 - 1)
        bg_config = rmsc03_simple.build_config(**cfg_kwargs)

        # Compute market window in ns (matches rmsc03's own calc)
        date_ns = datetime_str_to_ns(self.historical_date)
        self._mkt_open_ns = date_ns + str_to_ns(self.start_time)
        self._mkt_close_ns = date_ns + str_to_ns(self.end_time)

        # Children + coordinator. ID space continues from the background agents.
        next_id = len(bg_config["agents"])
        wakeup_ns = str_to_ns(self.wakeup_interval)
        coord_first = wakeup_ns
        # Children wake a hair after the coordinator each tick so action
        # distribution lands before order placement.
        child_first = wakeup_ns + 100  # 100 ns offset

        self._children = [
            MarlChild(
                id=next_id + i,
                symbol=self.symbol,
                starting_cash=self.starting_cash,
                max_size=self.max_size,
                tick_size=self.tick_size,
                gated=self.gated_actions,
                min_offset_ticks=self.min_offset_ticks,
                min_quote_size=self.min_quote_size,
                offset_span=self.offset_span,
                mid_rounding=self.mid_rounding,
                wakeup_interval_generator=ConstantTimeGenerator(
                    step_duration=wakeup_ns
                ),
                first_interval=child_first,
                subscribe_num_levels=self.k,
            )
            for i in range(self.n_agents)
        ]
        self._coord = MarlCoordinator(
            id=next_id + self.n_agents,
            symbol=self.symbol,
            starting_cash=self.starting_cash,
            children=self._children,
            mkt_open_ns=self._mkt_open_ns,
            mkt_close_ns=self._mkt_close_ns,
            k=self.k,
            max_inventory=self.max_inventory,
            max_size=self.max_size,
            agent_id_width=self.agent_id_width,
            include_ofi=self.include_ofi,
            ofi_alpha=self.ofi_alpha,
            wakeup_interval_generator=ConstantTimeGenerator(
                step_duration=wakeup_ns
            ),
            first_interval=coord_first,
            subscribe_num_levels=self.k,
        )
        config = config_add_agents(
            bg_config, list(self._children) + [self._coord]
        )

        # config_add_agents regenerates the latency model for the enlarged
        # agent count, which drops our agents at a uniformly random point on
        # ABIDES' Seattle-to-NYC line. That makes our latency to the exchange
        # an *uncontrolled* variable, redrawn every seed. Pinning it is an
        # improvement in experimental control even at factor 1.0, which places
        # us at the population median rather than wherever the draw landed.
        if self.latency_edge is not None:
            lat = config.get("agent_latency_model")
            our_ids = tuple(c.id for c in self._children) + (self._coord.id,)
            lat.min_latency = apply_latency_edge(
                lat.min_latency, our_ids=our_ids, factor=self.latency_edge
            )
            self._latency_summary = describe_edge(lat.min_latency, our_ids)

        self.kernel = Kernel(
            random_state=np.random.RandomState(seed=int(seed) % (2**32 - 1)),
            **subdict(
                config,
                [
                    "start_time",
                    "stop_time",
                    "agents",
                    "agent_latency_model",
                    "default_computation_delay",
                    "custom_properties",
                ],
            ),
        )
        self.kernel.initialize()
        raw = self.kernel.runner()  # runs until coordinator's first non-None wakeup

        self.agents = list(self.possible_agents)
        self._terminated = {a: False for a in self.possible_agents}
        obs, infos = self._unpack(raw)
        # Seed prev_equity from the first per-agent snapshot so the first
        # step's ΔPnL is computed against a real baseline rather than zero.
        self._prev_equity = {
            a: self._equity_from_traj_row(infos[a]["traj_row"])
            for a in self.possible_agents
        }
        return obs, infos

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        # Pass actions only for agents that haven't been terminated. For a
        # terminated agent, send a no-op so the coordinator still has a slot
        # per child but no orders go in. Under the gated layout the no-op is
        # a closed gate (a gate quotes only when strictly > 0, so 0.0 is off);
        # under the legacy layout it is zero size. Both resolve to "post
        # nothing", but the arity has to match the action space.
        noop = (0.0,) * self._act_space.shape[0]
        action_list = []
        for i, agent in enumerate(self.possible_agents):
            if self._terminated[agent]:
                vec = noop
            else:
                vec = tuple(float(x) for x in actions[agent])
            action_list.append({"agent_idx": i, "action_vec": vec})

        raw = self.kernel.runner((self._coord, action_list))
        if raw.get("done") and not (raw.get("result") or {}).get("per_agent"):
            # The ABIDES terminal path has no coordinator payload. Read actual
            # final accounts and pending executions instead of fabricating zeros.
            raw["result"] = self._coord.snapshot_state(int(self.kernel.current_time))
        obs, infos = self._unpack(raw)

        # ABIDES keeps a one-minute kernel buffer after the exchange closes.
        # A trading episode ends at market close, not after that shutdown buffer.
        timestamp_s = float((raw.get("result") or {}).get("timestamp_s", 0.0))
        session_seconds = (self._mkt_close_ns - self._mkt_open_ns) / 1e9
        market_closed = timestamp_s >= session_seconds
        truncated = bool(raw.get("done", False) or market_closed)
        rewards: dict[str, float] = {}
        terms: dict[str, bool] = {}
        truncs: dict[str, bool] = {}
        newly_terminated: list[str] = []

        for agent in list(self.agents):
            row = infos[agent]["traj_row"]
            inv = int(row[1])
            equity = self._equity_from_traj_row(row)
            term_now = abs(inv) > self.max_inventory

            reward = self._compute_reward(
                prev_equity=self._prev_equity[agent],
                equity=equity,
                inventory=inv,
                inventory_penalty=self.inventory_penalty,
                terminated=term_now,
                termination_penalty=self.termination_penalty,
            )
            self._prev_equity[agent] = equity

            if term_now:
                self._terminated[agent] = True
                newly_terminated.append(agent)

            rewards[agent] = reward
            terms[agent] = term_now
            truncs[agent] = truncated
            infos[agent]["market_closed"] = market_closed

        # Drop terminated/truncated agents from the active set so PettingZoo
        # stops sending us actions for them.
        if truncated:
            self.agents = []
        else:
            for a in newly_terminated:
                if a in self.agents:
                    self.agents.remove(a)

        return obs, rewards, terms, truncs, infos

    def close(self) -> None:
        self.kernel = None
        self._coord = None
        self._children = []
        self.agents = []

    # ── Internals ───────────────────────────────────────────────────────────
    @staticmethod
    def _equity_from_traj_row(row: tuple) -> int:
        """Mark-to-mid equity = cash + inventory * mid_cents (integer cents)."""
        _ts, inv, cash, mid, _fq, _fp = row
        return int(cash) + int(inv) * int(mid)

    @staticmethod
    def _compute_reward(
        prev_equity: int,
        equity: int,
        inventory: int,
        inventory_penalty: float,
        terminated: bool,
        termination_penalty: float,
    ) -> float:
        """ΔPnL minus inventory penalty, plus termination penalty if terminated.

        ΔPnL is mark-to-mid: equity − prev_equity (cents). Inventory penalty
        is symmetric in sign: λ·|inv| applies to long and short positions
        identically.
        """
        reward = float(equity - prev_equity) - inventory_penalty * abs(inventory)
        if terminated:
            reward += termination_penalty
        return reward

    def _unpack(
        self, raw: dict
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        result = deepcopy(raw.get("result") or {})
        per_agent = result.get("per_agent", [])
        timestamp_s = float(result.get("timestamp_s", 0.0))

        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict] = {}
        for i, agent in enumerate(self.possible_agents):
            if i < len(per_agent):
                entry = per_agent[i]
                obs[agent] = entry["obs"]
                infos[agent] = {
                    "executions": entry.get("executions", []),
                    "submissions": entry.get("submissions", []),
                    "traj_row": (
                        timestamp_s,
                        entry["inventory"],
                        entry["cash"],
                        entry["mid_cents"],
                        entry["fill_signed_qty"],
                        entry["fill_price_cents"],
                    )
                }
            else:
                raise RuntimeError(f"Simulator response missing state for {agent}")
        return obs, infos
