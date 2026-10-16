"""Give our agents a latency edge — the canonical market-maker advantage.

Speed is *the* structural edge a real market maker has, and grid2 sharpened why
it matters here: with no viable region anywhere in the informed-flow x
competition design, the question stops being "where is the boundary" and
becomes "does an edge create one at all".

How ABIDES models it
--------------------
`generate_latency_model(agent_count)` places every agent at a uniformly random
point on a line the length of Seattle-to-NYC and sets pairwise minimum latency
to the light-speed travel time between them. `min_latency[i][j]` is the
one-way delay in nanoseconds for a message from agent i to agent j, and the
kernel applies it to every message. The exchange is agent 0.

Two consequences worth stating plainly:

* **Our agents currently get a random position on that line**, because
  `config_add_agents` regenerates the whole model once our agents are appended.
  Latency has therefore been an *uncontrolled* variable across every run this
  project has done - a different draw every seed.
* Setting it deliberately is therefore an improvement in experimental control
  even at `factor=1.0`, which pins our agents to the population median instead
  of leaving them to the draw.

What counts as an edge
----------------------
Agents in ABIDES talk to the exchange, not to each other, so the meaningful
quantity is our latency *to and from agent 0*. That is what co-location buys.
`apply_latency_edge` scales exactly that pair and leaves every other pair
untouched, so the background population's own latency structure is preserved.
"""
from __future__ import annotations

import numpy as np

EXCHANGE_ID = 0


def median_latency_to_exchange(
    min_latency: np.ndarray,
    exchange_id: int = EXCHANGE_ID,
    exclude: tuple[int, ...] = (),
) -> float:
    """Median one-way latency from the background population to the exchange.

    This is the yardstick an edge is measured against: `factor=1.0` puts our
    agents exactly here, so "no edge" means "a typical participant" rather than
    "wherever the random draw landed".
    """
    n = min_latency.shape[0]
    others = [i for i in range(n) if i != exchange_id and i not in exclude]
    if not others:
        raise ValueError("no background agents to take a median over")
    return float(np.median(min_latency[others, exchange_id]))


def apply_latency_edge(
    min_latency: np.ndarray,
    our_ids: tuple[int, ...],
    factor: float,
    exchange_id: int = EXCHANGE_ID,
) -> np.ndarray:
    """Set our agents' latency to and from the exchange to `factor` x median.

    `factor=1.0` is a typical participant, `0.1` is ten times faster than the
    median, `0.0` is co-located. Values above 1 model a *disadvantage*, which
    is a legitimate arm of the experiment.

    Returns a modified copy; the input is not mutated. Latency is symmetric
    here because a co-located maker is closer in both directions - the wire
    does not care which way the message travels.
    """
    if factor < 0:
        raise ValueError(f"latency factor must be >= 0, got {factor}")
    out = np.array(min_latency, copy=True)
    median = median_latency_to_exchange(out, exchange_id, exclude=tuple(our_ids))
    edge = int(round(median * float(factor)))
    for i in our_ids:
        if i == exchange_id:
            raise ValueError("an agent cannot have a latency edge over itself")
        out[i, exchange_id] = edge
        out[exchange_id, i] = edge
    return out


def describe_edge(
    min_latency: np.ndarray,
    our_ids: tuple[int, ...],
    exchange_id: int = EXCHANGE_ID,
) -> dict:
    """Where our agents sit in the population's latency distribution.

    Reported at run start so a result can never be read without knowing what
    edge produced it - `percentile` is the share of background agents that are
    SLOWER than us, so 100 means fastest in the market.
    """
    n = min_latency.shape[0]
    others = [i for i in range(n) if i != exchange_id and i not in our_ids]
    pop = min_latency[others, exchange_id]
    ours = float(np.mean([min_latency[i, exchange_id] for i in our_ids]))
    return {
        "ours_ns": ours,
        "median_ns": float(np.median(pop)),
        "min_ns": float(pop.min()),
        "max_ns": float(pop.max()),
        "percentile": float(100.0 * (pop > ours).mean()),
    }
