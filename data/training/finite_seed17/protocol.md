# Concurrent learning and interaction controls

Protocol started 2026-09-25, before these runs. This is exploratory research,
with a held-out confirmation stage after inspecting the first training pair.
The active Goal is the broader research objective in `research_handoff.md`.

## Questions and design

Do independently learning traders change how they participate in the market?
Does a learned trader's state-dependent decision-making matter to its own or
its opponent's outcomes beyond the amount and kind of orders it submits?
If behavior changes during training, is it robust across initializations and
held-out markets? Mutual adaptation is a hypothesis, not a required result.

Two independently initialized PPO networks collect experience in one shared
market. Each has its own observations, actions, rewards, value estimates,
rollout buffer, optimizer, and reward normalization. Both update after each
256-step joint rollout, using the established SB3 PPO update. Neither updates
within a rollout. This is concurrent independent learning across repeated
sessions, not weight updates during a held-out trading session. Models are
frozen for evaluation. Retain paired initial/intermediate/final checkpoints.

Keep the pilot's market/action/reward settings initially: ten-minute sessions,
2,000 noise traders, 50 value traders, 10 momentum traders and two native
market makers; two learning traders, fixed quote size 10, independent gates,
1–16 tick offsets, outward midpoint rounding, observation OFI and identity,
inventory cap 1,000, inventory penalty 1e-4. Initial compute target is 32,768
joint steps (per-policy samples), saving pairs at 0/8,192/16,384/32,768. First
pair seed 17, then seeds 29 and 43 if the implementation and pilot justify
replication. No tuning on the confirmation market set.

Primary evaluations sample each trained policy's Gaussian, because training
uses sampling and the previous pilot exposed different behavior for the mean
action. Also report deterministic means as a diagnostic. Use private random
streams per trader so replacing B does not consume or shift A's random draws.

Candidate comparisons, to be fixed in run metadata before execution:

- Both trained, sampled: the primary learned pair.
- Both trained, deterministic: evaluation-semantics diagnostic.
- Both initial, sampled: starting behavior, without claiming it is a matched
  activity control.
- Trained A versus initial B: remove B's training while preserving A.
- Trained A versus inactive B: remove B's submitted flow.
- Trained A versus a state-blind empirical B: sample complete gate/offset/size
  rows from B's behavior on separate calibration markets. This preserves the
  calibration joint action distribution but removes its connection to current
  observations and time. It does not preserve exact realized quoting frequency,
  order timing, fills, or the market path; check those differences explicitly.

Calibration markets 20001–20004; exploratory evaluation markets 30001–30006.
Confirmation markets 40001–40012 remain unused until conditions are selected.
Initial action replicate 0; add repeat 1 to selected confirmation comparisons.
Market/action replicates from one trained pair are not independent training
replications. No significance claims based on treating time steps as samples.

Before the concurrent pilot, apply sampled-B, marginal-B, and inactive-B
comparisons to the existing sequential pair with A's deterministic inference
retained. Use the same calibration/exploratory seed partitions above and action
replicate 0. This secondary diagnostic addresses the earlier change in A's
outcomes; it is not a concurrent-learning result or a performance comparison
between the two training methods.

Measure both traders' P&L, inventory, fills, quote participation, offsets,
direct pair executions, and episode duration/risk stops. Reconcile fills and
accounts. Decompose marked P&L into execution relative to the observed mid
and inventory exposure where timing supports it; do not call midpoint marking
liquidated profit. Include uncertainty and seed-level results, including
negative findings. Inspect earlier/later checkpoint matchups if changes merit
an adaptation claim. Favor a mechanism intervention over a broad tuning sweep.

## Implementation/research references

We reuse the installed SB3 implementation for PPO training and inspect its
rollout/normalization conventions. The PettingZoo/CleanRL
[parallel PPO example](https://pettingzoo.farama.org/tutorials/cleanrl/advanced_PPO/)
is useful context, but its shared-network pattern does not satisfy our separate
policy requirement; no wholesale algorithm rewrite or copying is needed.
[Lanctot et al.](https://arxiv.org/abs/1711.00832) motivates evaluating agents
against other checkpoints instead of trusting one co-trained pair.
[ABIDES-MARL](https://arxiv.org/html/2511.02016v2) is related work on adaptive
liquidity and execution, not evidence that our symmetric quoting interface
will reproduce its behavior. Any later borrowed code must retain its license
and attribution; none has been copied for this protocol.

## Amendment, 2026-09-26: finite-horizon correction before confirmation

A targeted method audit found that carrying over SB3's external-timeout
bootstrap is inconsistent with the stated ten-minute task. Exchange close
ends the task and time-to-close is observed; no post-close reward is defined.
[Gymnasium's guidance](https://gymnasium.farama.org/main/tutorials/handling_time_limits/)
distinguishes this intrinsic finite horizon from an external rollout cutoff.
Close should have zero continuation value. Ordinary batch cutoffs within an
open session should still bootstrap.

The original concurrent runs (seeds 17, 29, 43) deliberately retained the old
bootstrap for pilot comparability. Preserve them as exploratory legacy runs;
their saved behavioral and P&L observations are valid, but they did not train
against the intended finite-session objective exactly. Do not silently pool
them with corrected runs or call the old results confirmation of that objective.

Implementation will add an explicit market-close marker to environment infos
without changing the existing evaluator's risk/close classification. The
concurrent trainer defaults to zero continuation at exchange close; an explicit
legacy option permits reproducing the previous behavior. External cutoffs and
within-session batch boundaries retain their existing bootstrap handling.

First corrected comparison: retrain seed17 from its original initialization
for 32,768 shared steps, with identical market/configuration and only this
learning-target correction. Evaluate the same six exploratory conditions on
30001–30006. Compare participation, inventories, P&L and policy responses;
if outcomes materially change or affect the main claims, replicate corrected
seeds 29/43 before selecting confirmation conditions. Keep 40001–40012
untouched until this comparison is understood. This amendment is recorded
before the corrected training starts.
