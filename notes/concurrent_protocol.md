# Concurrent learning and interaction controls

Protocol started 2026-09-25, before these runs. This is exploratory research,
with a held-out confirmation stage after inspecting the first training pair.
This public copy retains the experimental decisions but omits local operational
instructions. See `../PROVENANCE.md`: the fresh public
history does not independently establish the original recording chronology.

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

### Replication scheduling decision, 2026-09-26

The corrected seed 17 finished with identical initial tensors, configuration,
and training market-seed stream. Before its full exploratory evaluation ends,
the common-observation diagnostic already shows large behavioral changes:
B's mean either-quote probability is .786 versus legacy .598, and A's is
.575 versus .637. The legacy training seeds therefore cannot stand in for
training variability under the corrected target. Proceed with corrected
seeds 29 and 43 at the same 32,768-step budget and six exploratory conditions;
no hyperparameters or market settings change. Starting seed 29 while seed 17's
evaluation runs is a scheduling deviation from the original sequential order,
justified by this measured policy difference, not by selecting favorable P&L.
Complete these exploratory comparisons
before choosing confirmation conditions. This is the final planned training
replication batch for the current research stage, unless a validity defect
requires correction; further training sweeps are a next-stage decision.

## Confirmation decision after all corrected exploratory evaluations

Recorded 2026-09-26 before any 40001–40012 evaluation. All three corrected pairs
completed 32,768 training steps and the six exploratory conditions. The
learned-minus-empirical B P&L contrasts for A are -$102.79, -$58.07, +$10.95;
for B they are +$163.64, +$103.57, -$45.36. Every pair-level descriptive interval
includes zero. There is no coherent replicated profit advantage to promote as
a strategy result. We nevertheless have only one evaluation action draw per
market, and the policies have substantial sampling variance. Their changed
quote direction, pricing and inventory accumulation also deserve a check on
new market paths.

Run one bounded final evaluation batch, preserving the original intervention
question rather than selecting a profitable checkpoint or changing the setup:

- Freeze all three corrected final pairs (seeds 17, 29, 43). No further training.
- Test exactly `sampled_pair` and `marginal_b` on 40001–40012, replicates 0 and 1.
  This is 48 joint evaluation episodes per trained pair, 144 total.
- Use the same separate calibration markets 20001–20004 and their original
  action replicate 0 to construct each pair's empirical pool. Verify regenerated
  pools match the exploratory pools. Test markets never enter calibration.
- Primary outcome: B's full-session marked P&L difference, learned minus
  empirical B, reported separately for each trained pair. A's outcome is the
  secondary externality measure. Also report participation, side-specific
  requested offsets, gross buy/sell flows, inventory, risk stops and the exact
  equity decomposition. These explain behavior without redefining success
  around a favorable secondary number.
- Average the two action replicates within each market before paired market
  summaries/bootstrap intervals. Keep action-draw sensitivity visible; do not
  count 24 evaluations as 24 independent markets or 144 as 144 training seeds.
  There are still only three training replications. Intervals remain descriptive.
- If risk stops occur, retain full episodes as primary outcomes and inspect the
  existing common-decision-horizon sensitivity. Do not silently drop failures.
- Report the batch regardless of sign. A changed outcome can establish a
  simulator intervention effect; it cannot alone establish adaptation to a
  rival, exploitation, cooperation or equivalence of the policies.

This is a precision and generalization check on the predefined contrast, not
an attempt to prove a null through unlimited additional markets. It also checks
whether the observed directional inventories persist beyond the exploratory
paths. Complete the research-stage report and next-stage recommendation afterward;
no additional training sweep is planned.
