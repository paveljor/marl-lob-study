# Recommended next research stage

Final recommendation, 2026-09-26, after all three confirmation batches.
This proposes a follow-up; these new experiments have not been executed.

## The question worth isolating

Did B benefit by reacting to A, or by conditioning its inventory exposure on
its own state and the wider market? The completed seed 29 confirmation gives
us a specific candidate to investigate: learned B has $838.70 less marked loss
than its state-blind replacement on average, positive in nine of twelve new
markets. Both action-draw averages are positive. Most of the difference is
inventory marking; A's average difference is close to zero. Seed 17 does not
show the same B benefit. Seed 43 has a positive mean with substantial
uncertainty; its interval includes zero. All three batches are complete.

This changes the initial recommendation. Before building a smaller market,
run a **targeted opponent-presence check in the existing market**. The present
comparison establishes a benefit from state conditioning for one pair on this
market set. It does not isolate responses to A from responses to time,
inventory or background flow.

## Immediate proposed follow-up: does the benefit require A?

Use the existing seed 29 checkpoints, with no new training, in four conditions:

| A behavior | B behavior |
|---|---|
| Learned, sampled | Learned, sampled |
| Learned, sampled | Original state-blind action pool |
| Inactive | Learned, sampled |
| Inactive | Original state-blind action pool |

Keep B's pool fixed across the four conditions so the replacement itself does
not change when A is removed. Use fresh markets and independently seeded action
repeats, preserve private role streams, and predeclare the comparison before
running it. Selecting seed 29 after its current result is hypothesis generation;
reusing the same markets would not be a fresh confirmation. Twelve markets and
two action repeats would make 96 joint episodes, without additional training.
This is a proposed bounded design; it has not been executed.

The primary statistic would be the difference between B's learned-minus-pool
benefit with A present and that benefit with A inactive. Report both constituent
contrasts, full-session outcomes, inventory paths, requested quotes, share
flows and risk stops. Average repeats within market. The inactive intervention
changes liquidity and subsequent market paths; it does not reveal a hidden
counterfactual on an otherwise identical realized book.

If B benefits similarly without A, the current evidence points toward broader
state-conditioned trading rather than a benefit requiring its fellow learner.
If A's presence reproducibly changes that benefit, investigate the changed
flow/exposure mechanism and then compare relevant opponent checkpoints. Even
that result would establish opponent dependence, not by itself recognition,
exploitation or mutual adaptation. Freezing weights during these evaluations
makes the intervention interpretable; the discovery training remains concurrent
and self-interested.

This defines a possible next stage. The present study has completed its
three-pair comparison; it does not require an unlimited chain of follow-ups.

## Fallback: verify measurable competition before more training

If the candidate does not persist or the large background market obscures its
mechanism, use a small **competition positive control**. Here a positive control
means a setting where ordinary scripted quotes demonstrably affect who gets
filled. It checks whether the experiment exposes a learnable consequence. It is
deliberately constructed and must not be presented as emergent exploitation.

Direct A–B trades are not the success criterion: two makers can compete for the
same customer's order without ever trading with each other. Measure changes
in customer-flow allocation, inventory and account outcomes when one maker's
quotes change.

## Proposed sequence and decision gates

1. Preserve the present ABIDES configuration as a reference. Add an isolated
   configuration with a small, transparent customer-flow source and two quoting
   traders, retaining the existing exchange, adapters, accounting and policy
   interface where practical. Specify an external price/reference process and
   customer-order schedule that can be held fixed across interventions. Any
   backstop liquidity used to maintain a valid book must be documented. This
   changes the market design; it is not a like-for-like performance comparison
   against the current background population.
2. Before PPO training, use scripted quoting policies to verify that changing
   one trader's prices changes its share of customer fills and the other
   trader's opportunity. Check both directions of the intervention and swap
   physical trader slots to detect scheduling/identity artifacts. Keep sizes
   and opportunities comparable. A tighter quote should be checked for greater
   access to flow, not assumed to earn more profit.
3. If those checks fail, investigate the order interface, timing and routing;
   spending more on neural-network training would not resolve that failure.
   If they pass, test the learning setup on a fixed, documented competitor as
   a capacity check, then make concurrent independent learning the discovery
   experiment. The fixed competitor is a validation tool, not evidence of
   mutual adaptation.
4. Keep both discovery agents self-interested. Do not give either a reward for
   reducing the other's P&L. Save paired checkpoints and compare earlier/later
   opponents and unseen training partners after learning. A repeatable change
   under those matchups would be stronger evidence of adaptation than one
   profitable co-trained pair or synchronized quotes.
5. Predeclare held-out markets/action draws and the behavior to inspect before
   evaluating. If a policy discovers a directional inventory bias, distinguish
   that from a response to its opponent. If it withdraws, compare sampled and
   deterministic behavior and verify the action mapping. Stop with an honest
   negative result if the opportunity is detectable but the chosen learner
   cannot exploit it within the agreed budget.

Before choosing training targets for this new configuration, specify the
economic task explicitly: marked wealth versus terminal liquidation, capital
and shorting constraints, and inventory risk relative to the price scale.
Changing these assumptions changes the research question. The current
0.0001-cent/share/step penalty and ignored order-level cash check should not
quietly become a claim of realistic risk management. Start with transparent
units and scripted checks; do not tune penalties until an impressive story
appears.

## Scope of the recommendation

The seed 29 result justifies a narrow mechanism check before another training
sweep; it does not establish a universal strategy across independently trained
pairs. This recommendation is an inference from the project evidence, not a
result established by the cited papers. Keep the current study's limits
visible: three training pairs, short sessions, a 32,768-step training budget,
stochastic policies and an idealized financial objective. It cannot show that
reinforcement learning generally cannot trade or that interaction is impossible.

## Related work and what to borrow

[Ganesh et al. (2019)](https://arxiv.org/html/1911.05892v1) studies a dealer
market in which investors select among makers' prices. It includes scripted
competitors and a simplified pricing benchmark. The useful methodological
idea is to make a competitor's effect on order allocation interpretable. Its
dealer market, observation set and hedging action differ from this ABIDES LOB;
its results are not a benchmark score our agents can be compared with directly.

[Wang, Ventre and Polukarov (2025)](https://arxiv.org/html/2510.25929v1) uses
synthetic flow allocation and heterogeneous objectives, including an agent
explicitly trained to reduce another's reward and training against a fixed
agent. Those design choices make it unsuitable as evidence that self-interested
concurrent learners spontaneously invent predation. Its focus on behavior beyond
aggregate P&L is useful, but we should not import the adversarial reward merely
to obtain the desired narrative.

No external code has been copied for this recommendation. If a later stage
reuses an implementation, inspect its code, assumptions and license separately.

## Compute and scope

The current local machine trained each corrected pair in roughly 14–21 minutes.
That supports local prototyping and a measured timing run before deciding on
desktop or paid compute. It is not a performance estimate for the proposed
configuration.

The LLM-with-communication idea remains a separate future branch. It would add
new questions about message access, incentives, inference costs and evaluation,
so it should not obscure whether the current market exposes meaningful
strategic choices. The completed stage provides verified results and an explanation.
The communication branch remains separate from the next mechanism test.
