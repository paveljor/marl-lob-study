#!/usr/bin/env python3
"""Single source of truth for the experiment suites.

Emits one executable shell script per job into experiments/jobs/<suite>/ so the
runner can simply parallelise over files, each job gets its own log, and any job
can be re-run by hand. Nothing here runs anything.

Suites:
  smoke  1 short run - proves the pipeline works on a new machine
  cheap  the four high-value cheap experiments (see notes/design_space.md)
  grid   the 2D spine: informed fraction x number of market makers
  retrain  the May-2026 PPO ablation, re-run on post-E3 code (acts 3-4)
"""
import argparse
import os
import shlex
import stat
from pathlib import Path

SEEDS = (0, 1, 7)

# Ten seeds for the gate suite. Three was demonstrably too few here: the grid
# resolved nothing at 3 (signal/noise 0.73) and reversed sign between 15 and
# 50. Ten will not settle a magnitude either, but it is enough to say whether
# an effect the size of the single-seed gate's is real.
GATE_SEEDS = tuple(range(10))

# Twenty seeds for grid2. Chosen so the suite can be EXTENDED rather than
# re-run: seeds are contiguous from 0, jobs are seed-major, and adding
# 20..49 later only adds files. Sized off the primary dependent variable -
# withdrawn% has a seed sd of ~15-17pp, so a real 20->60pp swing across
# informed flow resolves at 20 seeds, and a +-5pp wobble would not resolve at
# 50 either. Override with MARL_GRID2_SEEDS.
# The grid needs far more seeds than the cheap suite. Measured on the 3-seed
# equalised grid (2026-08-30, M4): the spread of the 12 cell means was sd=838
# while the median within-cell seed sd was 1153, i.e. signal/noise = 0.73 - the
# design could not resolve its own competition effect. Detecting an effect that
# size against that noise at ~2 standard errors needs ~15 seeds per cell.
# Superset of SEEDS, so the 3-seed runs remain a subset.
def _seed_spec(env_var, default):
    """Seed set for a suite, overridable from the environment.

    Accepts "0-14" or "15,16,17". Lets a second machine extend the seed pool
    without editing this file: results are keyed by seed and carry their host
    in run_meta.json, so the pools merge.
    """
    spec = os.environ.get(env_var, "").strip()
    if not spec:
        return tuple(default)
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in spec.split(","))


GRID_SEEDS = _seed_spec("MARL_GRID_SEEDS", range(15))

# Twenty seeds for grid2. Chosen so the suite can be EXTENDED rather than
# re-run: seeds are contiguous from 0, jobs are seed-major, and adding
# 20..49 later only adds files. Sized off the primary dependent variable -
# withdrawn% has a seed sd of ~15-17pp, so a real 20->60pp swing across
# informed flow resolves at 20 seeds, and a +-5pp wobble would not resolve at
# 50 either. Override with MARL_GRID2_SEEDS.
GRID2_SEEDS = _seed_spec("MARL_GRID2_SEEDS", range(20))
OFI_SEEDS = _seed_spec("MARL_OFI_SEEDS", range(20))
LATENCY_SEEDS = _seed_spec("MARL_LATENCY_SEEDS", range(20))
RESCALE_SEEDS = _seed_spec("MARL_RESCALE_SEEDS", range(20))
P2_SEEDS = _seed_spec("MARL_P2_SEEDS", range(20))

# Eval seeds are cheap (~30s each, no training) and a single one leaves every
# number in the writeup resting on one sampled market. Three is the difference
# between "the agent made $X" and "the agent made $X across three markets".
EVAL_SEEDS = (42, 43, 44)


def job(name, train_args, *, n_agents=2, eval_seeds=EVAL_SEEDS, timesteps=50_000):
    return {"name": name, "n_agents": n_agents, "timesteps": timesteps,
            "train_args": list(train_args), "eval_seeds": list(eval_seeds)}


def suite_smoke():
    return [job("smoke", [], timesteps=2_000, eval_seeds=(42,))]


def suite_cheap():
    jobs = []
    # E1 (restated 2026-08-30 - the original premise was wrong). Observations
    # are NOT raw cents: extract_obs already returns prices as distance from
    # mid, sizes over max_size, and clips the whole vector to [-1, 1]. So this
    # is not the reward-scale bug on the input side.
    #
    # What it actually tests: the 44 features sit in [-1, 1] but do not move
    # equally within it. Measured on a rollout, ask_size_L1 has std 0.33 while
    # spread_norm has std 0.00026 - a ~1300x gap. Since first-layer weights all
    # start at the same scale, a feature's influence tracks how much it varies,
    # so the policy is nearly blind to the spread. --norm-obs (VecNormalize
    # per-dim running mean/std) equalises them; the risk is that near-constant
    # features get their jitter amplified to full scale, i.e. noise.
    #
    # Prediction (Pavel, before running): no material difference. Any effect
    # will be smaller than the seed-to-seed variance already known to be large,
    # and the honest reading at 3 seeds will be "this cannot tell us".
    for s in SEEDS:
        jobs.append(job(f"norm_obs/seed{s}", ["--norm-obs", "--seed", s]))
        jobs.append(job(f"norm_obs_off/seed{s}", ["--seed", s]))
    # Close BOTH escape hatches at once. min_size alone was routed around by
    # quoting ~50c off mid; clamp the offset at the same time.
    for s in SEEDS:
        jobs.append(job(f"both_hatches/seed{s}",
                        ["--min-size", 10, "--max-offset-cents", 5, "--seed", s]))
    # Restore the env's own inventory penalty, which the ablation showed is
    # strictly better than removing it when VecNormalize is on.
    for s in SEEDS:
        jobs.append(job(f"invpen_default/seed{s}",
                        ["--inventory-penalty", 1e-4, "--seed", s]))
    return jobs


def suite_grid():
    """informed fraction x competition.

    Predictions, committed before running (notes/design_space.md):
      - more informed flow  -> wider quotes, fewer fills, eventual withdrawal
      - more competitors    -> tighter quotes, less profit each
      - interaction: competition removes the cushion that lets a maker survive
        adverse selection, so withdrawal arrives at a LOWER informed fraction
        as n_agents rises

    Training budget is held constant PER AGENT (2026-08-30). SB3 counts
    total_timesteps as the sum across the vec env, and SuperSuit sets
    num_envs = n_agents, so a flat 50_000 gave each agent 50_176 / 25_088 /
    12_800 env steps at n_agents = 1 / 2 / 4 - a 4x spread in experience and
    in gradient updates, running along the very axis this grid varies. The
    competition effect would have been confounded with training budget.
    Scaling by n_agents gives every agent ~50k env steps regardless.
    """
    # Seed-major on purpose. The runner walks jobs in filename order, so this
    # makes every seed complete across all 12 cells before the next seed
    # starts. A run cut short then leaves a COMPLETE BALANCED design at
    # whatever seed count finished, instead of all of v10 and none of v400.
    # Two of these runs have already been killed mid-flight; this makes that
    # cost seeds rather than the whole grid.
    jobs = []
    for s in GRID_SEEDS:
        for value_agents in (10, 50, 150, 400):
            for n_agents in (1, 2, 4):
                jobs.append(job(
                    f"v{value_agents}_n{n_agents}/seed{s}",
                    ["--num-value-agents", value_agents, "--seed", s],
                    n_agents=n_agents,
                    timesteps=50_000 * n_agents,
                ))
    return jobs


def suite_retrain():
    """The May-2026 PPO experiments, re-run on fixed code.

    Every number in notes/experiments_results.md was produced before 808a6c0,
    with VecNormalize(norm_reward=True) on top of SuperSuit's uint8 dones - E3.
    Those policies learned against a corrupted reward scale, so re-evaluating
    the checkpoints only reproduces the bug faithfully. They need retraining.
    Presentation acts 3-4 rest on them.

    Two cells here have no VecNormalize at all, so E3 could not have touched
    them; they are included anyway because the whole point of the 2x2 is that
    the four cells differ in exactly one thing each, which stops being true if
    two of them were trained on different code from the other two.

    Budget is 50k env steps PER AGENT, matching the grid (E4). The May runs and
    the cheap suite both got 25k/agent from a flat 50_000 at n_agents=2, so
    these numbers are NOT comparable to either - which is why the two cells
    that duplicate cheap (exp1 == norm_obs_off, vecnorm_only == invpen_default)
    are re-run here rather than borrowed.

    Cells, and what each one is for in the talk:
      noop          act 3 - default inv penalty, no VecNormalize. The refusal:
                    raw ~1e7 rewards, value_loss ~1e11, approx_kl ~1e-5, policy
                    frozen at init. 0 fills, $0.
      invpen0_only  act 4 - removing the inventory penalty alone changes
                    nothing; it collapses to the same noop.
      vecnorm_only  act 4 - fix the reward scale and keep the default penalty.
                    Was the best PPO result of May (+$117 / +$113).
      exp1          act 4 - fix the scale, drop the penalty. Trades and loses.
      min_size_10   act 4 - force a minimum size and the agent routes around it
                    by quoting ~50c off mid (7 fills vs 1181). The talk's
                    specification-gaming beat.
      both_hatches  act 4 close - clamp the offset as well, and it finally
                    quotes. Budget-matched to min_size_10 on purpose: the
                    hatch-closing contrast is void if the two differ in
                    training steps as well as in constraints.
    """
    cells = [
        ("noop", ["--no-vecnormalize", "--inventory-penalty", 1e-4]),
        ("invpen0_only", ["--no-vecnormalize", "--inventory-penalty", 0.0]),
        ("vecnorm_only", ["--inventory-penalty", 1e-4]),
        ("exp1", ["--inventory-penalty", 0.0]),
        ("min_size_10", ["--min-size", 10]),
        ("both_hatches", ["--min-size", 10, "--max-offset-cents", 5]),
    ]
    jobs = []
    for s in SEEDS:
        for name, flags in cells:
            jobs.append(job(f"{name}/seed{s}", flags + ["--seed", s],
                            n_agents=2, timesteps=50_000 * 2))
    return jobs


def suite_gate():
    """Does the gated action space make markets, and does one identity bit
    make the agents distinct? At enough seeds to state a sign.

    The single-seed gate (`notes/gate_results.md`) said yes to both, and threw
    up a third thing worth replicating: with the identity one-hot the two
    agents' inventory correlation went *negative* (-0.477), finishing at -86
    and +651. Same network, one bit of identity, opposite positions. On one
    seed that is an anecdote - `notes/grid_results.md` records a contrast that
    read +448 (t=1.31) at 15 seeds and -88 at 50, reversing sign - so the
    question here is only whether the sign survives replication.

    Three cells isolate the two changes:

      legacy      the pre-2026-09 4-tuple, no identity. The control, and the
                  configuration every previous PPO number was measured on.
      gated_noid  gated actions, no identity. Separates "can it quote" from
                  "are they distinct" - the single-seed gate had this cell
                  trading 12x more than legacy while still posting identical
                  quotes on 99.94% of steps.
      gated_id    both changes. The intended new default.

    Budget is 50k env steps PER AGENT, matching `suite_grid` and
    `suite_retrain` (E4). The single-seed gate ran at 10k/agent, so its
    numbers are NOT comparable to these - see E4b in notes/design_space.md for
    the same trap costing a comparison before.
    """
    cells = [
        ("legacy", ["--legacy-actions", "--no-agent-id-obs"]),
        ("gated_noid", ["--no-agent-id-obs"]),
        ("gated_id", []),
    ]
    jobs = []
    for s in GATE_SEEDS:
        for name, flags in cells:
            jobs.append(job(f"{name}/seed{s}", flags + ["--seed", s],
                            n_agents=2, timesteps=50_000 * 2))
    return jobs


def suite_grid2():
    """informed flow x competition, on the gated action space. The spine.

    This is `suite_grid` re-run on code where the two prerequisites it silently
    depended on are actually met. The 2026-09 grid found nothing because the
    treatment was never applied: the agent posted a genuine two-sided market on
    ~2% of steps (`notes/grid_results.md`), and its "competing" agents were the
    same bot, quoting identically on 97-99.9% of steps
    (`notes/agent_divergence_results.md`). Both are now fixed and verified at
    10 seeds in `notes/gate_results.md`.

    The question has changed with them. `withdrawn%` used to be a rounding
    artifact and is now a gate the policy sets, so the Glosten-Milgrom
    prediction is directly testable rather than inferred from P&L. And because
    a properly quoting agent already loses at the default 50 value agents
    (-$1,373 per agent), the decisive cell is **v10_n1**: least informed flow,
    no competition. If a market maker cannot make money there, there is no
    viable region anywhere in this parameterisation - which is a result, not a
    failure.

    PREDICTIONS, committed before running:
      1. Withdrawal rate RISES with informed flow.
      2. Quoted spread WIDENS with informed flow.
      3. Delta-equity is NEGATIVE at every cell, v10_n1 included - no viable
         region.
      4. Competition effects stay small. Unlike the 2026-09 grid this is now a
         genuine test rather than a null by construction, because n_agents
         finally varies competitors rather than copies of one policy.

    `agent_id_width` is pinned to 4 so every cell shares one observation width
    regardless of n_agents; without it the 1/2/4 cells would train different
    input layers and stop being comparable. Budget stays 50k env steps per
    agent (E4).
    """
    jobs = []
    # Seed-major, for the reason given in suite_grid: a run cut short leaves a
    # complete balanced design at whatever seed count finished.
    for s in GRID2_SEEDS:
        for value_agents in (10, 50, 150, 400):
            for n_agents in (1, 2, 4):
                jobs.append(job(
                    f"v{value_agents}_n{n_agents}/seed{s}",
                    ["--num-value-agents", value_agents,
                     "--agent-id-width", 4, "--seed", s],
                    n_agents=n_agents,
                    timesteps=50_000 * n_agents,
                ))
    return jobs


def suite_ofi():
    """Can the agent respond to informed flow once it can SEE flow?

    grid2's P1 - withdrawal rises with informed flow - was unsupported, and its
    P2 - spread widens - was unsupported with a degenerate dependent variable
    (the policy sat pinned at the 2c floor at all twelve cells). But the
    observation could not have supported either. It is entirely a snapshot: the
    book as it stands, plus inventory, cash, spread and time. Adverse selection
    is a statement about *flow*, and an agent with no flow signal cannot
    condition on it - it would look exactly like this whether or not the effect
    is real. So P1's failure is currently a fact about our observation.

    This is the direct test: the same informed-flow sweep, with and without the
    two OFI features, `n_agents=1` so competition cannot confound it (grid2
    showed competition does nothing anyway, and n=1 halves the cost).
    `agent_id_width` stays pinned at 4 so these cells remain directly
    comparable to grid2's v*_n1 cells.

    PREDICTIONS, committed before running:
      1. WITHOUT OFI, withdrawal stays flat across informed flow - reproducing
         grid2's P1 null on n=1 cells. This is the control, and if it fails the
         comparison is void.
      2. WITH OFI, withdrawal RISES with informed flow. This is the real test.
      3. WITH OFI, quoted spread rises above the 2c floor at high informed
         flow. The floor binds at 2c, so the only direction it can move is up,
         and grid2 showed the raw pre-quantisation offsets sitting at 0.18-0.29c
         - the policy was not trying to widen at all.
      4. Delta-equity stays negative everywhere. Seeing informed flow is not
         the same as being able to profit despite it, and nothing here gives
         the agent an edge - that is the latency experiment, next.

    Falsification: prediction 2 fails if the v400 - v10 withdrawal contrast
    with OFI is within noise, or is no larger than the same contrast without.
    The interaction is the claim, not the main effect.
    """
    jobs = []
    for s in OFI_SEEDS:
        for value_agents in (10, 50, 150, 400):
            for tag, flags in (("ofi_off", []), ("ofi_on", ["--ofi"])):
                jobs.append(job(
                    f"v{value_agents}_{tag}/seed{s}",
                    ["--num-value-agents", value_agents,
                     "--agent-id-width", 4, "--seed", s] + flags,
                    n_agents=1,
                    timesteps=50_000,
                ))
    return jobs


def suite_latency():
    """Does a speed edge create a viable region where none exists?

    grid2 found no cell of the informed-flow x competition design profitable,
    so the question stops being "where is the boundary" and becomes "does an
    edge create one at all". Speed is the canonical market-maker edge and
    ABIDES has had it wired in and unused the whole time.

    Run at v50 deliberately: it was the LEAST unprofitable n=1 cell in grid2
    (-$1,820, against -$2,649 at v10), so it is where viability is closest and
    an edge has the best chance of mattering.

    **What a latency edge can and cannot buy here, stated before running.** The
    background median latency to the exchange is 3.2ms and our wake-up interval
    is 1s. So the edge cannot buy faster *reaction* - we cannot re-quote until
    our next wake-up regardless. What it buys is **queue priority**: our order
    reaches the book ahead of a competitor's identical-price order, and our
    cancel lands sooner. A real HFT speed edge works through microsecond
    re-quoting, which this cadence cannot express; that is a limitation of the
    experiment, not a finding, and it belongs in any writeup.

    factor 1.0 is not a no-op - it is a control. Left unset, config_add_agents
    drops our agents at a random point on ABIDES' Seattle-to-NYC line, so
    latency has been uncontrolled in every previous run.

    PREDICTIONS, committed before running:
      1. Fill rate RISES as latency falls. This is the mechanism the cadence
         permits - queue priority - and if it does not move, the edge was never
         applied and nothing else here is interpretable.
      2. Delta-equity does NOT improve, and plausibly gets WORSE. grid2 measured
         capture at ~1% of |P&L| with everything else adverse selection, so
         winning more of the same fills means losing more. A speed edge that
         makes things worse is the interesting outcome and it is the one
         predicted here.
      3. No cell becomes profitable. The edge does not create a viable region.
      4. Quoted spread stays pinned at the 2c floor - nothing here gives the
         agent a reason to widen.
    """
    jobs = []
    for s in LATENCY_SEEDS:
        for factor in (0.0, 0.25, 1.0, 4.0):
            tag = f"lat{str(factor).replace('.', '')}"
            jobs.append(job(
                f"{tag}/seed{s}",
                ["--num-value-agents", 50, "--agent-id-width", 4,
                 "--latency-edge", factor, "--seed", s],
                n_agents=1,
                timesteps=50_000,
            ))
    return jobs


def suite_rescale():
    """With the offset dimension actually usable, does quoting change at all?

    Every null this project has produced about *how* the agent quotes turned
    out to be a statement about the action space. Under gating a live side
    resolved as ``max(round(raw), 1)``, so every raw in [0, 1.5) posted one
    tick - identical quote, identical reward - and the policy emitted raw >
    1.5 on 0.06-1.4% of steps. Widening was unrepresentable, which makes
    grid2's P1 unfalsifiable rather than answered.

    A fourth defect in the same path is fixed but untested here: ``mid =
    (bid + ask) // 2`` floors an odd-spread mid toward the bid, so a symmetric
    offset puts the ask a tick closer to true mid. Measured, the
    fills-per-posted-order gap between odd and even spreads is +0.105 to
    +0.136, positive on 19-20 of 20 seeds.

    Five arms separate the two changes and probe the span rather than
    asserting it:

      legacy   - control, must reproduce the latency suite's lat10 cell
      span6    - rescale, ceiling 7 ticks, a fresh policy starts at 4
      span15   - rescale, ceiling 16 ticks, a fresh policy starts at 9
      ceil     - the mid mirror alone, on the legacy action space
      outward  - the full fix

    span is an arm, not a constant, because the cold start is the mirror of
    the defect being fixed: background one-sided distance from mid has p90 =
    4.5c, so a 9-tick quote sits outside the market on ~95% of snapshots and
    may take too few fills to learn from. span6 starts at 4 ticks, right at
    p90.

    latency_edge is pinned to 1.0 throughout. It was an uncontrolled variable
    in every suite before 2026-09-05, redrawn per seed.

    **No directional prediction is pre-registered** - see
    notes/rescale_preregistration.md, where that is the recorded position and
    the reasoning for it. The acceptance gates are what carry the weight here:

      1. offset dispersion - legacy reaches 4 distinct offsets with 93.4% at
         one tick; a span arm that trains back to that did not take
      2. fills per submitted order - a collapse means the arm says nothing
         about market making
      3. the parity diff-in-difference - ~0 under outward, sign-flipped under
         ceil, or the rounding change did not take
      4. legacy must reproduce lat10 within seed noise, or the suite is void
    """
    arms = (
        ("legacy",  []),
        ("span6",   ["--offset-span", 6]),
        ("span15",  ["--offset-span", 15]),
        ("ceil",    ["--mid-rounding", "ceil"]),
        ("outward", ["--offset-span", 15, "--mid-rounding", "outward"]),
    )
    jobs = []
    for s in RESCALE_SEEDS:
        for tag, flags in arms:
            jobs.append(job(
                f"{tag}/seed{s}",
                ["--num-value-agents", 50, "--agent-id-width", 4,
                 "--latency-edge", 1.0, "--seed", s] + flags,
                n_agents=1,
                timesteps=50_000,
            ))
    return jobs


def suite_p2():
    """Does the agent widen against informed flow, now that it CAN?

    This question has been asked three times and never once been askable.
    grid2's P1 found no spread response, but the observation had no order-flow
    signal. The OFI suite added one, confirmed it was healthy (std 0.42-0.46,
    the most active features in the observation) and still found nothing -
    because a live side resolved as max(round(raw), 1) and every raw in
    [0, 1.5) posted the same tick. Widening was unrepresentable, so neither
    null was evidence about market making.

    The rescale suite closed that. Under a span the trained policy emits every
    reachable tick with tick sd ~4.0 and TVD ~0.8 from its own initialisation,
    and because eval is deterministic that variation is *state-dependence*.
    The policy now conditions its offset on the observation. What it conditions
    ON is unmeasured, and that is exactly this suite.

    Design mirrors suite_ofi exactly - same four informed-flow levels, same
    ofi on/off contrast, same seeds - so the difference-in-differences is
    directly comparable to a suite that reproduced grid2. The only change is
    that the action space can now express the response being looked for:
    every arm runs --offset-span 15 --mid-rounding outward.

    The claim is the INTERACTION, as it was in suite_ofi: withdrawn% and
    spread should move with informed flow more under --ofi than without. A
    main effect without the interaction is not the claim.

    Gates, in order:
      1. the ofi_off arm at v50 must reproduce the rescale suite's outward
         cell (dEq -721, spread 16.94, fills 316) within seed noise
      2. offset dispersion must stay wide - >=12 of 16 ticks emitted, tick sd
         >2. If the policy re-collapses onto one tick the DV is degenerate
         again and nothing here is interpretable
      3. report effect sizes beside every p-value. grid2 produced p=0.041 on a
         spread effect of -0.026c in the wrong direction because the DV was
         pinned; spread is no longer pinned but seeds disagree about it by a
         factor of four, which is its own trap

    Predictions: see notes/p2_preregistration.md, committed before the first
    job.
    """
    jobs = []
    for s in P2_SEEDS:
        for value_agents in (10, 50, 150, 400):
            for tag, flags in (("ofi_off", []), ("ofi_on", ["--ofi"])):
                jobs.append(job(
                    f"v{value_agents}_{tag}/seed{s}",
                    ["--num-value-agents", value_agents,
                     "--agent-id-width", 4, "--seed", s,
                     "--latency-edge", 1.0,
                     "--offset-span", 15, "--mid-rounding", "outward"] + flags,
                    n_agents=1,
                    timesteps=50_000,
                ))
    return jobs


SUITES = {"smoke": suite_smoke, "cheap": suite_cheap,
          "grid": suite_grid, "retrain": suite_retrain,
          "gate": suite_gate, "grid2": suite_grid2, "ofi": suite_ofi,
          "latency": suite_latency, "rescale": suite_rescale,
          "p2": suite_p2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--jobs-dir", type=Path, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    jobs_dir = args.jobs_dir or repo / "experiments" / "jobs" / args.suite
    if jobs_dir.exists():
        for f in jobs_dir.glob("*.sh"):
            f.unlink()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    jobs = SUITES[args.suite]()
    for i, j in enumerate(jobs):
        out = f"runs/{args.suite}/{j['name']}"
        train = ["python", "scripts/train.py",
                 "--out-dir", out,
                 "--n-agents", str(j["n_agents"]),
                 "--total-timesteps", str(j["timesteps"])]
        train += [str(a) for a in j["train_args"]]
        ev = ["python", "scripts/eval.py", f"{out}/ppo_marl_lob.zip",
              "--n-agents", str(j["n_agents"]),
              "--seeds", *[str(s) for s in j["eval_seeds"]],
              "--out-dir", f"{out}/eval", "--no-baseline"]

        flat = j["name"].replace("/", "__")
        script = jobs_dir / f"{i:03d}_{flat}.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.."\n'
            f'echo "=== {j["name"]} ==="\n'
            f"{shlex.join(train)}\n"
            f"{shlex.join(ev)}\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    print(f"wrote {len(jobs)} jobs to {jobs_dir.relative_to(repo)}")
    if args.suite == "grid":
        print(f"NOTE: {len(jobs)} runs at {len(GRID_SEEDS)} seeds/cell. "
              "Measured on an M4 at -j8: ~56 min for the 36-job 3-seed "
              "version, so budget ~5h here and far longer on a slow box.")


if __name__ == "__main__":
    main()
