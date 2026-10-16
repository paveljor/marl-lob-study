#!/usr/bin/env python3
"""Build the research walkthrough without importing Jupyter or the simulator.

Run from any directory. --check verifies source cells, allowing reviewed outputs.
Building writes blank outputs; execution/review is a separate publication step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "research_walkthrough.ipynb"


def build_notebook() -> dict:
    cells = []

    def cell(kind: str, source: str, slide: str = "slide") -> None:
        item = {
            "cell_type": kind,
            "id": f"research-{len(cells):03d}",
            "metadata": {"slideshow": {"slide_type": slide}},
            "source": dedent(source).strip() + "\n",
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    def md(source: str, slide: str = "slide") -> None:
        cell("markdown", source, slide)

    def code(source: str, slide: str = "subslide") -> None:
        cell("code", source, slide)

    md(r"""
    # Two learning traders, one synthetic order book
    ## A research walkthrough — findings completed 26 September 2026

    **Can two independently learning traders develop useful behavior in the same market?**

    The honest short answer: behavior changed, an early interaction candidate failed
    after a training correction, and one held-out policy had **lower losses** when it
    could use current state. We have not shown exploitation, cooperation, or profit.

    This notebook walks through the question, controls, correction, held-out results,
    inventory accounting, and the most useful takeaways and open questions.

    **Reading mode:** read each explanation, run its code, then inspect the chart.
    Code is ordinary Python with NumPy, pandas, and Matplotlib. No training, simulator,
    model/checkpoint loading, downloads, or network calls occur here. Jupyter is only
    the container. Cells include slideshow metadata for optional presentation export.
    """)
    md("""
    ### Evidence, not a reconstructed origin story

    The numerical sources are committed CSV summaries in `notes/assets/` and numeric
    trajectories in `data/confirmation/`. The narrative reference is
    [`research_report.md`](../notes/research_report.md); read
    [`PROVENANCE.md`](../PROVENANCE.md) for the release's provenance boundaries.
    The public Git history is a curated source release, not the original development
    chronology. Recorded protocol timing comes from the original archive, not from
    the new public commit dates.

    All quantitative charts below are computed in cells, not screenshots or copied
    result arrays. Assertions stop execution if core evidence invariants change.
    The figures are descriptive, not a trading recommendation.
    """, "subslide")
    code("""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from IPython import get_ipython
    from IPython.display import display
    from matplotlib.patches import FancyBboxPatch

    # Render static figures even when CI inherits MPLBACKEND=Agg.
    get_ipython().run_line_magic('matplotlib', 'inline')

    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next((p for p in candidates if
                 (p / 'notes/assets/concurrent_finite_confirmation.csv').is_file()), None)
    if ROOT is None:
        raise FileNotFoundError('Open this notebook from the repository root or notebooks/.')
    ASSETS = ROOT / 'notes/assets'
    # Match scripts/analyse_interaction.py and the archived report's intervals.
    BOOTSTRAP_SEED = 12345
    BOOTSTRAP_DRAWS = 10000
    A_COLOR, B_COLOR = '#35688c', '#ad653a'
    LEARNED_COLOR, POOL_COLOR = '#35688c', '#969ba1'
    CONDITION = {'sampled_pair': 'Learned B', 'marginal_b': 'State-blind B'}
    AGENT = {'mm_0': 'A', 'mm_1': 'B'}
    plt.rcParams.update({
        'figure.figsize': (11, 5), 'figure.dpi': 110,
        'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .17, 'axes.axisbelow': True,
        'figure.constrained_layout.use': True,
        'savefig.facecolor': 'white',
    })
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.precision', 2)

    def load_summary(filename):
        frame = pd.read_csv(ASSETS / filename)
        # A training seed labels a separately trained pair, not a market path.
        frame['pair'] = frame.training_run.str.extract(r'(?:finite_)?seed(\\d+)').astype(int)
        return frame

    confirmation = load_summary('concurrent_finite_confirmation.csv')
    behavior = load_summary('concurrent_finite_confirmation_behavior.csv')
    horizon = load_summary('concurrent_seed17_horizon_comparison.csv')
    exploratory = load_summary('concurrent_finite_exploratory.csv')
    sequential = pd.read_csv(ASSETS / 'sequential_controls_20260925.csv')
    print('Data source: committed study tables and numeric trajectories.')
    print('Loaded evidence; no models or simulator imported.')
    """)
    md(r"""
    ## 1 · What is a market maker actually doing?

    A **bid** is an offer to buy; an **ask** is an offer to sell. Their price gap is
    the **spread**. The midpoint is halfway between the best bid and ask. A quote is
    only an offer: someone else must choose to trade with it.

    After buying, the trader holds **inventory** (shares). A positive position is
    *long*: it gains marked value if price rises. A negative position is *short*:
    it benefits if price falls. Spread income can be overwhelmed by price changes
    while inventory is held. Merely posting both sides does not guarantee balanced
    fills, nor does the label “market maker” guarantee a market-neutral strategy.

    Here ABIDES simulates ten-minute sessions in the synthetic symbol `ABM`, with
    2,000 noise traders, 50 value traders, 10 momentum traders, and two programmed
    adaptive makers **in addition to** the two learners. This is not historical replay.
    """)
    code("""
    # A conceptual flow diagram, not measured prices or an empirical chart.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis('off')
    def box(x, y, text, color, width=2.6):
        ax.add_patch(FancyBboxPatch((x, y), width, .9, boxstyle='round,pad=.12',
                                   facecolor=color, edgecolor='none', alpha=.12))
        ax.text(x + width / 2, y + .45, text, ha='center', va='center')
    box(.3, 3.4, 'Learner A\\nOwn cash + inventory', A_COLOR)
    box(7.1, 3.4, 'Learner B\\nOwn cash + inventory', B_COLOR)
    box(3.7, 1.9, 'Shared order book\\nBids  |  Asks', '#555555')
    box(3.7, .15, 'Background traders\\nOrders and fills', '#555555')
    for start, end in [((2.9, 3.65), (3.7, 2.6)), ((7.1, 3.65), (6.3, 2.6)),
                       ((5, 1.05), (5, 1.9))]:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops={'arrowstyle': '<->', 'color': '#666666'})
    ax.text(.3, 1.65, 'Buy fill → cash ↓, shares ↑\\nSell fill → cash ↑, shares ↓', va='center')
    ax.text(7.0, 1.65, 'Inventory stays exposed\\nbetween fills', va='center')
    ax.set_title('Offers become positions only when they fill — conceptual diagram', loc='left')
    plt.show()
    """)
    md(r"""
    ## 2 · Two PPO learners, not two copies of one brain

    PPO (proximal policy optimization) updates a neural network from sampled
    actions and rewards. A and B have **separate networks, optimizers, value
    estimates, reward statistics, and checkpoints**. An older shared-network
    implementation is not the independent-learning experiment discussed here.

    Each sees ten book levels, its own cash/inventory, spread, time remaining,
    and order-flow imbalance. There is no hidden view of the rival's account,
    messaging channel, explicit opponent model, or language model.

    Every roughly one-second decision enables/disables each side and sets offsets
    1–16 ticks from a rounded midpoint. Each enabled side offers ten shares: **size
    is fixed**, not learned. Orders are cancelled/replaced, losing queue priority.
    Both learners collect the same 256-step market rollout, then each updates its
    own weights; both buffers are finalized before either update. Evaluation freezes
    those weights. Each pair trained for 32,768 shared steps.

    ```text
    shared market → A observation → A sampled action → A reward/buffer → A PPO update
                  → B observation → B sampled action → B reward/buffer → B PPO update
                          both actions feed back into the shared market
    ```
    """)
    md(r"""
    ### The objective is not realistic net trading profit

    Step reward is the change in cash plus midpoint-valued inventory, minus a
    small inventory penalty and an inventory-breach penalty. The soft penalty is
    only `0.0001` cents per absolute share per step: holding 1,000 shares for 600
    steps costs 60 cents. The 1,000-share limit is the main guard.

    Rewards are normalized/clipped during training; `gamma = 0.99` discounts future
    rewards (roughly a 100-second scale). There are **no fees or forced liquidation**
    at close. The order-level cash check is bypassed (`ignore_risk=True`); shorting,
    negative cash, and large notional positions are possible without realistic margin
    or financing costs. Starting cash is not a fully specified capital requirement.

    Therefore: final **marked P&L** (profit and loss) is not cash-only realized profit,
    an investor return, or exactly the training objective.
    """, "subslide")
    md("""
    ### Why compare a learned policy with a state-blind action pool?

    In `sampled_pair`, both frozen trained policies sample actions. In `marginal_b`,
    A keeps its weights but B draws whole actions from its calibration-market pool.
    That pool uses separate calibration markets 20001–20004; held-out evaluation
    markets never enter it.
    The pool retains the joint distribution of gates and offsets, but disconnects
    them from **current** book state, inventory, and time. It does not preserve the
    exact realized timing, fills, or price path. Private role random streams avoid
    silently shifting A's random draws when B is replaced.

    This asks whether B benefits from **state conditioning**, not specifically
    whether it recognizes A. Even A's observations and realized actions can change
    under the replacement. Equal market seeds are paired starting conditions, not
    identical counterfactual market paths.

    The earlier sequential pilot froze A before B learned, so it could not establish
    mutual learning. B's deterministic mean action could look inactive while sampled
    actions still quoted. We retain that pilot as context, not another replication.
    """, "subslide")
    code("""
    seq_means = sequential.groupby(['scenario', 'agent'])[
        ['pnl_dollars', 'any_quote_pct']].mean()
    display(seq_means.rename(index=AGENT, level='agent'))
    print('Sequential controls: one trained pair, six exploratory markets; not confirmation data.')
    """)
    md(r"""
    ## 3 · A close-time correction overturned the early candidate

    A **value bootstrap** adds an estimated future value to a training target.
    If a batch stops while the session continues, this makes sense. But normal
    exchange close is the end of the task: no post-close reward exists. The legacy
    trainer bootstrapped at close as if it were an external timeout. The corrected
    finite-session target uses **zero continuation at close**, while retaining the
    bootstrap at open-session batch boundaries.

    The seed-17 comparison began with matching initial model tensors, environment
    configuration and training market-seed stream. The initial-pair evaluation
    reproduced exactly in the archived audit. Both policies then changed under
    retraining. The correction changed targets, **not saved evaluation cash**.

    Below, each dot is one exploratory market's learned-B minus state-blind-B P&L.
    Positive means better relative to the replacement, not necessarily profitable.
    """)
    code("""
    def paired_contrasts(frame, metric='pnl_dollars', groups=('pair',)):
        subset = frame[frame.scenario.isin(CONDITION)]
        keys = [*groups, 'agent', 'seed', 'replicate']
        wide = subset.pivot(index=keys, columns='scenario', values=metric)
        assert wide.notna().all().all(), 'A paired condition is missing.'
        delta = (wide['sampled_pair'] - wide['marginal_b']).rename('difference').reset_index()
        # Repeats are NOT extra independent markets: average within market first.
        return delta.groupby([*groups, 'agent', 'seed'], as_index=False).difference.mean()

    def bootstrap_interval(values):
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
        return np.quantile(means, [.025, .975])

    horizon['target'] = np.where(horizon.training_run.str.contains('finite_seed'),
                                 'Finite close', 'Legacy close')
    close_delta = paired_contrasts(horizon, groups=('target',))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, agent, color in zip(axes, AGENT, [A_COLOR, B_COLOR], strict=True):
        for x, target in enumerate(['Legacy close', 'Finite close']):
            vals = close_delta.loc[(close_delta.agent == agent) &
                                   (close_delta.target == target), 'difference'].to_numpy()
            assert len(vals) == 6
            ax.scatter(x + np.linspace(-.12, .12, len(vals)), vals, color=color, alpha=.7)
            ax.plot([x - .2, x + .2], [vals.mean()] * 2, color=color, lw=3)
        ax.axhline(0, color='#555555', lw=1)
        ax.set(xticks=[0, 1], xticklabels=['Legacy close', 'Finite close'],
               title=f'Trader {AGENT[agent]} · exploratory',
               ylabel='Paired marked P&L difference ($)')
    fig.suptitle('The early A candidate did not survive finite-close retraining')
    plt.show()
    display(close_delta.groupby(['target', 'agent']).difference.agg(
        mean='mean', positive_markets=lambda x: (x > 0).sum()))
    """)
    md("""
    ### Replications constrain the story before held-out testing

    Corrected pairs 17, 29, and 43 each completed the same training budget. The next
    chart retains **all three** corrected exploratory pairs. Intervals resample six
    markets within each fixed pair; they do not measure training-run uncertainty.
    These exploratory results did not establish a consistent state-conditioning
    benefit. Nor does an interval including zero establish equivalence.
    """, "subslide")
    code("""
    def contrast_plot(delta, title, market_count):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, agent, color in zip(axes, AGENT, [A_COLOR, B_COLOR], strict=True):
            for x, pair in enumerate(sorted(delta.pair.unique())):
                vals = delta.loc[(delta.agent == agent) & (delta.pair == pair),
                                 'difference'].to_numpy()
                assert len(vals) == market_count
                low, high = bootstrap_interval(vals)
                ax.vlines(x, low, high, color=color, lw=2, alpha=.8)
                ax.scatter(x + np.linspace(-.16, .16, len(vals)), vals,
                           color=color, s=22, alpha=.55)
                ax.plot([x - .2, x + .2], [vals.mean()] * 2, color=color, lw=3)
            ax.axhline(0, color='#555555', lw=1)
            ax.set(xticks=range(delta.pair.nunique()),
                   xticklabels=[str(p) for p in sorted(delta.pair.unique())],
                   xlabel='Training pair seed',
                   ylabel=f"{AGENT[agent]}'s marked P&L difference ($)",
                   title=f'{AGENT[agent]} · ' + ('secondary' if agent == 'mm_0' else 'primary'))
        fig.suptitle(title + '\\nB learned minus B state-blind · dots: markets · bars: means · '
                     'lines: descriptive 95% market bootstrap')
        plt.show()

    exploratory_delta = paired_contrasts(exploratory)
    contrast_plot(exploratory_delta, 'Corrected exploratory comparisons · separate panel scales', 6)
    """)
    md("""
    ## 4 · Held-out comparison: what is an observation?

    Three frozen trained pairs × twelve new markets × two action draws × two
    conditions = **144 joint episodes**, or **288 trader trajectories**. B's marked
    P&L contrast was the primary outcome; A and behavior are also reported. The
    recorded protocol reserved markets 40001–40012 before this evaluation.

    For each trader and pair:

    1. Subtract state-blind from learned B **within each market and action draw**.
    2. Average the two draw differences within that market.
    3. Describe the twelve market differences and bootstrap **markets**, not steps.

    There are still only **three training replications**, not 144 or thousands of
    independent decisions. Intervals are conditional on a fixed pair and this
    market sampling design. They are descriptive and not adjusted for multiple
    comparisons. Pair 29 is not a universal result or a newly confirmed mechanism.
    """)
    code("""
    KEYS = ['pair', 'scenario', 'seed', 'replicate', 'agent']
    expected = pd.MultiIndex.from_product(
        [[17, 29, 43], ['sampled_pair', 'marginal_b'], range(40001, 40013),
         [0, 1], ['mm_0', 'mm_1']], names=KEYS)
    assert len(confirmation) == 288
    assert not confirmation.duplicated(KEYS).any()
    actual = pd.MultiIndex.from_frame(confirmation[KEYS])
    assert actual.sort_values().equals(expected.sort_values())
    assert len(behavior) == 288 and not behavior.duplicated(KEYS).any()
    assert pd.MultiIndex.from_frame(behavior[KEYS]).sort_values().equals(expected.sort_values())
    assert not confirmation.risk_terminated.any()
    assert (confirmation.steps == confirmation.common_horizon_steps).all()
    np.testing.assert_allclose(confirmation.pnl_dollars,
                               confirmation.common_horizon_pnl_dollars, atol=1e-8)
    np.testing.assert_allclose(
        confirmation.inventory_marking_dollars + confirmation.execution_vs_step_end_mid_dollars,
        confirmation.pnl_dollars, atol=1e-8)
    assert (confirmation.gross_fill_cash_residual_cents == 0).all()
    np.testing.assert_allclose(behavior.bought_shares - behavior.sold_shares,
                               behavior.final_inventory, atol=1e-8)
    joined = confirmation.merge(behavior, on=KEYS, suffixes=('_summary', '_behavior'),
                                validate='one_to_one')
    for metric in ['pnl_dollars', 'final_inventory', 'mean_abs_inventory']:
        np.testing.assert_allclose(joined[metric + '_summary'], joined[metric + '_behavior'])
    print('Verified complete 3 × 12 × 2 × 2 × 2 evidence grid; no risk stops.')
    print('Summary equity decomposition and share-flow accounting agree for all 288 rows.')
    print('These are summary checks; independent raw-file accounting follows below.')

    heldout_delta = paired_contrasts(confirmation)
    rows = []
    for (pair, agent), group in heldout_delta.groupby(['pair', 'agent']):
        values = group.difference.to_numpy()
        low, high = bootstrap_interval(values)
        rows.append({'pair': pair, 'trader': AGENT[agent], 'mean_difference_$': values.mean(),
                     'lower_95%_$': low, 'upper_95%_$': high,
                     'positive_markets': int((values > 0).sum()), 'markets': len(values)})
    heldout_table = pd.DataFrame(rows)
    display(heldout_table)
    contrast_plot(heldout_delta, 'Held-out comparisons · separate panel scales', 12)
    """)
    md("""
    ## 5 · Seed 29 B: lower losses, not profit

    The pair-29 B contrast is positive, but its learned-policy mean is still
    **negative**. Check the absolute outcomes alongside the difference. No unfavorable
    pair or action draw is removed. A's contrast does not establish that B benefited
    at A's expense.

    In the chart, each dot averages two draws in one market. Gray lines connect
    paired markets, but do **not** imply the realized price paths stayed identical.
    """)
    code("""
    b29 = confirmation[(confirmation.pair == 29) & (confirmation.agent == 'mm_1')]
    b29_market = b29.groupby(['seed', 'scenario']).pnl_dollars.mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in b29_market.iterrows():
        ax.plot([0, 1], [row['marginal_b'], row['sampled_pair']], color='#cccccc', lw=1)
    for x, condition, color in [(0, 'marginal_b', POOL_COLOR), (1, 'sampled_pair', LEARNED_COLOR)]:
        values = b29_market[condition]
        ax.scatter(np.full(len(values), x), values, color=color, s=35, zorder=3)
        ax.plot([x - .15, x + .15], [values.mean()] * 2, color=color, lw=4)
    ax.axhline(0, color='#555555', lw=1)
    ax.set(xticks=[0, 1], xticklabels=['State-blind B', 'Learned B'],
           ylabel='B final marked P&L ($)', title='Pair 29 B · absolute held-out outcomes')
    plt.show()
    absolute_metrics = ['pnl_dollars', 'mean_abs_inventory', 'final_inventory']
    display(b29.groupby('scenario')[absolute_metrics].mean())
    draw_wide = confirmation.pivot(index=['pair', 'agent', 'seed', 'replicate'],
                                  columns='scenario', values='pnl_dollars')
    draw_delta = (draw_wide.sampled_pair - draw_wide.marginal_b).rename('difference').reset_index()
    display(draw_delta[draw_delta.agent == 'mm_1'].groupby(['pair', 'replicate']).difference.agg(
        mean_difference='mean', positive_markets=lambda x: (x > 0).sum()))
    # Rounded report anchors: detect accidental condition reversal or unit changes.
    b29_difference = heldout_table.query("pair == 29 and trader == 'B'").iloc[0]
    assert np.isclose(b29_difference['mean_difference_$'], 838.70, atol=.01)
    assert b29_difference.positive_markets == 9
    assert b29_difference['lower_95%_$'] > 0
    assert b29.groupby('scenario').pnl_dollars.mean().lt(0).all()
    """)
    md(r"""
    ## 6 · Inventory marking versus execution-relative accounting

    Let cash be $C_t$, shares $q_t$, and the recorded midpoint $m_t$.
    Marked equity is $E_t=C_t+q_tm_t$. Its change splits exactly into

    $$E_t-E_{t-1}=q_{t-1}(m_t-m_{t-1})+
       [(C_t-C_{t-1})+(q_t-q_{t-1})m_t].$$

    The first term is **inventory marking**: shares already held gain/lose value
    as the recorded midpoint changes. The second values new share flow at the
    step-ending midpoint relative to cash paid/received. It is **not pure spread
    capture**: fills and snapshots occur at different times within a step.

    Neither term proves causal **adverse selection** (trading with someone whose
    information predicts a subsequent unfavorable price move). That needs execution-
    time information and price-response evidence, not just this accounting identity.
    """)
    code("""
    components = ['inventory_marking_dollars', 'execution_vs_step_end_mid_dollars']
    component_delta = pd.DataFrame({
        metric: paired_contrasts(confirmation, metric).query(
            "pair == 29 and agent == 'mm_1'").set_index('seed').difference
        for metric in components})
    total = heldout_delta.query("pair == 29 and agent == 'mm_1'").set_index('seed').difference
    np.testing.assert_allclose(component_delta.sum(axis=1), total, atol=1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    means = b29.groupby('scenario')[components].mean().reindex(['marginal_b', 'sampled_pair'])
    means.rename(columns={components[0]: 'Inventory marking',
                          components[1]: 'Execution vs step-end mid'}, index=CONDITION).plot.bar(
        ax=axes[0], color=[B_COLOR, '#92999e'], rot=0)
    axes[0].set(title='Pair 29 B · absolute components',
                ylabel='Mean marked P&L component ($)', xlabel='')
    axes[0].axhline(0, color='#555555', lw=1)
    difference_means = component_delta.mean()
    axes[1].bar(['Inventory\\nmarking', 'Execution vs\\nstep-end mid'], difference_means,
                color=[B_COLOR, '#92999e'])
    axes[1].axhline(0, color='#555555', lw=1)
    axes[1].set(title='Learned minus state-blind B', ylabel='Mean paired component difference ($)')
    plt.show()
    display(difference_means.to_frame('mean_difference_$'))
    print('Accounting describes exposure paths; it does not identify why those paths differed.')
    """)
    md("""
    ### Independent raw-file checks, without loading policies

    The next cell reads only numeric arrays with `allow_pickle=False`. For every
    held-out trader trajectory it checks gross fills against cash and share changes,
    reconstructs marked P&L, and recomputes both accounting components. It also
    compares bought/sold totals with the behavior CSV. A missing raw export is a
    clear blocker rather than a silently skipped check.

    Raw cash and midpoint units are **cents**; shares are signed integers. The
    conversion to dollars occurs only at presentation time. The audit uses gross
    buy/sell fill arrays, not netted per-step fills that could hide offsetting trades.
    """, "subslide")
    code("""
    def raw_path(row):
        return (ROOT / 'data/confirmation' / f'seed{row.pair}' / 'eval' / row.scenario /
                f'seed{row.seed}_rep{row.replicate}' / f'{row.agent}.npz')

    def read_numeric_trajectory(path):
        keys = ['timestamps', 'cash', 'inventory', 'mid_price',
                'fill_side', 'fill_price', 'fill_quantity']
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in keys}

    raw_rows = []
    behavior_index = behavior.set_index(KEYS)
    for row in confirmation.itertuples(index=False):
        path = raw_path(row)
        if not path.is_file():
            raise FileNotFoundError(f'Missing public raw evidence: {path}')
        raw = read_numeric_trajectory(path)
        cash, q, mid = (raw[key].astype(np.int64) for key in ['cash', 'inventory', 'mid_price'])
        side, price, qty = (raw[key].astype(np.int64)
                            for key in ['fill_side', 'fill_price', 'fill_quantity'])
        assert len(cash) == len(q) == len(mid) == row.steps + 1
        assert q[0] == 0, 'Share-flow comparison assumes a flat opening account.'
        assert set(np.unique(side)).issubset({-1, 1}) and (qty > 0).all()
        assert cash[-1] - cash[0] == -(side * price * qty).sum()
        assert q[-1] - q[0] == (side * qty).sum()
        assert len(qty) == row.gross_fills and qty.sum() == row.gross_shares
        marking = (q[:-1] * np.diff(mid)).sum() / 100
        execution = (np.diff(cash) + np.diff(q) * mid[1:]).sum() / 100
        pnl = ((cash[-1] + q[-1] * mid[-1]) - (cash[0] + q[0] * mid[0])) / 100
        np.testing.assert_allclose([pnl, marking, execution, marking + execution],
                                   [row.pnl_dollars, row.inventory_marking_dollars,
                                    row.execution_vs_step_end_mid_dollars, row.pnl_dollars],
                                   atol=1e-8)
        b = behavior_index.loc[(row.pair, row.scenario, row.seed, row.replicate, row.agent)]
        np.testing.assert_allclose([qty[side == 1].sum(), qty[side == -1].sum(), q[-1]],
                                   [b.bought_shares, b.sold_shares, b.final_inventory])
        raw_rows.append({'path': str(path.relative_to(ROOT)), 'pnl_dollars': pnl})
    raw_audit = pd.DataFrame(raw_rows)
    assert len(raw_audit) == 288
    print(f'Independently reconciled {len(raw_audit)} raw trader trajectories.')
    """)
    md("""
    ### One exposure path, not a representative performance result

    The example below is **pair 29, first held-out market 40001, draw 0, learned B**.
    It is selected by that fixed identifier, not a performance ranking. One session
    cannot stand in for the market-level results above. The price and inventory
    panels illustrate why marked equity can move between executions; separate axes
    avoid visually manufacturing a correlation with arbitrary dual-axis scaling.
    """, "subslide")
    code("""
    example_row = confirmation.query(
        "pair == 29 and scenario == 'sampled_pair' and seed == 40001 and "
        "replicate == 0 and agent == 'mm_1'").iloc[0]
    example = read_numeric_trajectory(raw_path(example_row))
    elapsed = example['timestamps'] - example['timestamps'][0]
    q, cash, mid = (example[key].astype(np.int64) for key in ['inventory', 'cash', 'mid_price'])
    equity_change = (cash + q * mid - (cash[0] + q[0] * mid[0])) / 100
    marking_path = np.r_[0, np.cumsum(q[:-1] * np.diff(mid))] / 100
    execution_path = np.r_[0, np.cumsum(np.diff(cash) + np.diff(q) * mid[1:])] / 100
    np.testing.assert_allclose(equity_change, marking_path + execution_path, atol=1e-8)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(elapsed, mid / 100, color='#555555')
    axes[0].set(ylabel='Midpoint ($)',
                title='Illustrative session only · pair 29 / market 40001 / draw 0 / B')
    axes[1].step(elapsed, q, where='post', color=B_COLOR)
    axes[1].axhline(0, color='#555555', lw=1)
    axes[1].set(ylabel='Inventory (shares)')
    for values, label, color in [(equity_change, 'Marked P&L', A_COLOR),
                                  (marking_path, 'Inventory marking', B_COLOR),
                                  (execution_path, 'Execution vs step-end mid', '#92999e')]:
        axes[2].plot(elapsed, values, label=label, color=color)
    axes[2].set(ylabel='Cumulative change ($)', xlabel='Seconds since first recorded snapshot')
    axes[2].legend(loc='best')
    plt.show()
    """)
    md("""
    ## 7 · Sparse quoting can still build directional inventory

    A quote count is not a fill count. The next charts keep **all three learned pairs
    and both traders**, not just the most striking case. Market dots average two
    action draws; bars and the any-quote diamonds average markets. Bid/ask offsets are requested
    distances conditional on enabling that side, before extra half-tick rounding.

    Follow the chain: side participation → requested prices → actual shares bought
    and sold → ending inventory. Describing this chain does not prove a deliberate
    price forecast or identify the causal effect of one quote parameter.
    """)
    code("""
    learned_behavior = behavior[behavior.scenario == 'sampled_pair'].merge(
        confirmation[KEYS + ['any_quote_pct']], on=KEYS, validate='one_to_one')
    behavior_metrics = ['any_quote_pct', 'bid_gate_pct', 'ask_gate_pct',
                        'bid_active_offset_cents', 'ask_active_offset_cents',
                        'bought_shares', 'sold_shares', 'final_inventory']
    behavior_markets = learned_behavior.groupby(['pair', 'agent', 'seed'])[behavior_metrics].mean()
    behavior_means = behavior_markets.groupby(['pair', 'agent']).mean()
    display(behavior_means.rename(index=AGENT, level='agent'))
    labels = [f'{pair} / {AGENT[agent]}' for pair, agent in behavior_means.index]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for metric, offset, label, color in [('bid_gate_pct', -.18, 'Bid enabled', A_COLOR),
                                          ('ask_gate_pct', .18, 'Ask enabled', B_COLOR)]:
        axes[0].bar(x + offset, behavior_means[metric], width=.32,
                    label=label, color=color, alpha=.8)
    axes[0].scatter(x, behavior_means.any_quote_pct, marker='D', color='#333333', label='Any quote')
    axes[0].set(ylabel='Decisions (%)', ylim=(0, 105),
                title='Learned pairs · held-out quote participation')
    axes[0].legend(ncol=3)
    for metric, offset, label, color in [('bought_shares', -.18, 'Bought', A_COLOR),
                                          ('sold_shares', .18, 'Sold', B_COLOR)]:
        axes[1].bar(x + offset, behavior_means[metric], width=.32,
                    color=color, alpha=.7, label=label)
        for i, key in enumerate(behavior_means.index):
            vals = behavior_markets.loc[key, metric].to_numpy()
            axes[1].scatter(i + offset + np.linspace(-.09, .09, len(vals)), vals,
                            s=12, color=color, alpha=.55)
    axes[1].set(xticks=x, xticklabels=labels, xlabel='Training pair / trader',
                ylabel='Shares per session', title='Actual gross flows · dots are market averages')
    axes[1].legend()
    plt.show()
    """)
    code("""
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, key in enumerate(behavior_means.index):
        values = behavior_markets.loc[key, 'final_inventory'].to_numpy()
        color = A_COLOR if key[1] == 'mm_0' else B_COLOR
        ax.scatter(i + np.linspace(-.15, .15, len(values)), values, color=color, alpha=.55)
        ax.plot([i - .2, i + .2], [values.mean()] * 2, color=color, lw=3)
    ax.axhline(0, color='#555555', lw=1)
    ax.set(xticks=x, xticklabels=labels, xlabel='Training pair / trader',
           ylabel='Ending inventory (shares; + long, − short)',
           title='Sparse quotes do not imply flat inventory · learned held-out pairs')
    plt.show()
    a43 = behavior_means.loc[(43, 'mm_0')]
    print(f'Pair 43 A: any quote {a43.any_quote_pct:.2f}%; '
          f'bought {a43.bought_shares:.2f}, sold {a43.sold_shares:.2f}; '
          f'ending inventory {a43.final_inventory:+.2f} shares.')
    b29_behavior = behavior[behavior.pair.eq(29) & behavior.agent.eq('mm_1')].merge(
        confirmation[KEYS + ['any_quote_pct']], on=KEYS, validate='one_to_one')
    display(b29_behavior.groupby('scenario')[behavior_metrics].mean())
    print('Pair 29 B quotes slightly more often when learned, yet receives fewer shares.')
    """)
    md("""
    ## 8 · What remains unidentified?

    - Only three training pairs, short sessions, and a limited training budget.
    - The replacement removes responses to **own inventory, time, and background
      flow**, not only responses to the rival.
    - Same seeds pair starting conditions; changing orders changes later paths.
    - Large background liquidity may make learner influence modest. Direct A–B
      executions are rare; their absence does not rule out indirect competition.
    - Marked wealth omits close liquidation, fees, financing, and realistic capital
      constraints. The reward is not a calibrated inventory-risk objective.
    - Conditional market-bootstrap intervals cannot quantify generalization over
      new training runs. No step-level significance tests are appropriate here.

    **Not established:** exploitation, collusion, recognition of the rival, mutual
    adaptation, robust profitability, or that reinforcement learning cannot trade.
    """)
    code("""
    # A direct match is reported for both traders: count each joint episode once.
    episode_keys = ['pair', 'scenario', 'seed', 'replicate']
    assert confirmation.groupby(episode_keys).direct_pair_matches.nunique().eq(1).all()
    episode_matches = confirmation.groupby(episode_keys).direct_pair_matches.first()
    display(episode_matches.groupby(['pair', 'scenario']).sum().unstack())
    print('Total direct pair matches:', int(episode_matches.sum()))
    print('A small direct-match count does not measure all indirect book influence.')
    """)
    md(r"""
    ### Next test: does B's state-conditioning benefit require A?

    **Proposed, not executed.** Freeze pair 29 and compare four conditions on fresh
    markets: learned/pool B crossed with learned/inactive A. Keep the **same original
    B pool** in all conditions and private role random streams. Predeclare outcomes,
    average action repeats within market, and report both constituent contrasts.

    The primary comparison would be:

    $$(B_{learned}-B_{pool})_{A\ present}
      -(B_{learned}-B_{pool})_{A\ inactive}.$$

    A similar B benefit without A would point toward broader state-conditioned
    trading. A reproducible difference would establish opponent **dependence**, not
    automatically recognition or exploitation. Removing A also changes liquidity.
    Selecting pair 29 after seeing its result is hypothesis generation; reusing
    these markets would not be fresh confirmation.

    See [`next_research_stage.md`](../notes/next_research_stage.md). If the mechanism
    remains unclear, a smaller scripted competition positive control can first
    verify that changing a rival's quotes measurably changes fill allocation.
    """, "subslide")
    md("""
    ## 9 · Takeaways and open questions

    The strongest angles are methodological and behavioral. The next cell makes a
    compact evidence ledger directly from the loaded data. Each finding pairs a
    finding with its boundary; figures above supply the receipts.

    **Discussion prompts**

    1. What would distinguish inventory-aware state conditioning from a response
       specifically to the other learner?
    2. Why can a mostly inactive-looking quoting policy still accumulate exposure?
    3. Which economic assumptions would have to change before calling this profit?
    4. How should we report a promising single-pair contrast without hiding the
       other training pairs or pretending the finding is universal?
    """)
    code("""
    close_means = close_delta.groupby(['target', 'agent']).difference.mean()
    b29_absolute = b29.groupby('scenario').pnl_dollars.mean()
    marking_difference = component_delta[components[0]].mean()
    old_a = close_means.loc[('Legacy close', 'mm_0')]
    corrected_a = close_means.loc[('Finite close', 'mm_0')]
    ledger = pd.DataFrame([
        {'angle': 'A correction that changed the conclusion',
         'finding': f"An early A contrast went from ${old_a:+.2f} "
                  f"to ${corrected_a:+.2f} after correcting "
                  'close-time training targets. Same seed-17 comparison; both learners retrained.',
         'boundary': 'Exploratory pair-specific change, not an isolated opponent mechanism.'},
        {'angle': 'Less loss is not profit',
         'finding': f"Held-out pair 29 B: learned mean ${b29_absolute['sampled_pair']:,.2f}, "
                  f"state-blind mean ${b29_absolute['marginal_b']:,.2f}; paired benefit "
                  f"${b29_difference['mean_difference_$']:,.2f}, positive in "
                  f"{int(b29_difference.positive_markets)}/12 markets.",
         'boundary': 'One of three pairs; state conditioning is not rival recognition.'},
        {'angle': 'Quote counts can hide directional exposure',
         'finding': f"Pair 43 A quoted on {a43.any_quote_pct:.2f}% of decisions but bought "
                  f"{a43.bought_shares:.2f} and sold {a43.sold_shares:.2f} shares/session, "
                  f"ending {a43.final_inventory:+.2f} shares on average.",
         'boundary': 'Behavioral description, not proof of deliberate forecasting.'},
        {'angle': 'Accounting before a mechanism story',
         'finding': f"Of pair 29 B's ${b29_difference['mean_difference_$']:,.2f} paired benefit, "
                  f"${marking_difference:,.2f} is inventory-marking difference.",
         'boundary': 'Descriptive identity; not causal adverse selection or spread capture.'},
    ])
    display(ledger)
    for row in ledger.itertuples(index=False):
        print(f'\\n{row.angle}\\n{row.finding}\\nLimit: {row.boundary}')
    """)
    md("""
    ### Closing: a narrower result is still a useful result

    **Supported:** independent policies changed behavior; evaluation sampling matters;
    finite-close targets changed the early candidate; quote rates hid directional
    positions; and one B policy had a held-out, pair-specific lower-loss result
    relative to a state-blind replacement, largely associated with inventory marking.

    **Not supported:** “two AIs learned to exploit each other,” “emergent collusion,”
    "profitable autonomous market makers," or treating exploratory clues as confirmed
    discoveries. The full results retain the loss baseline and all three pairs.

    The next useful question is whether the benefit **requires A's presence**.
    That experiment is a proposal, not a result in this notebook.

    ---

    **Reproduction:** open from the repository root or `notebooks/`, select an
    environment with NumPy/pandas/Matplotlib and Jupyter, and run all cells. Numeric
    evidence is read-only. The builder is `scripts/build_research_notebook.py`;
    `--check` verifies deterministic source cells while allowing reviewed outputs.
    Bootstrap calculations use the report analysis script's fixed seed and draw
    count. Reviewed executed output is included without changing the inputs.

    Earlier desktop experiments are preserved separately in
    [`data/historical/`](../data/historical/README.md). They have known validity
    limits and are not pooled with this corrected study.
    """, "subslide")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": "Two learning traders, one synthetic order book",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_notebook() -> str:
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


def source_signature(notebook: dict) -> list:
    """Ignore execution outputs/metadata, including nbformat's string-to-list rewrite."""
    return [
        (
            cell["cell_type"],
            cell["id"],
            "".join(cell["source"]),
            cell["metadata"]["slideshow"],
        )
        for cell in notebook["cells"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check source; allow reviewed executed outputs."
    )
    args = parser.parse_args()
    rendered = render_notebook()
    if args.check:
        current = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.is_file() else None
        if current is None or source_signature(current) != source_signature(build_notebook()):
            print(f"Out of date: {OUTPUT}; run {Path(__file__).name} to rebuild.")
            return 1
        print(f"Up to date: {OUTPUT}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
