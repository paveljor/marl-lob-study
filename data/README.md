# Study data

These are **synthetic ABIDES simulation artifacts**, not real exchange data or
real investors' accounts. The corrected confirmation data is committed here so
the reported accounting, plots, and comparisons can be checked without training
new policies or simulating new markets.

## What is included

| Location | Contents | Role |
| --- | --- | --- |
| `confirmation/seed17/`, `seed29/`, `seed43/` | Complete corrected held-out evaluations and separate calibration sessions | Primary raw evidence |
| `training/finite_seed17/`, `finite_seed29/`, `finite_seed43/` | Training metadata, diagnostics, initial/final models, and historical source snapshots | Training and implementation provenance |
| `manifest.json` | Original/exported hashes, sizes, and transformation records | Byte-identity checks |
| `historical/` | Five separately labeled desktop summary tables | Earlier research history, not confirmation |
| `../notes/assets/` | Derived tables, analysis JSON, and figures | Report and walkthrough inputs |

The confirmation study has three trained pairs × twelve markets × two action
draws × two conditions = **144 joint evaluations / 288 agent trajectories**.
Four separate calibration markets per pair add 24 agent trajectories. There
are **312 numeric NPZ trajectory files** altogether. Action draws are averaged
within market for the reported contrasts; they are not independent training
replications. The two conditions are `sampled_pair` and `marginal_b`.

## Read the arrays safely

```python
import numpy as np

with np.load(
    "data/confirmation/seed29/eval/sampled_pair/seed40001_rep0/mm_1.npz",
    allow_pickle=False,
) as episode:
    inventory = episode["inventory"]       # integer shares; 600 account snapshots
    cash = episode["cash"]                 # integer cents
    midpoint = episode["mid_price"]        # integer cents
    actions = episode["actions"]           # 599 decisions × 6 action fields
    observations = episode["observations"] # 600 snapshots × 48 features
    equity_dollars = (cash + inventory * midpoint) / 100
```

`timestamps` are simulated seconds. `fill_side`, `fill_quantity`, `fill_price`,
and `fill_timestamps` record gross executions separately from account snapshots.
Cash and prices are integer cents; convert to dollars only for presentation.
The saved actions contain gates, offsets, and fixed sizes. An enabled quote is
an offer, not a completed trade.

`market.json` records attempted submissions, received executions, and exchange
matches. Its IDs identify simulated agents, not human traders. An execution's
received time need not equal exchange-match time. Direct A–B counts appear in
both agents' summary rows: count one role per joint episode, not both.

The state-blind B pool is built from B's complete action rows on four separate
calibration markets. It preserves their joint action distribution, not its
connection to the current state, realized fill count, or exact quote timing.

## Recompute the reported analyses

After installing the project as described in the README, run from the repo root:

```bash
python scripts/verify_study_data.py

python scripts/analyse_interaction.py \
  data/confirmation/seed17 data/confirmation/seed29 data/confirmation/seed43 \
  --out-prefix outputs/confirmation

python scripts/describe_trader_behavior.py \
  data/confirmation/seed17 data/confirmation/seed29 data/confirmation/seed43 \
  --scenarios sampled_pair marginal_b --out outputs/behavior.csv
```

These commands read saved arrays; they do not train or simulate. They reconcile
gross executions against cash/inventory and the marked-equity decomposition.
Their full environment imports include SB3 and ABIDES; the walkthrough's direct
NumPy/Pandas/Matplotlib calculations do not need to instantiate those components.

## Provenance and safety

- Raw trajectories, action pools, execution logs, summary CSVs, historical source
  ZIPs, and model ZIPs preserve their original bytes. Do not edit them in place.
- Evaluation metadata uses archive-relative parent identifiers. Those identifiers
  describe the original run hierarchy, not the public data directory layout.
- Editable local-install locations in historical package listings are redacted.
  Other version information is retained. These listings are evidence, not
  installation lockfiles to execute blindly.
- `manifest.json` records original and exported hashes and each transformation.
  Supplemental source-member manifests cover simulator files omitted from the
  older manifest format without changing those original manifests or ZIPs.
- Source snapshots predate the public cleanup. Their original implementation is
  distinct from today's analysis code; the original private Git history is not
  imported. Hashes show byte identity, not independent proof of preregistration.
- **Model ZIPs contain serialized Python objects.** Do not load untrusted model
  archives. The walkthrough does not deserialize models, and viewing their
  hashes does not require loading them.
- Full upstream notices are retained alongside this data. See
  `THIRD_PARTY_NOTICES.md`; the project's MIT license does not replace ABIDES'
  terms.

This release includes the corrected confirmation/calibration raw evidence, not
every earlier pilot, exploratory trajectory, or intermediate training checkpoint.
The earlier desktop tables are quarantined in `historical/` with their known
validity limits. No new research run was launched to assemble this dataset.
