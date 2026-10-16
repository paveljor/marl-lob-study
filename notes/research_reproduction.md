# Reproducing the two-trader study

The original study completed on 2026-09-26. This guide was curated for public
sharing on 2026-10-14. Read [the report](research_report.md) for interpretation,
[the protocol](concurrent_protocol.md) for the recorded design, and
[source provenance](../PROVENANCE.md) for the relationship to the original archive.

## Environment and source provenance

Run commands from the repository root in the environment described in the
README. Python 3.11 is the verified research interpreter. The original checked
environment used Python 3.11.15, NumPy 2.4.4, Pandas 3.0.2, PyTorch 2.11.0,
Stable-Baselines3 2.8.0, Gymnasium 1.2.3, PettingZoo 1.26.1, and SuperSuit 3.10.0.
Those are observed historical research versions, not the public setup constraints.
Sanitized package listings for the corrected training/confirmation runs are
included in `data/`. Their local editable-install locators are removed; the
listings remain historical evidence, not installation commands.

The public Linux/Python 3.11 setup was separately validated in a fresh virtual
environment on 2026-10-14, with all 369 original tests passing; subsequent
source-archive regression coverage brings the passing suite to 383. The README uses
`requirements/linux-py311-cpu.txt` to pin that environment. Passing on newer
dependencies does not establish exact replay of the original trained policies.

Training uses CPU and one PyTorch thread. No GPU is required.

```bash
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
```

ABIDES is pinned to `f9cbe51342b7dedd9587e4e069040d68a5c6477f`, with the
two required changes in [patches/README.md](../patches/README.md). Pin before
applying the patch. Install the local `abides-core`, `abides-markets`, and this
project into the same environment. The README supplies setup commands.

**Historical source snapshots record what actually ran.** The original archive
contains `source_snapshot.zip`, `source_manifest.json`, `pip-freeze.txt`, and
`run_meta.json`. Corrected training/confirmation copies are included under
`data/`, with metadata sanitization recorded in its manifest. A source ZIP records project
Python files and the two patched ABIDES files. Editable Git entries in a package
listing are not enough to reconstruct a run: the executed tree may have contained
newer edits. Original commit IDs may not resolve in this fresh public history.
Do not substitute this cleaned source tree for a historical snapshot when
claiming an exact replay. Restore archived source into a separate checkout and
verify it against its manifest; never overwrite an existing run or active tree.

## Train and evaluate a fresh corrected pair

To recompute the existing confirmation results without new experiments, use
[the committed raw data](../data/README.md) or the
[executed walkthrough](../notebooks/research_walkthrough.ipynb). The commands
below are for generating new training and evaluation runs.

These commands create new runs; they do not require unpublished checkpoints.
Output names must not already exist. The included configuration records the
market settings used in the corrected study.

```bash
python scripts/train_concurrent.py \
  --config-from experiments/concurrent_base.json \
  --out-dir runs/concurrent/reproduce_finite_seed17 \
  --seed 17 --steps 32768 --quote-size 10 --n-steps 256 --save-every 8192

python scripts/evaluate_interaction.py runs/concurrent/reproduce_finite_seed17 \
  --out-dir runs/concurrent/reproduce_finite_seed17_exploratory \
  --calibration-seeds 20001 20002 20003 20004 \
  --market-seeds 30001 30002 30003 30004 30005 30006 \
  --replicates 0 \
  --scenarios sampled_pair mean_pair initial_pair initial_b marginal_b inactive_b
```

Use seeds 29 and 43 with distinct output directories for the other training
replications. Repeating seed 17 checks reproducibility; it is not another
independent training seed. Each trader receives 32,768 samples from the same
32,768 joint market steps, not 65,536 market transitions.

The corrected default adds no continuation value at exchange close.
`--legacy-close-bootstrap` reproduces the former concurrent training target;
label its results separately. The sequential pilot predates the corrected
single-learner wrapper: reproducing its historical training requires its
archived implementation rather than today's defaults.

Originally, corrected seed 17 training took 852.9 seconds, and its exploratory
evaluation plus calibration took 454.3 seconds. These are observed local times,
not performance guarantees for another machine or dependency environment.

## Reproduce the held-out comparison

After the fresh training above, evaluate the same two predeclared conditions on
the twelve confirmation markets with two action draws:

```bash
python scripts/evaluate_interaction.py runs/concurrent/reproduce_finite_seed17 \
  --out-dir runs/concurrent/reproduce_finite_seed17_confirmation \
  --calibration-seeds 20001 20002 20003 20004 \
  --market-seeds 40001 40002 40003 40004 40005 40006 40007 40008 40009 40010 40011 40012 \
  --replicates 0 1 --scenarios sampled_pair marginal_b
```

Repeat for freshly trained seeds 29 and 43 using corresponding directory names.
This is 48 joint evaluations per pair, plus four calibration sessions. Analysis
averages action draws within each market before paired contrasts and bootstrap
intervals. Keep per-draw results visible. These markets are now used evidence;
a new follow-up experiment requires new markets.

## Analyze freshly generated results

Analysis does not train policies or launch market simulations. It requires the
raw evaluation directories created above; committed derived assets alone do not
provide those inputs.

```bash
python scripts/analyse_interaction.py \
  runs/concurrent/reproduce_finite_seed17_confirmation \
  --out-prefix notes/assets/reproduce_seed17_confirmation

python scripts/plot_interaction_contrasts.py \
  notes/assets/reproduce_seed17_confirmation.json \
  --out notes/assets/reproduce_seed17_confirmation_contrasts.png

python scripts/describe_trader_behavior.py \
  runs/concurrent/reproduce_finite_seed17_confirmation \
  --scenarios sampled_pair marginal_b \
  --out notes/assets/reproduce_seed17_confirmation_behavior.csv

python scripts/plot_trader_behavior.py \
  notes/assets/reproduce_seed17_confirmation_behavior.csv \
  --out notes/assets/reproduce_seed17_confirmation_behavior.png
```

Pass all three freshly generated confirmation directories to the first and
third commands to combine training replications. The analysis reconciles gross
executions with cash and inventory and checks the exact marked-equity
decomposition. Bootstrap intervals are descriptive and conditional on each
trained pair, not uncertainty over all possible training runs.

The sequential controls, legacy-versus-corrected comparison, and same-state
checkpoint diagnostics in the report additionally require the original archived
policies and trajectories. Paths retained in the derived JSON identify those
original runs; they are not files supplied by this public checkout. Selected
CSV/JSON summaries and figures are included for inspecting the reported evidence.

## Evidence to preserve from a new run

| Artifact | Purpose |
| --- | --- |
| Training `run_meta.json` | Configuration, seeds, step count, close target, status, policy hashes |
| `training_diagnostics.json` | Rewards, updates, episode end reasons |
| `a/`, `b/`, `checkpoints/` | Initial/intermediate/final policies and normalizer state |
| Evaluation `run_meta.json` | Scenarios, market/action seeds, checkpoint hashes |
| `summary.csv` | Outcomes, participation, fills, risk stops |
| `eval/*/*/mm_*.npz` | Account paths, gross executions, observations, requested actions |
| `eval/*/*/market.json` | Execution counterparties and attempted order submissions |
| `calibration/`, `b_action_pool.npy` | Separate-market source for the state-blind control |
| Source ZIP, manifest, package listing, protocol copy | Source and intended design at launch |

New `runs/` directories are ignored by Git. The corrected confirmation and
calibration trajectories, initial/final policies, source ZIPs, and sanitized
package listings are included under `data/`; earlier raw runs are not all
included. Preserve complete new run directories separately. Checkpoint files save policies, optimizers,
and statistics, not live simulator state for an exact mid-episode resume.

Direct A–B match counts appear in both roles' summary rows; count one role per
joint episode. Submitted-order logs record attempts, not exchange acknowledgments.

The original corrected implementation passed 364 tests and an exact short
legacy replay. All twelve unchanged initial-pair evaluation trajectories matched
between legacy and corrected seed 17. These checks establish specific historical
reproducibility properties; they do not guarantee identical floating-point
training across hardware/dependency versions or establish strategic interaction.
