# Two learning traders in one limit order book

An experimental research project on **independent reinforcement-learning traders
sharing a simulated market**, built with ABIDES, PettingZoo, and Stable-Baselines3.
This release contains source, the corrected held-out study's raw data, training
artifacts, and derived results. It does not contain every earlier experimental
run. See [PROVENANCE.md](PROVENANCE.md) for the study baseline and artifact scope.

**This is a synthetic research experiment, not a trading system or investment
advice.** It does not demonstrate profitable trading, exploitation, cooperation,
or mutual adaptation.

## What the completed study found

The September 2026 study trained three pairs of independent PPO policies. Each
trader has its own weights, optimizer, reward statistics, and rollout buffer;
both learn from the same market sessions.

- One trained B policy had lower held-out marked losses when using current
  observations than when its actions were drawn from a state-blind calibration
  pool. The average improvement was **$838.70**, conditional on that trained
  pair. The other two pairs did not establish the same benefit.
- Most of that difference was associated with inventory marking. The control
  removes responses to inventory, time, and the broader market as well as any
  response to the other learner, so it does **not** isolate rival adaptation.
- Sparse quoting did not necessarily mean low risk: some policies accumulated
  strongly one-sided inventories.
- Correcting the finite-session training target overturned an earlier apparent
  interaction. Sampling versus using a policy's deterministic mean also changed
  whether it appeared inactive.

Outcomes are cash plus inventory marked at the midpoint, without fees or forced
terminal liquidation. Three training pairs are three replications, not hundreds
of independent discoveries. See the report for intervals, controls, and limits.

## Start here

| Document | Purpose |
| --- | --- |
| [Executable walkthrough](notebooks/research_walkthrough.ipynb) | Presentation-style explanations, computed plots, and raw-accounting checks |
| [Reading copy (HTML)](docs/research_walkthrough.html) | Download and open for the same story with code hidden; no Jupyter needed |
| [Study data](data/README.md) | Complete corrected confirmation/calibration evidence, model/source provenance, and checksums |
| [Research report](notes/research_report.md) | Findings and mechanisms, with finance concepts explained for programmers |
| [Current state](notes/current_state.md) | Current conclusions; separates them from superseded historical results |
| [Reproduction guide](notes/research_reproduction.md) | Configuration, source provenance, saved evidence, and exact commands |
| [Experiment protocol](notes/concurrent_protocol.md) | Designs recorded before execution |
| [Next research stage](notes/next_research_stage.md) | Proposed opponent-presence test; not executed |

Legacy shared-policy code is retained for reference. Do not pool older results
with the corrected independent-policy study: the older evaluator omitted background-population
settings, and action-space and timing corrections invalidated earlier comparisons.

## Local setup

**Linux with Python 3.11 and CPU PyTorch is the validated setup.** A fresh virtual
environment passed all 369 original tests, Ruff, and `pip check` on 2026-10-14.
The suite also includes source-archive, dataset, and notebook regression coverage.
[Constraints](requirements/linux-py311-cpu.txt) pin the tested dependencies.
These versions are distinct from the historical study environment recorded in
the reproduction guide. Other Python/platform combinations are not verified.

Create and activate a Python 3.11 virtual environment, then run these commands
from this repository's root. They create an ABIDES checkout beside the project;
use a fresh location rather than overwriting an existing dependency checkout.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/jpmorganchase/abides-jpmc-public.git ../abides-jpmc-public
git -C ../abides-jpmc-public checkout f9cbe51342b7dedd9587e4e069040d68a5c6477f
git -C ../abides-jpmc-public apply "$PWD/patches/abides-jpmc-public-local.patch"

python -m pip install -c requirements/linux-py311-cpu.txt torch \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/linux-py311-cpu.txt coloredlogs psutil scipy tqdm
python -m pip install -e ../abides-jpmc-public/abides-core \
  -e ../abides-jpmc-public/abides-markets
python -m pip install -c requirements/linux-py311-cpu.txt -e '.[dev]'
python -m pip check
```

The [ABIDES patch](patches/README.md) is required. It fixes time conversion with
modern Pandas and guards an unavailable execution-agent import. Without the
time fix, plausible-looking simulations can run on the wrong time scale.
ABIDES' old top-level `requirements.txt` includes Gym, Ray, and Pomegranate
versions this project does not use; do not install it for this setup.

`experiments/bootstrap.sh` is an alternative setup tool that creates its own
environment under `.work/`, installs dependencies, and runs tests. Its suite
suggestions describe the **historical experiments**, not the current study.

## Checks

```bash
# Fast checks without market integration tests.
OMP_NUM_THREADS=1 python -m pytest -q -m 'not abides'

# Integration tests require the patched ABIDES installation.
OMP_NUM_THREADS=1 python -m pytest -q -m abides

# Static checks.
python -m ruff check src scripts tests
```

## Train and evaluate

**To inspect the completed study, no training is required.** Open the walkthrough
or recompute the analyses from the committed data using [these commands](data/README.md).
The commands below create new experiments instead.

These commands launch new simulations and write local artifacts. Use a **new
output directory** each time; the scripts refuse to overwrite existing runs.
The saved study used 32,768 joint steps per training pair, with a 256-step rollout.

```bash
export OMP_NUM_THREADS=1

python scripts/train_concurrent.py \
  --config-from experiments/concurrent_base.json \
  --out-dir runs/concurrent/my_pair --seed 17 --steps 32768 --n-steps 256

python scripts/evaluate_interaction.py runs/concurrent/my_pair \
  --out-dir runs/concurrent/my_evaluation

python scripts/analyse_interaction.py runs/concurrent/my_evaluation \
  --out-prefix notes/assets/my_results
```

Evaluation freezes policy weights. It compares sampled and deterministic
policies, initial networks, an inactive opponent, and a state-blind action pool
calibrated on separate markets. The reproduction guide specifies the held-out
market sets and action repeats used in the completed study.

Exchange close is terminal for learning. `--legacy-close-bootstrap` reproduces
the earlier timeout approximation; label such runs separately. The legacy
`scripts/train.py` uses a shared policy and is not the independent concurrent
trainer. Do not use its evaluator for the new fixed-size policies.

## Repository map

```text
src/marl_lob/    Market environment, quote actions, independent/concurrent PPO,
                policy controls, accounting, and trajectory analysis
scripts/        Training, evaluation, analysis, and plotting entry points
tests/          Unit, property, and ABIDES integration tests
experiments/    Current base configuration and historical suite runners
patches/        Required patch against a pinned ABIDES revision
notes/          Report, protocols, research history, and derived evidence
```

The corrected confirmation/calibration trajectories, execution logs, initial/final
models, and historical source snapshots are committed under `data/`. Derived
CSV/JSON tables and figures are under `notes/assets/`. Earlier desktop summaries
are kept separate in `data/historical/` and are not confirmation evidence.

New local runs and environments remain ignored by Git. Earlier pilot/exploratory
raw trajectories and intermediate checkpoints are not all included. Model ZIPs
contain serialized Python objects; the walkthrough never loads them.

## Run the walkthrough

The committed notebook includes reviewed outputs and can be read directly on
GitHub. The HTML reading copy hides long code cells while retaining their plots
and results. To execute all plots and accounting checks locally after setup:

```bash
python -m pip install -c requirements/linux-py311-cpu.txt \
  -c requirements/notebook-py311.txt -e '.[dev,notebook]'
python scripts/verify_study_data.py
python -m jupyter nbconvert --execute --to notebook \
  notebooks/research_walkthrough.ipynb --output-dir outputs \
  --output research_walkthrough --ExecutePreprocessor.timeout=180
```

This reads committed evidence only: no training, model deserialization, simulator
runs, or data downloads. The notebook includes slideshow metadata and can also
be exported to HTML with `jupyter nbconvert --to html`.

## License and contributions

See [CONTRIBUTING.md](CONTRIBUTING.md) for code and research conventions.
Project-authored code uses the [MIT license](LICENSE). ABIDES is developed by
JPMorgan Chase and retains its own BSD terms; this repository includes a local
patch, not a vendored simulator.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the retained upstream notices.

This is experimental research software, not a maintained production trading
service. Changes to market assumptions or learning objectives require new
experiments; they should not silently alter interpretation of the saved results.

Original contributors: Pavel Jordanov, Lorenzo Mossini, and Miliks343. Their
attribution is retained in [PROVENANCE.md](PROVENANCE.md).
