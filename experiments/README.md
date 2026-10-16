# Legacy shared-policy experiment suites

These runners use the older shared-policy workflow, **not** the corrected
independent concurrent-PPO study. For current training and evaluation, start
with the [main README](../README.md) and [reproduction guide](../notes/research_reproduction.md).
Do not use these suites to reproduce the September 26 report.

The optional bootstrap places the environment and simulator under `.work/` and
records outputs under `results/`. It installs no system-wide packages, but pip
may use shared caches. Use a separate checkout for disposable experiments;
`teardown.sh` removes local runs and environments, not shared caches.

```bash
# From a checkout on Linux, with Python 3.11 available:
experiments/bootstrap.sh              # venv + patched ABIDES + install + verify
experiments/run.sh smoke              # ~2 min: proves the pipeline works here
experiments/run.sh cheap -j 8         # the four cheap experiments
python experiments/collect.py cheap   # -> results/cheap.csv
experiments/teardown.sh               # removes .work/, runs/, jobs/; keeps results/
```

## What bootstrap does

1. Checks the tested Python 3.11 version and creates `.work/venv`.
2. Clones ABIDES at pinned commit `f9cbe51` into `.work/`.
3. **Applies `patches/abides-jpmc-public-local.patch` and refuses to continue if
   it isn't applied.** Without it `str_to_ns` is 1000× wrong and a "1 hour"
   simulation actually simulates 3.6 seconds — results would be silently wrong
   rather than obviously broken.
4. Installs `abides-core` + `abides-markets` editable. They declare no
   `install_requires`, so nothing is pulled from ABIDES' top-level
   `requirements.txt` (which pins `gym`/`ray`/`pomegranate` — not needed here
   and no longer building cleanly).
5. Installs CPU PyTorch and the simulator's dependencies using the public setup
   constraints. These are a separately validated environment, not historical
   training-version pins.
6. Installs this project with dev extras.
7. Writes provenance to `results/<host>-<stamp>/`: `pip-freeze.txt`, `machine.txt`.
8. Verifies by running both the fast suite and the ABIDES integration tests.

## Suites

`experiments/matrix.py` is the single source of truth and generates one
executable script per job into `experiments/jobs/<suite>/`, so any job can be
re-run by hand and each gets its own log.

| suite | jobs | what it answers |
|---|---|---|
| `smoke` | 1 (2k steps) | does the pipeline work on this machine |
| `cheap` | 12 | `--norm-obs` (E1), both escape hatches closed, default inventory penalty — 3 seeds each |
| `grid` | 36 | the spine: `num_value_agents ∈ {10,50,150,400}` × `n_agents ∈ {1,2,4}` × 3 seeds |

Predictions for `grid` are written down in `matrix.py`'s docstring **before** it
runs. That is the difference between an experiment and a sweep.

## Performance notes

- **A GPU does not help.** The policy is a 64×64 MLP on a 44-dim observation;
  the bottleneck is the single-threaded pure-Python ABIDES kernel. `train.py`
  defaults to `--device cpu` on purpose.
- **The win is running many sims at once**, not making one faster. `run.sh`
  defaults to `ncpu − 2` capped at 8.
- **It's RAM-bound too.** Each sim spawns 2000+ background agents. On a machine
  with little memory use `-j 1`; with 32GB+ you can push `-j 8`.
- Reference: on an Intel i5-2415M (2 cores, 2011) a 50k-step run takes ~11 min
   and only `-j 1` is realistic. These are historical performance notes, not
   validation of the public setup on other platforms.
  `run_meta.json` records `train_seconds`, host, and machine for every run, so
  results from different machines stay comparable.

## Results survive teardown

`results/` is never deleted by `teardown.sh`. To take it with you:

```bash
tar czf ~/marl-lob-results.tgz -C "$(pwd)" results
```

Each run also writes `run_meta.json` next to its checkpoint — full args, git
SHA, host, platform, python version, wallclock. Without that, numbers collected
on a borrowed machine can't be reproduced or compared later.
