# Contributing

This is a research codebase, not a production trading application. Start with
the README and `notes/current_state.md` before changing an experiment.

## Development

Use the patched ABIDES revision and environment described in the README.
Run `python -m ruff check src scripts tests` and
`OMP_NUM_THREADS=1 python -m pytest -q -m 'not abides'` for ordinary changes.
Run the ABIDES integration tests when changing environment behavior, exchange
timing, actions, accounting, or the concurrent collector.

The GitHub Actions workflow runs these checks with Python 3.11 and the pinned,
patched simulator. It does not launch research training or evaluation batches.

Keep changes small. Add a regression test for a bug fix. Separate mechanical
cleanup from changes to learning targets or market behavior. Do not replace
historical code merely because a newer experiment no longer uses it.

## Research conventions

- Record a comparison's protocol before running it. Label exploratory and
  post-hoc analyses explicitly.
- Preserve configurations, seeds, source snapshots, checkpoints, and failures.
  Never overwrite an existing run directory or edit its archived source.
- Distinguish independent training replications from evaluation markets,
  action repeats, and individual time steps.
- Check gross executions against cash and inventory before interpreting P&L.
  Mark-to-mid outcomes are not realized returns after liquidation and fees.
- When a correction changes semantics, label the old runs as historical rather
  than silently pooling or replacing them.
- Keep measured behavior separate from causal interpretation. Lower losses do
  not by themselves demonstrate opponent adaptation or profitability.

## Sharing changes

Do not commit credentials, local environments, raw run directories, or notebook
outputs containing private machine details. Review both staged changes and
history before public release: `.gitignore` does not remove committed data.
See `PROVENANCE.md` for the scope of the included research evidence.
