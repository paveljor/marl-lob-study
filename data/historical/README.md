# Historical desktop summaries

These are earlier **synthetic ABIDES experiment summaries**, not real-market
historical data. They precede the corrected concurrent-independent-policy study
and must not be pooled with its results. They are included for research history,
not as additional confirmation of the current findings.

The archive was recovered from a checkout at original revision `5efa14f`.
Individual rows record earlier source revisions. These files are collected
summaries, not primary execution logs; recovery and inventory checks are not
scientific revalidation. The original raw archive remains separate.

| File | Training-run rows | Coverage |
| --- | ---: | --- |
| `desktop/grid.csv` | 600 | Twelve cells, fifty seeds 15–64 |
| `desktop/retrain.csv` | 18 | Six cells, seeds 0, 1, 7 |
| `desktop/latency.csv` | 80 | Four cells, twenty seeds |
| `desktop/rescale.csv` | 100 | Five cells, twenty seeds |
| `desktop/p2.csv` | 160 | Eight cells, twenty seeds |

The historical 780-run grid additionally included 180 runs from another
machine, seeds 0–14. This recovered 600-row file cannot reproduce the pooled
65-seed statistics. A CSV row is a training run, not an independent observation
for every agent, evaluation seed, or time step. Legacy timesteps are also not
universally per-agent counts.

## Known validity limits

- The historical evaluator omitted restoration of background-population
  settings. Policies trained with different value-trader populations were
  evaluated in the default fifty-value-trader market. Without contrary primary
  metadata, these are common-market evaluations of differently trained policies,
  not evaluations across increasingly informed markets.
- Several action-space and reward-normalization corrections superseded earlier
  claims. The old constraint-evasion story did not survive retraining. Legacy
  shared-policy traders were not independent PPO optimizers; rescale and P2
  used one learned trader.
- Historical “adverse selection” includes later inventory exposure. Recorded
  “fills” may be nonzero-net decision intervals rather than individual executions.
  Equity is marked at the midpoint without forced liquidation or explicit fees.
- These CSVs do not contain the observations/actions required to regenerate
  tick histograms, participation diagnostics, or exact action semantics.

Host labels are pseudonymized; numerical values and source revision identifiers
are preserved. `manifest.json` records original and exported hashes. Neither
these summaries nor the current study establish profitable real-market trading,
equilibrium, collusion, or mutual adaptation.

The walkthrough focuses on the corrected study. A future instrument-audit
appendix could investigate these older runs, but should first verify primary
metadata, simulator/source versions, and actual evaluation populations.
