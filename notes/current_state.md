# Current study and public-source status

The bounded research study completed on **2026-09-26**. Start with the
[research report](research_report.md). This source-and-derived-results release
does not add a new research experiment.

- Three independently trained concurrent PPO pairs and their exploratory and
  held-out evaluations are complete.
- One B policy had a held-out benefit from state conditioning relative to its
  state-blind replacement. This was lower marked loss, not profitability.
- The comparison does not isolate a response to the other learner, and the
  other two pairs did not establish the same benefit.
- The proposed next test checks whether the benefit persists without A's orders.
  It has not been run. See [the recommendation](next_research_stage.md).

Code cleanup and public-document curation took place after the study. No new
research training or evaluation batches were launched during public preparation.
See [source provenance](../PROVENANCE.md).

The corrected confirmation/calibration raw data and associated training artifacts
are included under `data/`, with an executed walkthrough under `notebooks/`.
Earlier desktop summaries are separately labeled historical, not pooled into
confirmation. The corrected report supersedes earlier experimental results
invalidated by implementation corrections.
