# Public source provenance

This repository releases source, corrected confirmation/calibration raw data,
associated training artifacts, and derived results. The original research Git
history and the complete archive of earlier experiments are retained separately.
The fresh public history does not reconstruct the original development chronology.

## Relationship to the completed study

The source was derived from original baseline
`a1b4cf287256f4fe96c69d8f777c9166fd37ffe3`, followed by local code/documentation
cleanup on 2026-10-14, followed by audited data export and an executable
walkthrough on 2026-10-16. Cleanup is intended to preserve experiment behavior;
it is not new training, an additional replication, or an exact replay of all
historical runs.

The September 26 report describes experiments that ran before this public
cleanup. Historical hashes, seeds, configurations, numerical results, and dates
are retained in the exported evidence. Machine-local paths are converted to
repository-relative references. Original commit identifiers may not resolve in
this new repository. For an exact historical implementation, the original run's
source snapshot and manifest are the relevant records. The corrected training
and confirmation snapshots are included under `data/`; their bytes and original
manifests are preserved, with supplemental full member-hash manifests added.

The exported protocol retains its recorded experimental decisions, with
readability edits and local operational instructions removed. Fresh public Git
history does not independently establish that the protocol was committed before
the experiments. The original study archive is the source of that provenance claim.

## Included and excluded

Included: source, tests, configurations, report, executed walkthrough, derived
tables/figures, and the complete corrected confirmation/calibration data for
training seeds 17, 29, and 43. Training metadata, initial/final checkpoints,
diagnostics, source snapshots, sanitized package listings, and upstream notices
are also included. See `data/README.md` and `data/manifest.json` for the exact
allowlist, checksums, and metadata transformations.

Five older desktop summary CSVs are separately labeled in `data/historical/`.
Recovery does not scientifically revalidate those experiments or supply the
missing 180-run portion of the historical pooled grid.

Not included: original Git history, working notebooks, intermediate checkpoints,
all older pilot/exploratory raw runs, or local environments. Some historical
diagnostics therefore still require archives beyond this release.

Model ZIPs contain serialized Python objects. They are byte-verified artifacts,
not certified safe executable input, and the walkthrough does not load them.

## Attribution and permissions

Original Git authors include Pavel Jordanov (also recorded as Pavel),
Lorenzo Mossini, and Miliks343. Their attribution is preserved in this release.

ABIDES-derived material retains the applicable upstream notices in
`third_party/abides-jpmc-public/`. See `THIRD_PARTY_NOTICES.md`.

Project-authored code is released under MIT; see `LICENSE`. Upstream notices
remain applicable to ABIDES-derived material; they are not replaced by the
project's MIT license.
