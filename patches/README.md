# Local ABIDES patches

`abides-jpmc-public-local.patch` holds the two local modifications this project
requires against the ABIDES jpmc fork. They are **not upstream**. Without them
results are silently wrong rather than obviously broken.

Generated against upstream commit `f9cbe51342b7dedd9587e4e069040d68a5c6477f`
(`main`).

## Apply

```bash
git clone https://github.com/jpmorganchase/abides-jpmc-public.git
cd abides-jpmc-public
git checkout f9cbe51342b7dedd9587e4e069040d68a5c6477f
git apply /path/to/patches/abides-jpmc-public-local.patch
pip install -e abides-core -e abides-markets
```

## What is in it

1. **`abides-core/abides_core/utils.py` — `str_to_ns` was 1000x wrong.**
   On pandas >= 2.x it returned microseconds, not nanoseconds, via
   `.to_timedelta64().astype(int)`. Fixed to `pd.to_timedelta(string).value`.
   Consequence of the bug: what looked like a 1-hour simulation was actually
   simulating **3.6 seconds** of market time. Runs using the unpatched conversion
   under affected Pandas versions are suspect, regardless of their calendar date.

2. **`abides-markets/abides_markets/configs/rmsc03.py`** — `POVExecutionAgent`
   guarded as a soft `None` import; the symbol does not exist in this fork, so
   the module fails to import without the guard.

The top-level README explains setup; `notes/research_reproduction.md` records
the study's source-snapshot and dependency provenance.
