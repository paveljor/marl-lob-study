# Third-party notices

This project uses the ABIDES JPMorgan Chase fork at upstream revision
`f9cbe51342b7dedd9587e4e069040d68a5c6477f`.

`src/marl_lob/env.py` acknowledges adaptation from ABIDES' Gym environment.
`patches/abides-jpmc-public-local.patch` contains upstream source excerpts and
local modifications. The simulator itself is installed separately, not vendored.

The upstream notices are retained verbatim:

- [J.P. Morgan Chase BSD-3-Clause notice](third_party/abides-jpmc-public/LICENSE)
- [Georgia Tech BSD-3-Clause, attribution, funding, and bundled library notices](third_party/abides-jpmc-public/LICENSE.txt)

These notices cover the applicable ABIDES-derived material and retain their
original terms and disclaimers. They do not select or grant a license for all
project-authored code. Project-authored code has a separate MIT license in
`LICENSE`. Other
installed dependencies retain their respective licenses.
