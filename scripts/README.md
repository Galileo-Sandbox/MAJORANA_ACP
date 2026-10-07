# General scripts

These scripts support the repository-wide diagnostic and notebook workflows.
Paper-specific frozen campaign runners are documented separately in
`ml4phy-paper/scripts/README.md`.

## Diagnostics

`diagnostics/` runs test-set conditional-model inference, builds regional
audits, compares peak behavior, and evaluates historical cells. These tools may
load checkpoints or launch prediction, so inspect their command-line options
before use.

## Cache tools

`tools/build_notebook_cache.py` converts selected saved inference outputs into
small tracked arrays consumed by `notebooks/data_visualization.ipynb`. A cache
is a presentation convenience, not a replacement for its source provenance.

Generated checkpoints and event-level predictions belong in ignored runtime
directories, not in Git.
