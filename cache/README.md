# Tracked cache

This directory contains compact, reviewable inference products used by the
historical visualization notebook. The arrays are tracked so that selected
figures can be inspected without downloading checkpoints or rerunning models.

Each cached array is paired with an audit JSON where available. The files are
historical presentation inputs; the ML4PS paper comparison uses the newer
frozen exports under `ml4phy-paper/tables/` and their provenance manifests.

Do not place raw datasets, checkpoints, or unrestricted event-level exports in
this directory. Rebuild supported cache entries with
`scripts/tools/build_notebook_cache.py` from their recorded local sources.
