# Paper configurations

This directory contains compact, tracked configurations resolved for the
ML4PS paper campaigns. They are inputs to paper runners and records of the
settings actually used; they do not contain checkpoints or data rows.

## Configuration groups

- `m0_cnp_seed0.yaml`, `m1_attentive_cnp_seed0.yaml`, and
  `m2_attentive_pe10_seed0.yaml` define the matched neural baselines.
- `cell17_controlled_seed0.yaml` defines the density-guided architecture from
  which seed-specific paper runs were constructed.
- `extension_models_v1.json`, `recovered_models_v1.json`, and
  `trained_models_v1.json` record model inventories across recovery and
  extension stages.
- `phase2_*` files freeze the 2k and original 5k training-budget campaign.
- `phase3_*` files freeze the 10k prefix and the two additional 5k orderings.
- `gp_protocol_v1.json` freezes dense Bernoulli-GP fitting and selection.
- `mc_smoothness_protocol_v1.json` freezes the bounded MC-noise diagnostic.

Authoritative execution status belongs in `../manifests/`, not in configuration
filenames. Historical configurations are preserved even when a later report
records a protocol amendment.
