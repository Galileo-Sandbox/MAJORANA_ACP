# Configurations

This directory contains user-facing YAML configurations for classifier and
historical efficiency-model workflows.

## Layout

| Path | Purpose |
| --- | --- |
| `small_data_configs/` | Primary small-data classifier configuration family used by the paper lineage. |
| `full_data_configs/` | Full-data classifier configurations. |
| `ultra_small_data_configs/` | Very small development configurations. |
| `smoke_tests/` | Fast configuration and pipeline checks. |
| `cut_acceptance/` | Historical name for conditional efficiency-model configurations and ablations. |

Paper campaigns do not infer settings from directory names. Their executed
copies, hashes, seeds, and subset identities are frozen under
`ml4phy-paper/configs/` and `ml4phy-paper/manifests/`.

Large output locations referenced by these files remain ignored. Before
running a configuration, check its data paths and use the corresponding CLI
help rather than assuming that a historical path exists on another machine.
