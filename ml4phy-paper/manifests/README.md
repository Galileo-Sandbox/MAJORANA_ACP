# Paper manifests

Manifests are the machine-readable provenance layer for the paper. They record
frozen roles, source and output hashes, seeds, model inventories, resource
gates, execution status, warnings, and scientific limitations.

## Reading order

1. `artifact_release_v1.json` defines the public artifact boundary and the
   portable verification anchors.
2. `data_budget_ledger.json` records unique event counts and overlaps.
3. `frozen_protocol_v1.json` and `extension_protocol_v1.json` define the fixed
   evaluation roles and context protocol.
4. `phase2_protocol_v1.json` and `phase3_protocol_v1.json` define training
   subsets and campaign matrices.
5. `phase2_result.json`, `phase3_result.json`,
   `paper_presentation_export.json`, and `paper_coverage_export.json` record
   completed outputs.
6. `resum_flex_compatibility.json` and `server_environment.json` record the
   historical dependency and execution environment.

Manifests are append-only audit records. Do not rewrite an older frozen
protocol to match a later execution decision; record the amendment in a new
manifest or report. Paths to ignored files are logical restoration paths, and
their SHA-256 hashes are the identity checks.
