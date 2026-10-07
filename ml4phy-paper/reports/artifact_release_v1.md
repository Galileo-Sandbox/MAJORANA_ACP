# Artifact release audit

Audit date: 2026-10-07 (America/Los_Angeles)

Evidence base: `40d3ba4d1a14e6f13e2e226de46b3af800a3bbbe`

## Outcome

The Git artifact now has a paper-specific landing page, a tiered reproduction
guide, citation metadata, a machine-readable release manifest, a portable
integrity verifier, and a safe restoration tool for the pinned historical
RESUM_FLEX source. No training, fitting, inference, or data selection was run
for this release task. Existing scientific results were not changed.

The portable artifact is complete for independent verification of the
committed paper tables, figures, and mechanism-control regional scores. A
fully matched training rerun starts from the public waveform data and remains
conditional on a licensed source for the historical RESUM_FLEX revision. It
does not require publication of historical checkpoints.

## Validation record

| Check | Result |
| --- | ---: |
| Portable release anchors | 12/12 passed |
| Manifest-recorded portable outputs | 67/67 passed |
| Mechanism per-cell bin rows | 75,000/75,000 passed |
| Server-only prediction, curve, checkpoint, role, subset, and classifier-input files | 1,486/1,486 passed |
| Paper and mechanism tests | 32 passed |
| Repository tests | 410 passed, 11 recorded warnings |
| Independent mechanism-summary reconstruction | passed; maximum Ck difference `2.842170943040401e-14` percentage points |
| Historical RESUM_FLEX archive and critical source hashes | passed in check-only mode |
| Ruff checks for new artifact code | passed |

The repository test warnings are two NumPy constant-array runtime warnings,
eight expected sampler scale-imbalance warnings, and one PyTorch same-padding
warning. No test failed.

## Reproduction boundary

| Capability | Fresh Git clone | Additional requirement |
| --- | --- | --- |
| Inspect protocols, code, tables, reports, and figures | Yes | None |
| Verify portable file hashes and experiment inventories | Yes | Python 3.12+ standard library |
| Recompute mechanism regional summaries from committed per-bin predictions | Yes | Locked project environment |
| Regenerate exports from event-level predictions | No | Author-side saved predictions at recorded logical paths |
| Rerun training and stochastic inference | No | Public raw data, deterministic protocol reconstruction, CUDA environment, and pinned RESUM_FLEX source |

The server retains about 4.3 GiB under `ml4phy-paper/runs/`; it is intentionally
not added to Git. The release manifest records exact source hashes, and the
optional server mode of the verifier checks all files selected by the paper
exporters and the mechanism registry.

## Remaining owner actions

1. Publish or identify a licensed source for RESUM_FLEX revision
   `edba6a294581fde6f905b330d477f2c1b42d6adb`.
2. Select a repository license. This audit does not infer or grant one.
3. Update citation metadata if the paper receives a publication DOI.

The paper title, author list, and arXiv identifier are now recorded in
`CITATION.cff`. Checkpoints and event-level predictions are not publication
requirements for this artifact.
