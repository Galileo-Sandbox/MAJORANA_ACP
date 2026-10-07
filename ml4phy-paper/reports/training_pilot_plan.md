# Controlled M2 training pilot

Status: authorized for one training seed after the frozen development candidate and recovered-baseline gates. This document defines the run before training starts.

## Purpose

Train one M2 attentive CNP with ten Fourier bands and no density-guided components. The run fills the cheapest missing controlled comparison identified in `gap_decision.md`. It is a pilot for configuration parity, runtime, numerical stability, and a non-degenerate learning result; it does not authorize the full M0-M3 ladder or three training seeds.

## Frozen comparison

The pilot configuration is `ml4phy-paper/configs/m2_attentive_pe10_seed0.yaml`. Relative to the recovered Cell 17 configuration, every data, event-sampling, trial-size, context-size, encoder, decoder-hidden-width, dropout, optimizer, batching, threshold-range, and positional-encoding setting is identical.

The intentional architecture changes are:

- keep one 128-dimensional cross-attention head and ten Fourier bands;
- disable decoder coordinate gating;
- disable Gaussian attention bias and all density modulation/SFN paths;
- keep Fourier features in the attention query/key projections;
- remove direct density-contrast injection.

This M2-to-Cell17 comparison therefore tests the complete density-guided package, not the decoder frequency gate alone. The M2 model has 151,682 parameters and recovered Cell 17 has 130,693; exact parameter matching is not claimed.

## Execution gate

1. Commit the configuration and isolated training runner.
2. Run a 10-step smoke job in a unique ignored directory and verify CUDA use, finite loss, checkpoint creation, and provenance capture.
3. If the smoke job passes, run the frozen 3,000-step seed-0 pilot once.
4. Evaluate its checkpoint on development roles only before considering any additional training seed or final-protocol inference.

The runner refuses existing output directories and dirty worktrees, validates the recovered training input and RESUM_FLEX compatibility hashes, streams a server log, and records hashes for the checkpoint, training pool, summary, resolved configuration, and log. Large outputs remain under `ml4phy-paper/runs/training/` on the server.
