# Pre-training mechanism-control audit

Status: passed. No scientific training or evaluation was run during this audit.

- Base commit: `ceff71ea623cac12e3ba025b633b3782d380f88a`.
- Full-mode deterministic and fixed-RNG stochastic maximum differences: exactly 0.
- Original 5k subset: 5,000 nominal, 4,984 sampling-eligible, 5,000 density-buffer events; identity and file hashes passed.
- Existing inventory: 600 neural plus 30 classical/control cells; frozen target support remains 442/500 bins with 101 events in 58 excluded bins.
- Control checkpoint round trips, finite intended gradients, global parameterization tests, shared initialization, and all three 3,000-step task-schedule hashes passed.
- The frozen target and context reservoirs are identity-disjoint from the training subset according to the Phase 2/extension manifests.

The implementation retains all original model tensors. Global-gate modes
repurpose `kappa_raw` as `phi`; global-attention modes retain both original
networks but their two first-layer input-weight columns are structurally inactive
under `Z0=(0,0)`. The density-free mode additionally makes the decoder's direct
R-input weight column inactive. Exact total/effectively participating parameter
counts are in `configs/protocol_v1.json`.

The full-mode parity gate is satisfied, so the authorized single-job resource
pilot may proceed only after this protocol and implementation are committed.
