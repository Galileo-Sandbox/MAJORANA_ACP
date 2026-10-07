# Controlled three-training-seed development result

Status: the prespecified M2-versus-Cell17 development comparison is complete for training seeds 0, 1, and 2. Final-protocol results have not been inspected for these newly trained models.

## Primary result

Training seeds are the replication unit. After averaging the ten paired context draws within each seed, Cell 17 minus M2 had a global Brier difference of -0.005364474 ± 0.000667050 across the three training-seed differences. The equal-supported-region difference was -0.002753224 ± 0.000884383. Negative favors Cell 17; both criteria favored Cell 17 for 3/3 training seeds and 10/10 context draws within every seed.

The context draws overlap and share one 2,074-event development target. Their within-seed standard deviation is a context-sensitivity diagnostic, not an independent-sample confidence interval. With only three training seeds, the across-seed standard deviations are descriptive.

## Regional boundary

The benefit is not uniform. Cell 17 improves the 1700-2000 keV continuum strongly for every training seed, while M2 improves the 2200-2400 keV window for every training seed and context draw. Unsupported DEP and sparse-tail evidence remains inconclusive. Any paper claim must state this regional trade-off and cannot say that density guidance is universally better.

## Resource result

M2 training took 84.86-87.93 seconds for the three new paper runs; Cell 17 seeds 1 and 2 took 125.55-126.10 seconds. The recovered Cell 17 seed-0 runtime is unavailable. Checkpoints and event-level predictions remain server-only; portable tables contain their hashes.

## Next gate

Freeze Cell 17 as the three-seed development candidate. Run one final-protocol timing/memory pilot before deciding how many paired final-context evaluations are computationally justified. Do not add architectures, hyperparameters, or training seeds.
