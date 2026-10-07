# Controlled training-seed expansion

Status: authorized after the M2 seed-0 pilot; seeds 1 and 2 have not yet run when this decision is committed.

The seed-0 M2 pilot completed in 84.94 seconds with finite loss and valid outputs. On ten fixed-dropout paired development contexts, Cell 17 improved both frozen aggregate Brier criteria in 10/10 draws relative to M2. The global mean difference was -0.006134 and the equal-supported-region difference was -0.002821, where negative favors Cell 17. The 2200-2400 keV region favored M2 in 10/10 draws, so the effect is not uniform.

The next authorized jobs are exactly:

- M2 attentive PE10 at training seeds 1 and 2;
- Cell 17 at training seeds 1 and 2.

Together with the existing seed-0 checkpoints, these jobs provide three training seeds per core model. All four new jobs retain 3,000 steps and the frozen data, sampler, network-capacity, dropout, and optimizer settings. They run sequentially on the same GPU. No M0/M1, E3/E4, hyperparameter sweep, or final-protocol inference is authorized by this decision.
