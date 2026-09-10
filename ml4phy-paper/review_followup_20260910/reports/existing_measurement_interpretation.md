# Existing-prediction measurement interpretation

Status: complete from saved artifacts only in 21.5 seconds. No model call, inference, MC sampling, fit, interpolation, or tuning was performed.

The 630-cell inventory is complete: 600 neural cells, ten context-only kernel cells, ten pooled-kernel cells, and ten selected Bernoulli-GP cells. The frozen reference contains 114,400 events in 500 bins. Exactly 442 bins are supported, 58 are excluded, and the excluded bins contain 101 events. All 630 global C1/C2/C3 rows reproduce the prior export; maximum absolute difference is 0 percentage points.

`P_R-Y_R` is the difference between a sum of predicted passing probabilities and the observed passing count in this finite calibration reference. Its efficiency and relative-count versions are descriptive discrepancies, not independently known physical bias or calibrated significance. The event-weighted bin MAE is reported beside signed differences so cancellation cannot masquerade as shape accuracy. Supported-bin and all-nonempty/all-event domains are separate.

Physical names in this export are thallium-208 double escape (DEP), bismuth-212 full energy at 1620.74 keV, thallium-208 single escape (SE), and thallium-208 full energy (FE). Historical machine IDs and exact endpoints remain in the tables. Core event masks reproduce the presentation export's inclusive endpoints; broader masks reproduce the historical regional-error half-open endpoints.

The exploratory `exploratory_continuum_tail_equal_three` score is the equal per-cell mean of Continuum 1, Continuum 2, and sparse-tail Ck. All three components and their bin/event counts remain separately reported. It is not a replacement primary endpoint and the rest of the non-peak spectrum is not called featureless continuum.

Sparse-tail sampling-eligible counts by audited pool are:
- original_n2000: 0 eligible of 2 nominal tail events.
- original_n5000: 0 eligible of 3 nominal tail events.
- original_n10000: 0 eligible of 10 nominal tail events.
- seed20260910_n5000: 0 eligible of 8 nominal tail events.
- seed20260911_n5000: 0 eligible of 4 nominal tail events.
- full_n18866: 0 eligible of 22 nominal tail events.

Zero eligible tail events in a small pool identifies missing sampler support; it does not by itself establish the causal mechanism of a prediction failure. This analysis concerns calibration-mixture selection efficiency only. It does not validate signal efficiency, cross-section bias, a physics-search region, or transfer to another experiment.
