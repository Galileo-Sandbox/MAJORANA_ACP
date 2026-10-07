# Final protocol result

Status: controlled M2-versus-Cell17 evaluation complete for three training seeds and ten paired final contexts. This is a prospectively frozen repeated-split evaluation, not an untouched test.

## Primary result

After first averaging paired context draws within each training seed, Cell 17 minus M2 had a global event-level Brier difference of -0.005567787 ± 0.000734862 across the three training-seed differences. The equal-region difference was -0.011736380 ± 0.001768341. Negative favors Cell 17. Both aggregate criteria favored Cell 17 for 3/3 training seeds and 10/10 context draws within every seed.

The development-selected context-only kernel obtained global Brier 0.229468948 ± 0.000342237 and equal-region Brier 0.206233692 ± 0.002884890 across contexts. Kernel bandwidths were frozen from development-only cross-validation and were not retuned with final outcomes.

## Prespecified regional results

| Region | Mean Cell 17 minus M2 | SD across paired training-seed differences | Cell 17-favoring seeds |
|---|---:|---:|---:|
| full | -0.005567787 | 0.000734862 | 3/3 |
| DEP | -0.004790807 | 0.001144672 | 3/3 |
| Bi-214 | -0.012162184 | 0.000949182 | 3/3 |
| continuum_1700_2000 | -0.012106859 | 0.002412834 | 3/3 |
| SE | -0.000557887 | 0.001375485 | 2/3 |
| continuum_2200_2400 | -0.007667343 | 0.001517179 | 3/3 |
| FE | -0.000633561 | 0.000393337 | 3/3 |
| sparse_tail | -0.044236021 | 0.005572356 | 3/3 |
| equal_region_mean | -0.011736380 | 0.001768341 | 3/3 |

Lower is better, so a negative difference favors Cell 17. In development, M2 had been better in `continuum_2200_2400` for every seed and context. The direction reversed under the frozen final protocol: Cell 17 was better for all three training seeds and all ten contexts within each seed. This reversal is reported as an outcome, not used for model or region selection.

## Interpretation boundary

- The 114,400-event target and all ten contexts were frozen before these newly trained checkpoints were evaluated, but the underlying full-test export had been inspected historically. “Untouched test” is therefore not an accurate description.
- Training seeds are the top-level replication unit. Context draws overlap and share the same target; their dispersion is a sensitivity diagnostic, not an independent-sample confidence interval.
- All prespecified regions are retained in the portable tables. The final-set regional pattern must be reported even where it differs from development.
- The 50-pass grid roughness diagnostic is not used for a smoothness claim because the development 0.5-keV check showed material MC-estimator noise. A separate deterministic or higher-precision curve diagnostic is required before making that claim.

## Resource result

One 50-pass Cell 17 final run used about 10,073 MiB peak GPU memory and 10.8 seconds of inference; M2 used about 2,998 MiB and 0.55 seconds. Large event predictions remain on the server. The committed manifest contains hashes but no event identities or server paths beyond project-relative provenance.
