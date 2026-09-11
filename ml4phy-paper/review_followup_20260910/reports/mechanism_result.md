# Mechanism-control result

This post-review analysis uses the historically exposed frozen final reference. It is not an untouched-test experiment. All 150 mechanism cells use the original 5k efficiency-training ordering, 500 conditioning events, the fixed classifier and threshold, and the same reference. Scores were computed per cell before averaging contexts within each seed and then the three seed means.

## Primary endpoints

| Variant | Peaks C2 (%) | Continuum C2 (%) | Peaks weighted MAE (pp) | Continuum weighted MAE (pp) |
| --- | ---: | ---: | ---: | ---: |
| Full density guidance | 45.0 | 88.8 | 5.12 | 3.96 |
| Global decoder gate | 20.8 | 81.7 | 13.24 | 4.97 |
| Global attention | 45.8 | 88.8 | 5.10 | 3.95 |
| Global gate + attention | 22.9 | 81.6 | 13.17 | 4.94 |
| Density-free global | 20.0 | 86.8 | 16.59 | 4.27 |

Reference-band C2 is the fraction of supported 5-keV bins within two fixed Wilson-reference half-widths of the finite empirical fraction. It is not calibrated interval coverage. Peaks are the equal-feature mean of the four frozen narrow cores; Continuum is the equal-window mean of the two frozen continuum regions. No scalar combines these endpoints.

The regional count difference is a sum of predicted passing probabilities minus the observed passing count. It interprets the efficiency discrepancy on a count scale but is neither an independently known physical bias nor a calibrated significance. Event-weighted bin MAE is retained beside signed differences to reveal cancellation.

The paired-contrast table reports every prespecified comparison by initialization seed and context. Contexts overlap and share training/reference data, so the 30 cells per variant are not treated as independent experiments. Learned maps are deterministic checkpoint diagnostics, not MC-noise-corrected predictive smoothness evidence.

The complete-minus-global-gate differences are +24.2 percentage points for Peaks C2 and +7.1 points for Continuum C2. The complete-minus-global-attention differences are -0.8 and -0.0 points. With both modulation rules global, the direct-density-input contrast (`global_both` minus `density_free_global`) is +2.9 points for Peaks and -5.2 points for Continuum. The full-package contrast is +25.0 and +2.0 points. These contrasts must be interpreted jointly with their three seed means in the paired summary; the full-package comparison cannot uniquely attribute its difference to one component.

The decoder-gate contrast is positive for both endpoints in all three seed means (Peaks +12.5, +25.0, +35.0 points; Continuum +4.5, +8.7, +8.0 points). This supports a useful role for the energy-dependent decoder gate in this one-subset slice. The attention contrast is near zero and changes sign across seeds, so a necessary benefit from density-adaptive attention is not supported. The direct-input control is mixed: `global_both` improves mean Peaks C2 by +2.9 points but changes Continuum C2 by -5.2 points relative to `density_free_global`; it does not improve the joint endpoint unconditionally.

Each narrow core contains only two supported 5-keV bins, making Ck highly discrete. The frozen reference has 828 events/638 passes in DEP, 790/264 at 1620 keV, 1,356/295 in SE, and 8,522/1,729 in FE. For the full model, mean predicted-minus-observed passing counts are -14.3, +61.4, -1.5, and -333.2 in those cores. The weighted absolute discrepancies and signed efficiency differences in the tables prevent these count-scale cancellations from being mistaken for shape accuracy.

Sparse-tail behavior remains adverse and variable: full-density C2 is 53.1%, global-gate C2 is 70.3%, and global-attention C2 is 13.3%. No universal-superiority or tail-mechanism claim follows.

All individual core, broad historical peak, continuum, Overall, and sparse-tail results and C1/C2/C3 sensitivity values are preserved in the cell and summary tables. The mechanism slice has one training-subset ordering; it does not establish subset robustness for controls. The fixed classifier used 18,866 events, so this remains efficiency-model data efficiency conditional on pretraining, not end-to-end 5k training.
