# Paper regional reference-band coverage export

## Status

This saved-artifact-only export completed in 9.7 seconds. It recovered 600 neural cells, 10 context-only kernel cells, 10 selected Matérn-3/2 Bernoulli-GP cells, and 10 pooled-kernel control cells. No training, fitting, prediction, MC sampling, or tuning was run.

Reference-band coverage is the percentage of supported 5-keV bins for which the saved actual-event bin mean lies within `k` times the frozen z=1 Wilson half-width of the empirical target fraction. It is descriptive agreement with a finite empirical reference, not calibrated model-interval coverage, coverage of unknown truth, or the slide's pooled Gaussian transform G.

## Main-paper Table 1 candidate: original 5k and context-only comparators

C2 reference-band coverage (%); whole percentages are display rounding. Continuum is the equal per-cell mean of [1700,2000) and [2200,2400), not an event-pooled score.

| Method | Overall | FE | SE | DEP | 1620 | Continuum |
|---|---:|---:|---:|---:|---:|---:|
| CNP | 80 | 43 | 0 | 50 | 0 | 73 |
| Attentive CNP | 75 | 20 | 0 | 50 | 0 | 66 |
| Attentive CNP + PE | 36 | 17 | 72 | 17 | 60 | 37 |
| Density-guided CNP | 80 | 13 | 17 | 100 | 50 | 89 |
| Kernel | 57 | 10 | 5 | 35 | 0 | 63 |
| Bernoulli GP | 55 | 0 | 5 | 35 | 5 | 61 |

Sensitivity (C1/C2/C3, %):

- CNP: Overall 47.8/80.1/92.5; FE 21.7/43.3/60.0; SE 0.0/0.0/0.0; DEP 21.7/50.0/50.0; 1620 0.0/0.0/0.0; Continuum 39.8/73.4/91.7.
- Attentive CNP: Overall 43.8/74.8/89.8; FE 15.0/20.0/31.7; SE 0.0/0.0/0.0; DEP 18.3/50.0/50.0; 1620 0.0/0.0/0.0; Continuum 34.8/66.4/86.2.
- Attentive CNP + PE: Overall 18.4/36.2/52.8; FE 6.7/16.7/31.7; SE 33.3/71.7/80.0; DEP 15.0/16.7/35.0; 1620 36.7/60.0/78.3; Continuum 17.0/36.7/56.5.
- Density-guided CNP: Overall 47.6/79.5/93.4; FE 11.7/13.3/23.3; SE 8.3/16.7/76.7; DEP 40.0/100.0/100.0; 1620 21.7/50.0/50.0; Continuum 50.9/88.8/99.4.
- Kernel: Overall 30.9/56.8/76.3; FE 10.0/10.0/30.0; SE 0.0/5.0/5.0; DEP 5.0/35.0/50.0; 1620 0.0/0.0/0.0; Continuum 35.2/63.2/82.5.
- Bernoulli GP: Overall 29.3/55.5/74.3; FE 0.0/0.0/0.0; SE 0.0/5.0/10.0; DEP 15.0/35.0/50.0; 1620 0.0/5.0/15.0; Continuum 33.9/61.2/80.7.

The pooled-data kernel is excluded from this matched headline because it uses the full 18,866-event efficiency-training pool plus context; it remains in the CSV as a stronger-data control.

## Main-paper Table 2 candidate: original training ordering

C2 reference-band coverage (%). Peaks is the equal per-cell mean of FE, SE, DEP, and the 1620-keV feature; Continuum is the equal per-cell mean of the two fixed windows.

| Method | 2k Overall | 2k Peaks | 2k Continuum | 5k Overall | 5k Peaks | 5k Continuum | 10k Overall | 10k Peaks | 10k Continuum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNP | 77 | 27 | 82 | 80 | 23 | 73 | 82 | 22 | 77 |
| Attentive CNP | 75 | 12 | 78 | 75 | 18 | 66 | 82 | 26 | 74 |
| Attentive CNP + PE | 28 | 22 | 26 | 36 | 41 | 37 | 47 | 51 | 49 |
| Density-guided CNP | 47 | 13 | 42 | 80 | 45 | 89 | 81 | 42 | 88 |

## Budget evidence and fixed comparison

For Density-guided CNP, original-ordering C2 coverage (Overall / equal-feature Peaks / equal-window Continuum) is 2k: 46.6/13.3/41.6%; 5k: 79.5/45.0/88.8%; 10k: 80.8/42.1/87.6%.
The 10k result is nonmonotonic relative to 5k for: peak_equal_feature, continuum_equal_window.
The 2k row quantifies the low-budget failure directly; it must remain visible rather than being omitted from the learning curve.

C2 change from original 5k to original 10k (percentage points):

- CNP: Overall +2.1; Peaks -1.7; Continuum +3.6.
- Attentive CNP: Overall +7.7; Peaks +8.3; Continuum +7.9.
- Attentive CNP + PE: Overall +10.5; Peaks +9.6; Continuum +12.5.
- Density-guided CNP: Overall +1.3; Peaks -2.9; Continuum -1.2.

Exact exceptions to the statement that 5k ours matches or improves both peak and continuum coverage relative to every original-10k neural comparator:

- original_n5000 vs Attentive CNP + PE at k=1: Peaks -5.83 pp, Continuum +25.74 pp.
- original_n5000 vs Attentive CNP + PE at k=2: Peaks -5.83 pp, Continuum +39.57 pp.
- original_n5000 vs Attentive CNP + PE at k=3: Peaks -8.33 pp, Continuum +29.71 pp.
- original_n5000 vs Density-guided CNP at k=3: Peaks -7.92 pp, Continuum +0.43 pp.
- seed20260910_n5000 vs Attentive CNP + PE at k=2: Peaks -17.92 pp, Continuum +31.17 pp.
- seed20260910_n5000 vs Attentive CNP + PE at k=3: Peaks -26.25 pp, Continuum +26.62 pp.
- seed20260910_n5000 vs Density-guided CNP at k=1: Peaks +9.58 pp, Continuum -3.74 pp.
- seed20260910_n5000 vs Density-guided CNP at k=2: Peaks -9.17 pp, Continuum -7.18 pp.
- seed20260910_n5000 vs Density-guided CNP at k=3: Peaks -25.83 pp, Continuum -2.65 pp.
- seed20260911_n5000 vs Attentive CNP + PE at k=2: Peaks -0.83 pp, Continuum +35.58 pp.
- seed20260911_n5000 vs Attentive CNP + PE at k=3: Peaks -2.92 pp, Continuum +28.86 pp.
- seed20260911_n5000 vs Density-guided CNP at k=2: Peaks +7.92 pp, Continuum -2.76 pp.
- seed20260911_n5000 vs Density-guided CNP at k=3: Peaks -2.50 pp, Continuum -0.42 pp.

## Region construction and checks

Bins are selected by center. The narrow memberships are FE_2614_pm5: [2612.5, 2617.5]; SE_2103_pm5: [2102.5, 2107.5]; DEP_1592_pm5: [1587.5, 1592.5]; feature_1620_pm5: [1617.5, 1622.5]. These two-bin feature scores are discrete (0%, 50%, or 100% per cell at each k). They are not identical to the previous inclusive +/-5-keV pooled-event G regions because boundary event membership differs.

Overall contains 442 supported and 58 excluded bins; excluded bins contain 101 target events. All methods use the identical support and frozen 114,400 targets.

Aggregation first averages the ten contexts within each initialization seed, then equally averages the three initialization seeds. The three 5k training-subset summaries are kept separate; their combined row equally averages subset means and reports subset range and descriptive SD. Overlapping contexts are not treated as independent datasets.

All 630 source hashes were checked. Direct event-level bin means were reconciled with saved neural/GP `curve.npz` means to a maximum absolute difference of 0. The consolidated kernel archive has no second stored bin-mean route; its predictions and frozen identities were verified directly.

Exact context membership hashes agree across all neural and GP cells for seeds 100--109. The consolidated kernel archive records context seeds but not context-row identities, so its context association remains inherited from the frozen kernel protocol rather than independently recoverable from that output file.

The newly computed global C1/C2/C3 values reproduce all 630 available rows in `paper_reference_pull_cells.csv` within 1.11e-16 in fraction units.

## Interpretation limits

Global coverage can reward continuum agreement while hiding missed narrow features, so the global, equal-feature, and equal-window columns must be read together. Higher reference-band coverage can also reflect broad finite-reference bands; it is not a calibrated 68/95/99.7% statement.

The existing 5k result supports an efficiency-model training-budget comparison conditional on a fixed classifier trained with 18,866 events and 500 conditioning events. It does not establish end-to-end training on 5,000 events. There is no 20k efficiency-training run; the archived full pool is exactly 18,866 events.

Sparse-tail limitations from the previous export remain: at original 5k, Density-guided CNP sparse-tail MAE was 19.39 percentage points versus 2.99 for the stronger-data pooled-kernel control. This export does not redefine the tail or claim calibrated uncertainty.

No missing cells or scientific failures were replaced by new computation. The nonlinear per-cell fractions here cannot be reconstructed from the compact mean curves alone; they were calculated from the saved per-cell event predictions.

## Portable files

- `paper_coverage_cells.csv`: all per-cell individual and composite regional scores.
- `paper_coverage_summary.csv`: hierarchical setting summaries and all-three-5k subset dispersion.
- `paper_coverage_region_bins.csv`: exact bin membership, target counts, fractions, and support.
- `paper_coverage_budget_comparison.csv`: all thresholds for the fixed neural budget comparison.
- `paper_coverage_cross_budget_pairs.csv`: descriptive paired 5k-ours versus original-10k comparisons, including ties.
- `paper_coverage_export.json`: definitions, source/output hashes, checks, and provenance.
