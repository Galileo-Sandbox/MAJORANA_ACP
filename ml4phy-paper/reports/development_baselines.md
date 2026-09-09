# Development baseline comparison

Status: recovered-model and inference-only baseline evaluation complete on the frozen development protocol. No new model training contributed to this comparison.

## Result

Cell 17 remains the frozen candidate. Its mean global Brier score is 0.225620633, compared with 0.230185155 for the context-only kernel and 0.229235712 for the best recovered historical baseline, Base 3. The corresponding equal-supported-region means are 0.234185498, 0.233245855, and 0.232818034.

Each value is the mean over ten paired context draws with a fixed target. Neural models use dropout seed 10100 and 50 MC passes. The kernel bandwidth is selected independently for each draw by 5-fold context-only cross-validation over 2, 5, 10, 20, 50, 100 keV; target outcomes are never used for bandwidth selection.

## Interpretation boundary

- True CNP and Base 1/2/3 are recovered historical models with different samplers, context ranges, attention settings, positional encodings, or other training choices. Their comparison with Cell 17 is informative but confounded and is not a controlled architecture ablation.
- All neural models have only training seed 0. The ten overlapping context draws share one 2,074-event development target and do not establish training-seed uncertainty.
- DEP and the sparse tail remain unsupported in the development target. The equal-region summary excludes unsupported regions by the frozen rule.
- The context-only kernel has a matched 2,000-event inference budget. Its bandwidth tuning cost and effective support are recorded in the tables and metrics JSON.

## Training gate

The recovered controls and the eligible kernel do not close the controlled M2-versus-density-guided comparison. Together with the candidate decision, this satisfies the gap-decision condition for one M2 training pilot at seed 0. A full ladder or three-seed campaign remains unauthorized until the pilot is checked for configuration parity, runtime, and a non-degenerate learning result.
