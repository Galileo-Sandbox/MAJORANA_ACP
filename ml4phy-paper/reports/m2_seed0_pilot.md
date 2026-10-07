# Controlled M2 seed-0 pilot result

Status: training and paired development evaluation complete. The checkpoint remains server-only.

## Training result

The 3,000-step M2 attentive-PE10 pilot completed at training seed 0 in 84.94 seconds with a finite final training loss of 0.415522. It used 18,866 training events, 212 retained energy bins, and the frozen classifier lineage. The checkpoint SHA-256 is `2295bdc7e8015ec6387e13f8777c6c293ebe916a7cbb986ef9f21aaaf2d2dba8`.

The committed M2 configuration matches Cell 17 on every non-architecture training field. M2 has 151,682 parameters and Cell 17 has 130,693; exact parameter matching is not claimed.

## Paired development result

Across ten identical context draws with dropout seed 10100, Cell 17 minus M2 had a mean global Brier difference of -0.006134401 and a mean equal-supported-region difference of -0.002821048. Negative favors Cell 17. Both aggregate comparisons favored Cell 17 in 10/10 draws.

The result is not uniform by region: Cell 17 improves the 1700-2000 keV continuum strongly but is worse in the 2200-2400 keV window in 10/10 draws. The full paired table preserves all prespecified supported and diagnostic regions.

## Expansion decision

The pilot passed configuration-parity, runtime, finite-loss, checkpoint-load, and non-degenerate-inference gates. The seed-0 effect is consistent across context draws but does not measure training-seed variability. Authorize only M2 and Cell 17 training seeds 1 and 2 next. Do not launch M0/M1, E3/E4, a hyperparameter sweep, or final-protocol evaluation at this stage.
