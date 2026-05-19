# Cell 15 — Hard physics prior delivers the cleanest continuum yet, but the 5× threshold misses SE/DEP

## Empirical result

Cell 15 replaces Cell 14's learnable λ-MLP with a parameter-free
closed form anchored to the HPGe physics rule that real γ-lines
spike to ≥5× the adjacent Compton continuum density:

    R(E_*) = (σ_global / σ_local) · ρ_local(E_*) / ρ_global(E_*)
    λ(E_*) = 1.0 + 9.0 · sigmoid(10 · (R(E_*) − 5))

with σ_local back to 2 keV (HPGe FWHM). Zero trainable parameters
in the gate — the optimizer cannot drift to Cell 14's compressed-λ
degenerate regime.

| metric | Cell 11 | Cell 14 (λ-MLP) | **Cell 15 (hard)** |
| --- | --- | --- | --- |
| MASD (1.7-2.0 MeV) | 0.0038 | 0.0094 | **0.0034** ✓ cleanest |
| MASD (2.2-2.4 MeV) | 0.0041 | 0.0058 | **0.0043** ✓ |
| ACF1 (1.7-2.0) | −0.54 | −0.57 | −0.47 |
| FE Z | +0.04 | +1.02 | **−1.37** ✓ |
| **SE Z** | **−26.7** | **−20.2** | **−22.6** ✗ |
| **DEP Z** | **+18.7** | **+22.8** | **+18.8** ✗ |
| Bi Z | −0.79 | −0.64 | −0.55 ✓ |
| final NBLL | 0.437 | 0.437 | 0.4375 |

**The continuum is the cleanest of any cell so far** — MASD 0.0034
beats even Cell 11's 0.0038. The frequency-domain hard cutoff is
absolute: bins below the 5× threshold *cannot* draw bin-scale
sawtooth no matter what the loss gradient prefers.

FE recovered cleanly (Z = −1.37, p = 0.17). Bi-214 1620 stayed
clean as in all post-Cell-11 cells.

**But SE and DEP relapsed to Cell 11/12 levels.** SE Z = −22.6,
DEP Z = +18.8 — close to the smooth-only catastrophes.

## The mechanism is exactly the pre-training probe

A static λ probe on the actual training pool, evaluated before any
weights are touched, gave:

```
  E_*       R(σ_l=2)     λ        bands open       region
 1460       1.01        1.00      w0, half w1      K-40 (weak peak)
 1592       4.26        1.01      w0, half w1      DEP
 1620       3.92        1.00      w0, half w1      Bi
 1700       0.84        1.00      w0, half w1      ctrl
 1850       0.98        1.00      w0, half w1      ctrl
 2103       4.84        2.55      w0, w1, w2 (94%) SE
 2400       0.88        1.00      w0, half w1      ctrl
 2614      43.32       10.00      ALL 10 bands     FE
 2700       0.003       1.00      w0, half w1      tail
```

The empirical density contrasts on the real ²²⁸Th calibration
spectrum with σ_local = 2 keV are:

* **FE 2614**: R = 43.3 — comfortably above threshold → λ = 10.0
* **SE 2103**: R = 4.84 — *just below* threshold → λ = 2.55
* **DEP 1592**: R = 4.26 — *below* threshold → λ = 1.0 (closed!)
* Bi 1620: R = 3.92 — below threshold, but Bi is naturally broad
  (2 bins) so smooth fit suffices

The hard-gate worked exactly as written. The catastrophe at SE/DEP
isn't a learning failure — it's a *threshold-calibration* failure.
The user-specified rule "γ-lines display ≥5× contrast" is correct
asymptotically but the empirical numbers on this dataset for SE
and DEP fall in the 4.2-4.8 range with σ_local = 2 keV. The
high-sigmoid steepness (s = 10) makes the transition near-binary:
0.16 below threshold is enough to lock λ at 1.0.

## Why the empirical contrast is < 5 at SE/DEP

Two compounding effects:

1. **HPGe physical broadening**. Real lines are not delta functions —
   they're Gaussians with FWHM ~ 2 keV. A "true" 5× contrast would
   require the σ_local kernel to integrate exactly over the line's
   core, which means σ_local must be ≤ line FWHM / 2 ≈ 1 keV. At
   σ_local = 2 keV the kernel mixes a line's core with ~1 keV of
   adjacent continuum, diluting the apparent contrast.

2. **SE/DEP are weaker than FE 2614 in this dataset.** FE is the
   ²²⁸Th source's strongest line. SE (single escape) and DEP
   (double escape) are pair-production secondary peaks with much
   smaller branching ratios — even at full HPGe resolution their
   density contrast is modest compared to FE.

This is the same tension the σ_local = 1 keV → 2 keV decision
(Cell 12 → Cell 14 → Cell 15) was trying to navigate:
* σ_local = 1 keV: peak contrast amplified, but continuum
  contrast is shot-noise-dominated and the kernel is below
  detector physics.
* σ_local = 2 keV: matches FWHM, robust continuum, but peak
  contrast diluted below the 5× threshold for SE/DEP.

## Fork

This cell beautifully isolates the design tradeoff: **the
architecture is correct, but the threshold isn't calibrated to
this dataset's SE/DEP regime.** Three options, all
NBLL-untouched and structurally clean:

1. **Lower the threshold to T = 4.0.** Empirical SE = 4.84 and
   DEP = 4.26 both clear T = 4 comfortably. Continuum baseline
   R ≈ 0.8-1.0 stays well below. K-40 (R = 1.01) and the
   continuum controls stay closed. The 5× rule was a useful
   first-pass physics estimate; the empirical pool says 4× is
   the right cut. The architecture stays parameter-free.

2. **Asymmetric sigmoid centred lower, with two-stage attenuation.**
   Allow partial λ at lower R while keeping the full 5× as the
   "open everything" threshold:
       λ(E_*) = 1.0 + 9.0 · sigmoid(s · (R − 3.5))
   At R = 4.84 (SE) this gives λ ≈ 9.5, at R = 4.26 (DEP) λ ≈ 7.0.
   Both peaks get usable high-frequency access, K-40 (R = 1) still
   gives λ ≈ 1, continuum (R < 1) stays at λ = 1.

3. **Hybrid Cell 14 + Cell 15.** Initialize a learnable λ-head
   to *match* the closed-form expression at training start, then
   let the optimizer fine-tune it — keeping the hard prior as the
   strong initial bias but letting the model adjust the threshold
   for under-represented peaks. Recovers Cell 14's flexibility
   without its compressed-λ degenerate solution.

Cell 16 should test option (1) first — minimal change, cheapest
test of whether the empirical contrast actually separates real
peaks from continuum at T = 4. If 4.0 still misses Bi (R = 3.92)
or admits noise events, escalate to (2) or (3).
