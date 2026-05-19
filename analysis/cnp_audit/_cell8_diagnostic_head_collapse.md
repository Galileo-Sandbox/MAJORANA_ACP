# Cell 8 — Dual-Gated SFN head-collapse diagnostic

**Outcome.** Cell 8 added a temperature-gating head (τ ∈ [1, 10]) on top
of Cell 7's pool-density σ-head (σ ∈ [5, 200] keV) to close the
asymmetric loophole where σ → σ_max collapses the spatial penalty and
leaves the raw `Q·K^T` term free to fit shot noise. **The dual gate
did not close the loophole empirically:**

| paradigm | MASD (1.7-2.0 MeV) | ACF1 (1.7-2.0) | DEP Z |
| --- | --- | --- | --- |
| Cell 6 SFN (physics_anchored, head-wise σ) | **0.080** | −0.43 | +4.55 |
| Cell 7 PD-SFN (flat, head-wise σ)         | 0.292 | −0.46 | +3.26 |
| Cell 8 DG-SFN (flat, head-wise σ + τ)     | **0.296** | **−0.48** | +3.73 |

MASD in the primary control window is statistically identical to
Cell 7; ACF1 marginally worsened. The temperature head registered
correctly in the state-dict, executed in the forward pass — but never
learned to fire.

## Mechanism — head-wise σ/τ probe

We re-loaded the trained Cell 8 checkpoint and computed σ_head(E_*)
and τ_head(E_*) per head at 7 representative energies. The
two MLPs share input `Z_* = [log ρ_local, log ρ_global]` (un-normalised
kernel sums against the fixed pool buffer).

```
   E_*    log_l  log_g    σ_head per head (keV)              τ_head per head
   ────────────────────────────────────────────────────────────────────────
   1592   5.30   7.48    200 200 200 200 200 200 200 │ 5    1.01 ... 1.16 ... 1.00   PEAK
   1620   5.17   7.49    200 200 200 200 200 200 200 │ 5    1.01 ... 1.17 ... 1.00   PEAK
   1850   4.68   7.50    200 200 200 200 200 200 200 │ 5    1.01 ... 1.21 ... 1.00
   2103   5.66   7.71    200 200 200 200 200 200 200 │ 5    1.01 ... 1.13 ... 1.00   PEAK
   2300   5.15   7.79    200 200 200 200 200 200 200 │ 5    1.01 ... 1.16 ... 1.00
   2614   7.26   7.60    200 200 200 200 200 200 200 │ 5    1.00 ... 1.08 ... 1.00   PEAK
   2800  −0.06   6.62    199 199 198 200 199 199 200 │ 6    1.12 ... 2.48 ... 1.00   tail
```

Two pathologies are visible:

### 1. Single-head 1-NN specialisation

Heads 0-6 saturate at **σ_max = 200 keV uniformly across the spectrum**.
Head 7 (the eighth head) collapses to **σ_min = 5 keV — also uniformly,
peaks and continuum alike**. The model carved out a degenerate solution:
seven wide heads doing global averaging, and a single narrow head doing
~5 keV local lookup that is effectively 1-nearest-neighbour. That single
head + the PE10-equipped decoder is enough to fit individual events at
the bin-grid scale, which is exactly what produces the continuum sawtooth.

σ_head therefore **carries no position dependence at all** — neither in
peaks nor in continuum. The pool-density signal is being ignored.

### 2. Temperature gate pinned at τ_min

`τ_head` stayed in the range **1.00 - 1.21** across all 8 heads × 7 probe
energies (the 2800 keV tail outlier is well outside any training-relevant
density regime). The MLP never learned to grow τ in vacuum regions. The
temperature gate is architecturally active but has no effect on the
softmax distribution because the model never moves off τ ≈ τ_min.

## Why training found this configuration

The NBLL loss against binary labels `y_i ∈ {0, 1}` has **no smoothness
prior**. Two configurations achieve the same training loss:

- σ varies with density + τ varies with density → smooth β(E)
- One narrow head + τ ≈ 1 → β(E) that toggles per-event

The second is closer to the random initialisation; gradient descent has
no incentive to walk to the first. The "do not touch the loss" rule
permits this degeneracy by construction — the dual gate created a wider
optimum but training stayed in the same basin.

## Implication for Cell 9

To close the loophole architecturally rather than via a loss
regulariser, **strip the model of head-wise specialisation**. If σ and
τ are *shared across all heads* (one scalar per target query, broadcast
to all H heads in the score calculation), the 1-NN-narrow-head escape
hatch is removed. A single narrow σ would then force every head to
mis-fit continuum noise, producing a large NBLL penalty — driving
training toward the σ-grows-and-τ-grows-jointly-in-vacuum configuration
that Cells 7/8 failed to discover.
