"""Build the inclusive (``target_class="all"``) audit table across paradigms.

Companion to ``build_peak_comparison.py``: focuses on the inclusive
cell (the spectrum-wide pass-rate the experiment ultimately reports),
which the user has flagged as the highest-priority view for the
hybrid-scale sweep.

For each (paradigm, bin10/inclusive) cell, loads ``test_set_audit.json``
and emits:

* Coverage table — combined-σ + CNP-only at 1σ / 2σ / 3σ. Target =
  Gaussian (0.683 / 0.954 / 0.997).
* Per-peak χ² / Z / p table — same four γ-peak markers as the signal
  report, in a single side-by-side grid across all 5 paradigms.
* Global sanity check — Pearson r + mean offset + combined-σ coverage
  as a single-row summary per paradigm.

Cells whose JSON is missing render as ``—``; the script is useful
mid-training as inclusive cells come online.
"""

from __future__ import annotations

import json
from pathlib import Path

# (nickname, audit-tree paradigm path under analysis/cnp_audit/<model>/)
PARADIGMS = [
    ("true_cnp", "true_cnp"),
    ("w10_fixed48", "hybrid_scale/mixed_density_f0_70_w10"),
    ("w10_varN_large", "hybrid_scale/mixed_density_f0_70_w10_varN32-1024"),
    ("physics_w10_varN", "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024"),
    ("hyper_zoom_w5", "hybrid_scale/mixed_density_f0_85_w5_varN32-1024"),
    ("w10_varN_pe10", "hybrid_scale/mixed_density_f0_70_w10_varN32-1024_pe10"),
    (
        "w10_varN_pe10_attn",
        "hybrid_scale/mixed_density_f0_70_w10_varN32-1024_pe10_attn8x64",
    ),
    (
        "physics_pe10_attn_gated",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gated",
    ),
    (
        "physics_pe9_attn_gated",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe9_attn8x64_gated",
    ),
    (
        "physics_pure_attn4x64",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_attn4x64",
    ),
    (
        "physics_pe10_attn_gab",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gab",
    ),
    (
        "physics_pe10_attn_gab_debinned",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gab_debinned",
    ),
    (
        "physics_pe10_attn_gab_dense",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gab_dense",
    ),
    (
        "physics_peOff_attn_gab_dense",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_attn8x64_gab_dense",
    ),
    (
        "physics_pe10_attn_gab_bpbn",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gab_bpbn",
    ),
    (
        "physics_pe10_attn_gab_sfn",
        "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024_pe10_attn8x64_gab_sfn",
    ),
    (
        "flat_pe10_attn_gab_pdsfn",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn8x64_gab_pdsfn",
    ),
    (
        "flat_pe10_attn_gab_dgsfn",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn8x64_gab_dgsfn",
    ),
    (
        "flat_pe10_attn_gab_dgsfn_tied",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn8x64_gab_dgsfn_tied",
    ),
    (
        "flat_pe10_attn1_pedetach_dgsfn_tied",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gab_dgsfn_tied_pedetach",
    ),
    (
        "flat_pe10_attn1_gated_pedetach_dgsfn_tied",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_pedetach",
    ),
    (
        "flat_attn1_fullgated_dgsfn_tied_sl2",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_sl2_pedetach",
    ),
    (
        "flat_attn1_fullgated_dgsfn_tied_pegate_sl2",
        "hybrid_scale/flat_stratified_varN640-1024_pe10_attn1x128_gated_gab_dgsfn_tied_pegate_sl2_pedetach",
    ),
]

PEAK_ORDER = [
    ("Tl-208 FE", "FE 2614"),
    ("Tl-208 SE", "SE 2103"),
    ("Tl-208 DEP", "DEP 1592"),
    ("Bi-214 1620", "Bi 1620"),
]

# Pristine Compton-continuum control regions for the sawtooth-diagnostic
# suite. Both windows are deliberately clear of any γ-peak so we measure
# the model's behaviour on flat physics where the truth is "smooth slope"
# — any roughness here is the model's fault.
SAWTOOTH_REGIONS = [
    ("1.7–2.0 MeV", "region_1700_2000"),
    ("2.2–2.4 MeV", "region_2200_2400"),
]

AUDIT_ROOT = Path("analysis/cnp_audit")
MODEL = "simple_cnn_small"
BIN_DIR = "bin10"
CLS = "inclusive"
OUT_PATH = AUDIT_ROOT / "_audit_inclusive.md"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _fmt_chi2(x: float | None) -> str:
    return "—" if x is None else f"{x:6.2f}"


def _fmt_z(z: float | None) -> str:
    return "—" if z is None else f"{z:+6.2f}"


def _fmt_cov(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def _load(paradigm_rel: str) -> dict | None:
    p = AUDIT_ROOT / MODEL / paradigm_rel / BIN_DIR / CLS / "test_set_audit.json"
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)


def _peak_lookup(audit: dict | None, peak_name: str) -> dict | None:
    if audit is None:
        return None
    for pm in audit.get("peak_metrics", []):
        if pm["peak_name"] == peak_name:
            return pm
    return None


def _coverage_verdict(cov_1sigma: float | None) -> str:
    if cov_1sigma is None:
        return "—"
    # Wilson-band fairness: 0.683 ± 0.05 is "calibrated".
    if 0.633 <= cov_1sigma <= 0.733:
        return "✓ calibrated"
    if cov_1sigma > 0.733:
        return "↑ over-covered (σ too wide)"
    return "↓ overconfident (σ too tight)"


def _peak_verdict(p: float | None, chi2: float | None) -> str:
    """Single-cell verdict from p-value + reduced χ² (held-out side).

    Tiers:
        p > 0.5  & χ² < 2  → ✓ clean (mean within 1σ, χ² well-calibrated)
        p > 0.05 & χ² < 2  → ~ marginal (not statistically significant, but
                              not as tight as "clean")
        p > 0.5            → △ mean OK, χ² high (oscillation)
        otherwise          → ✗ local miss (p ≤ 0.05 → reject indistinguishability)
    """
    if p is None or chi2 is None:
        return "—"
    if p > 0.5 and chi2 < 2:
        return "✓ clean"
    if p > 0.05 and chi2 < 2:
        return "~ marginal"
    if p > 0.5:
        return "△ mean OK, χ² high"
    return "✗ local miss"


def main() -> None:
    audits = {nick: _load(rel) for nick, rel in PARADIGMS}
    have = [nick for nick, a in audits.items() if a is not None]
    missing = [nick for nick, a in audits.items() if a is None]

    lines: list[str] = []
    lines.append("# Inclusive cell audit  ·  bin10/inclusive across paradigms")
    lines.append("")
    lines.append(
        f"Source: `{AUDIT_ROOT}/{MODEL}/<paradigm>/{BIN_DIR}/{CLS}/"
        "test_set_audit.json` (regenerated via "
        "`python -m scripts.diagnostics.cnp_test_inference <cfg>` or the "
        'full sweep). The inclusive cell uses ``target_class="all"`` — '
        "every test event contributes to D_T, no label filter — so this is "
        "the spectrum-wide pass-rate view the experiment ultimately reports."
    )
    lines.append("")
    if missing:
        lines.append("**Missing cells (rendered as `—`):** " + ", ".join(missing))
        lines.append("")
    lines.append(f"Loaded cells: {', '.join(have) if have else '(none yet)'}")
    lines.append("")

    # ──────────────────────────────────────────────────────────────
    # Coverage table — combined-σ at 1/2/3σ + CNP-only at 1/2/3σ.
    # ──────────────────────────────────────────────────────────────
    lines.append("## Coverage  (target: Gaussian 0.683 / 0.954 / 0.997)")
    lines.append("")
    lines.append(
        "**combined-σ** uses √(σ_CNP² + σ_emp²) (model uncertainty + Wilson "
        "binomial scatter); **cnp-only** uses just σ_CNP and reveals whether "
        "the model is honest about its own epistemic uncertainty. The 1σ "
        "value is the key calibration knob — much above 0.683 = bands too "
        "wide; much below = overconfident."
    )
    lines.append("")
    lines.append(
        "| paradigm | combined 1σ | combined 2σ | combined 3σ | "
        "cnp-only 1σ | cnp-only 2σ | cnp-only 3σ | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for nick, _ in PARADIGMS:
        a = audits[nick]
        if a is None:
            lines.append(f"| {nick} | — | — | — | — | — | — | — |")
            continue
        cc = a.get("coverage_combined", {})
        co = a.get("coverage_cnp", {})
        verdict = _coverage_verdict(cc.get("1sigma"))
        lines.append(
            f"| {nick} | "
            f"{_fmt_cov(cc.get('1sigma'))} | "
            f"{_fmt_cov(cc.get('2sigma'))} | "
            f"{_fmt_cov(cc.get('3sigma'))} | "
            f"{_fmt_cov(co.get('1sigma'))} | "
            f"{_fmt_cov(co.get('2sigma'))} | "
            f"{_fmt_cov(co.get('3sigma'))} | "
            f"{verdict} |"
        )
    lines.append("")

    # ──────────────────────────────────────────────────────────────
    # Per-peak χ² / Z / p tables — one table per peak, one row per
    # paradigm, one number per cell (mirrors the coverage table's
    # readability).
    # ──────────────────────────────────────────────────────────────
    lines.append("## Localized peak-region goodness-of-fit  (±5 keV window, held-out D_T)")
    lines.append("")
    lines.append(
        "Reduced χ²_DT target ≈ 1.0; p_DT > 0.5 = mean indistinguishable from "
        "sharp truth; p_DT < 0.05 = significant local miss; |Z_DT| ≫ 3 = "
        "many-σ disagreement. Z_DT is **signed**: ``Z > 0`` means the "
        "empirical rate is *above* the model (CNP under-predicts); "
        "``Z < 0`` means *below* (CNP over-predicts)."
    )
    lines.append("")
    for peak_name, label in PEAK_ORDER:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| paradigm | χ²_DT | Z_DT | p_DT | verdict |")
        lines.append("|---|---|---|---|---|")
        for nick, _ in PARADIGMS:
            pm = _peak_lookup(audits[nick], peak_name)
            if pm is None:
                lines.append(f"| {nick} | — | — | — | — |")
                continue
            verdict = _peak_verdict(pm.get("p_DT"), pm.get("chi2_DT"))
            lines.append(
                f"| {nick} | "
                f"{_fmt_chi2(pm.get('chi2_DT'))} | "
                f"{_fmt_z(pm.get('z_DT'))} | "
                f"{_fmt_p(pm.get('p_DT'))} | "
                f"{verdict} |"
            )
        lines.append("")

    # ──────────────────────────────────────────────────────────────
    # Per-peak ranking by held-out p_DT (auto-verdict).
    # ──────────────────────────────────────────────────────────────
    lines.append("## Per-peak ranking (held-out D_T)")
    lines.append("")
    lines.append(
        "Ranked by *p_DT* (high = mean indistinguishable from data). High "
        "p_DT with χ²_DT ≫ 1 means oscillation rather than tracking."
    )
    lines.append("")
    for peak_name, label in PEAK_ORDER:
        entries = []
        for nick, _ in PARADIGMS:
            pm = _peak_lookup(audits[nick], peak_name)
            if pm is None or pm.get("p_DT") is None:
                continue
            entries.append((nick, pm["p_DT"], pm["chi2_DT"], pm.get("z_DT")))
        if not entries:
            lines.append(f"**{label}** — _no cells trained yet._\n")
            continue
        entries.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"**{label}**")
        for rank, (nick, p, chi2, z) in enumerate(entries, 1):
            verdict = _peak_verdict(p, chi2)
            lines.append(
                f"  {rank}. `{nick}` — p_DT={_fmt_p(p)}, "
                f"Z_DT={_fmt_z(z)}, χ²_DT={_fmt_chi2(chi2)}  ·  {verdict}"
            )
        lines.append("")

    # ──────────────────────────────────────────────────────────────
    # Sawtooth-diagnostic suite — three roughness axes evaluated in
    # two pristine Compton-continuum windows clear of any γ-peak.
    # ──────────────────────────────────────────────────────────────
    lines.append("## Sawtooth diagnostic suite  (control regions, no γ-peaks)")
    lines.append("")
    lines.append(
        "Three complementary roughness metrics computed on the dense "
        "predicted β̂(E) curve inside two control windows. **MASD** = "
        "amplitude axis (mean |2nd difference|, scales with the size of "
        "the wiggles); **ED** = frequency axis (local extrema per keV); "
        "**ACF1** = pattern axis (Pearson correlation of successive "
        "first-differences). A smooth slope gives MASD ≈ 0, ED ≈ 0, "
        "ACF1 ≥ 0; random noise lands ACF1 ≈ −0.5; a deterministic "
        "up-down-up-down sawtooth drives ACF1 → −1. The two "
        "complementary axes (amplitude vs frequency vs pattern) "
        "distinguish structured oscillation from random wiggle even "
        "when individual metrics agree."
    )
    lines.append("")
    for label, key in SAWTOOTH_REGIONS:
        lines.append(f"### Control region {label}")
        lines.append("")
        lines.append("| paradigm | MASD | ED (keV⁻¹) | ACF1 |")
        lines.append("|---|---|---|---|")
        for nick, _ in PARADIGMS:
            a = audits[nick]
            if a is None:
                lines.append(f"| {nick} | — | — | — |")
                continue
            sw = a.get("sawtooth_metrics", {}).get(key)
            if sw is None:
                # Cell trained before the diagnostic suite was wired in —
                # JSON predates the metric write. Run inference again to
                # populate (or the inference script saved an empty dict
                # for some structural reason).
                lines.append(f"| {nick} | — | — | — |")
                continue
            masd = sw.get("masd")
            ed = sw.get("extrema_density")
            acf = sw.get("lag1_acf")
            lines.append(f"| {nick} | {masd:.4f} | {ed:.3f} | {acf:+.3f} |")
        lines.append("")

    # ──────────────────────────────────────────────────────────────
    # Global sanity check.
    # ──────────────────────────────────────────────────────────────
    lines.append("## Global sanity check")
    lines.append("")
    lines.append(
        "Spectrum-wide metrics. Pearson r close to +1 = CNP β(E) tracks D_T; "
        "mean offset close to 0 = no systematic bias."
    )
    lines.append("")
    lines.append("| paradigm | N target | N context | Pearson r | mean offset | combined 1σ |")
    lines.append("|---|---|---|---|---|---|")
    for nick, _ in PARADIGMS:
        a = audits[nick]
        if a is None:
            lines.append(f"| {nick} | — | — | — | — | — |")
            continue
        cc = a.get("coverage_combined", {})
        lines.append(
            f"| {nick} | {a.get('n_target_total', '—')} | "
            f"{a.get('n_context_total', '—')} | "
            f"{a.get('pearson_r', float('nan')):+.3f} | "
            f"{a.get('mean_offset', float('nan')):+.4f} | "
            f"{_fmt_cov(cc.get('1sigma'))} |"
        )
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}  ({len(have)}/{len(PARADIGMS)} cells loaded)")


if __name__ == "__main__":
    main()
