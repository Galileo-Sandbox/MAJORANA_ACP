"""Build the localized peak-region comparison table across paradigms.

Loads ``test_set_audit.json`` from each (model, paradigm, bin, class) cell
and emits a markdown table at ``analysis/cnp_audit/_peak_comparison.md``
showing reduced χ² / two-tailed p across **D_C** and **D_T** for the four
γ-peak markers, side-by-side across paradigms.

The table layout (per the Phase-3 spec in CLAUDE.md):

    peak     | true_cnp                | w10_fixed48              | ...
             | χ²_DC χ²_DT  p_DC  p_DT | χ²_DC χ²_DT  p_DC  p_DT | ...
    FE 2614  |  1.23  1.85  0.20  0.04 |  ...

Cells whose JSON is missing render as ``— —  — —``; that lets the
script run incrementally as training fills in cells.
"""

from __future__ import annotations

import json
from pathlib import Path

# (nickname, results-tree paradigm path under analysis/cnp_audit/<model>/)
PARADIGMS = [
    ("true_cnp", "true_cnp"),
    ("w10_fixed48", "hybrid_scale/mixed_density_f0_70_w10"),
    ("w10_varN_large", "hybrid_scale/mixed_density_f0_70_w10_varN32-1024"),
    ("physics_w10_varN", "hybrid_scale/physics_anchored_f0_80_w10_varN32-1024"),
    ("hyper_zoom_w5", "hybrid_scale/mixed_density_f0_85_w5_varN32-1024"),
]

PEAK_ORDER = [
    ("Tl-208 FE", 2614.0, "FE 2614"),
    ("Tl-208 SE", 2103.0, "SE 2103"),
    ("Tl-208 DEP", 1592.0, "DEP 1592"),
    ("Bi-214 1620", 1620.0, "Bi 1620"),
]

AUDIT_ROOT = Path("analysis/cnp_audit")
MODEL = "simple_cnn_small"
BIN_DIR = "bin10"
CLS = "signal"
OUT_PATH = AUDIT_ROOT / "_peak_comparison.md"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def _fmt_chi2(x: float | None) -> str:
    return "—" if x is None else f"{x:6.2f}"


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


def main() -> None:
    audits = {nick: _load(rel) for nick, rel in PARADIGMS}
    have = [nick for nick, a in audits.items() if a is not None]
    missing = [nick for nick, a in audits.items() if a is None]

    lines: list[str] = []
    lines.append("# Localized peak-region comparison")
    lines.append("")
    lines.append(
        f"Source: `{AUDIT_ROOT}/{MODEL}/<paradigm>/{BIN_DIR}/{CLS}/"
        "test_set_audit.json` for each paradigm."
    )
    lines.append("")
    lines.append(
        "Half-window = ±5 keV around each γ-peak. Reduced χ² target ≈ 1.0; "
        "p_DT > 0.5 = indistinguishable from sharp truth; p_DT < 0.05 = "
        "significant local miss. D_C measures match to the conditioning set, "
        "D_T measures generalization to held-out events."
    )
    lines.append("")
    if missing:
        lines.append("**Missing cells (rendered as `—`):** " + ", ".join(missing))
        lines.append("")
    lines.append(f"Loaded cells: {', '.join(have) if have else '(none)'}")
    lines.append("")

    # Header row: peak | paradigm1 | paradigm2 | ...
    header_cells = ["peak"] + [nick for nick, _ in PARADIGMS]
    sub_cells = [""] + ["χ²_DC  χ²_DT   p_DC    p_DT" for _ in PARADIGMS]
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
    lines.append("| " + " | ".join(sub_cells) + " |")

    for peak_name, _peak_e_kev, label in PEAK_ORDER:
        row_cells = [label]
        for nick, _ in PARADIGMS:
            pm = _peak_lookup(audits[nick], peak_name)
            if pm is None:
                row_cells.append("—  —     —     —")
                continue
            cell = (
                f"{_fmt_chi2(pm.get('chi2_DC'))} "
                f"{_fmt_chi2(pm.get('chi2_DT'))}  "
                f"{_fmt_p(pm.get('p_DC'))} "
                f"{_fmt_p(pm.get('p_DT'))}"
            )
            row_cells.append(cell)
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append("## Paradigm legend")
    lines.append("")
    lines.append(
        "- `true_cnp` — `flat_stratified`, fixed N=48. The original True-CNP "
        "baseline (bin-uniform context, no focus window)."
    )
    lines.append("- `w10_fixed48` — `mixed_density`, 10 keV window, 70% local, fixed N=48.")
    lines.append(
        "- `w10_varN_large` — `mixed_density`, 10 keV window, 70% local, "
        "per-step variable N ∈ [32, 1024], n_context_max=1023."
    )
    lines.append(
        "- `physics_w10_varN` — `physics_anchored` on {Tl-208 DEP/SE/FE, "
        "Bi-214 1620}, 10 keV window, 80% local, variable N ∈ [32, 1024]."
    )
    lines.append(
        "- `hyper_zoom_w5` — `mixed_density`, 5 keV window (HPGe FWHM scale), "
        "85% local, variable N ∈ [32, 1024]."
    )

    lines.append("")
    lines.append("## Per-peak ranking (held-out D_T)")
    lines.append("")
    lines.append(
        "Ranked by *p_DT* (high = mean indistinguishable from data). Note: "
        "p_DT measures only the *mean* offset over the window; reduced χ²_DT "
        "near 1.0 is the stricter local goodness-of-fit. A cell with high "
        "p_DT and χ²_DT ≫ 1 (e.g. Bi 1620 for physics_w10_varN, χ²=4.33) is "
        "oscillating around the truth rather than tracking it."
    )
    lines.append("")
    for peak_name, _, label in PEAK_ORDER:
        entries = []
        for nick, _ in PARADIGMS:
            pm = _peak_lookup(audits[nick], peak_name)
            if pm is None or pm.get("p_DT") is None:
                continue
            entries.append((nick, pm["p_DT"], pm["chi2_DT"]))
        if not entries:
            continue
        entries.sort(key=lambda x: x[1], reverse=True)
        lines.append(f"**{label}**")
        for rank, (nick, p, chi2) in enumerate(entries, 1):
            verdict = (
                "✓ clean"
                if p > 0.5 and chi2 < 2
                else ("△ mean OK, χ² high (oscillation)" if p > 0.5 else "✗ local miss")
            )
            lines.append(
                f"  {rank}. `{nick}` — p_DT={_fmt_p(p)}, χ²_DT={_fmt_chi2(chi2)}  ·  {verdict}"
            )
        lines.append("")

    # Global sanity-check block: Pearson r, combined-σ coverage at 1σ/2σ/3σ.
    lines.append("## Global sanity check")
    lines.append("")
    lines.append(
        "These are spectrum-wide metrics from the same audit JSONs. They "
        "tell you whether a paradigm earned its better peak fits honestly "
        "(tighter r, calibrated coverage) or by inflating σ_CNP."
    )
    lines.append("")
    lines.append("| paradigm | Pearson r | mean offset | combined cov 1σ | 2σ | 3σ |")
    lines.append("|---|---|---|---|---|---|")
    for nick, _ in PARADIGMS:
        a = audits[nick]
        if a is None:
            lines.append(f"| {nick} | — | — | — | — | — |")
            continue
        cov = a.get("coverage_combined", {})
        lines.append(
            f"| {nick} | {a.get('pearson_r', float('nan')):+.3f} | "
            f"{a.get('mean_offset', float('nan')):+.4f} | "
            f"{cov.get('1sigma', float('nan')):.3f} | "
            f"{cov.get('2sigma', float('nan')):.3f} | "
            f"{cov.get('3sigma', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append(
        "Target combined coverage: 1σ ≈ 0.683, 2σ ≈ 0.954, 3σ ≈ 0.997. "
        "A 1σ value much above 0.683 means uncertainty bands are too wide; "
        "much below means the model is overconfident."
    )
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}  ({len(have)}/{len(PARADIGMS)} cells loaded)")


if __name__ == "__main__":
    main()
