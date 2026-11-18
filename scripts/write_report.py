#!/usr/bin/env python3
"""Generate WRITEUP.md from comparison.json using pre-registered thresholds."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


TEMPLATE = """# Writeup: Attribution Patching vs Activation Patching on IOI

**Date:** {today}  
**Model:** GPT-2 small · **Task:** Indirect Object Identification · **N prompts:** {n_prompts}  
**Primary verdict:** **{verdict}**

## Question

Does attribution patching reproduce activation-patching head rankings on the IOI circuit in GPT-2 small, and where does the linear approximation fail?

## Setup (locked in PREREGISTRATION.md)

- Baseline: full activation patching of every head output (`hook_z`) and MLP output.
- Intervention: attribution patching (Syed et al. / Nanda).
- Refinement: EAP-IG with n_steps={n_ig_steps}.
- Metric: mean logit-diff recovery fraction.
- Thresholds: Spearman ρ(AP, AttrP) ≥ 0.80; Precision@14(AttrP) ≥ 0.50.

## Sanity checks

| Quantity | Value |
|----------|------:|
| Clean logit-diff | {clean_ld:.4f} |
| Corrupted logit-diff | {corr_ld:.4f} |
| Wall-clock AP (s) | {wall_ap:.1f} |
| Wall-clock AttrP (s) | {wall_attr:.1f} |
| Wall-clock EAP-IG (s) | {wall_eap} |
| Speedup AttrP / AP | {speedup} |

Clean logit-diff should be clearly positive and larger than corrupted (IOI working).

## Main results

| Metric | Value | Threshold | Pass? |
|--------|------:|----------:|:-----:|
| Spearman ρ(AP, AttrP) | {rho_attr:.4f} | ≥ 0.80 | {pass_rho} |
| Precision@14 AttrP | {p14_attr:.4f} | ≥ 0.50 | {pass_p14} |
| Spearman ρ(AP, EAP-IG) | {rho_eap} | — | — |
| ΔSpearman (EAP−AttrP) | {delta_rho} | ≥ 0.05 for “helps” | {eap_helps} |
| ΔPrecision@14 heads (EAP−AttrP) | {delta_p14} | ≥ 2 for “helps” | {eap_helps} |

### Precision@k

{precision_table}

### Top-14 heads

**Activation patching:** {top_ap}  
**Attribution patching:** {top_attr}  
**EAP-IG:** {top_eap}  
**Known positives H\\*:** {known}

## Disagreement localisation

Layers with Pearson r(AP, AttrP) < 0.5: {weak_layers}

MLP Pearson r(AP, AttrP): {mlp_r:.4f}{mlp_flag}

## Verdict interpretation

{interpretation}

## Limits (what this cannot tell you)

1. GPT-2 small + IOI only — no claim about larger models or other tasks.
2. We patch head *outputs* and MLP outputs, not Q/K/V or attention patterns separately.
3. H\\* is a literature checklist from prior activation-patching work; Precision@k measures recovery of that checklist, not circuit completeness.
4. Single data-generation seed (seed=0); prompt-sampling variance not quantified.
5. EAP-IG uses midpoint Riemann sum with n_steps={n_ig_steps}; more steps could change the refinement arm.

## Artifacts

- Scores: `scores.npz`
- Comparison JSON: `comparison.json`
- Heatmaps: `heatmaps.png`
- Pre-registration: `PREREGISTRATION.md`
"""


def md_bool(x: bool) -> str:
    return "yes" if x else "no"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    comp = json.loads((args.results / "comparison.json").read_text())
    meta = json.loads((args.results / "meta.json").read_text())
    v = comp["verdict"]

    weak = [
        str(row["layer"])
        for row in comp["layerwise_pearson_ap_attr"]
        if row["flag_r_lt_0.5"]
    ]
    weak_layers = ", ".join(weak) if weak else "(none)"

    mlp = comp["mlp_pearson_ap_attr"]
    mlp_flag = " ⚠ r<0.5" if mlp["flag_r_lt_0.5"] else ""

    pt_lines = ["| k | AP | AttrP | EAP-IG |", "|--:|--:|--:|--:|"]
    for k, row in sorted(comp["precision_at_k"].items(), key=lambda x: int(x[0])):
        eap = row.get("eap_ig")
        ap_v = row.get("activation_patching")
        at_v = row.get("attribution_patching")
        if ap_v is None or at_v is None:
            # v2 non-IOI: fall back to top-k overlap with AP
            ov = comp.get("topk_overlap_ap_attr", {}).get(str(k))
            eap_ov = comp.get("topk_overlap_ap_eap_ig", {}).get(str(k))
            if ov is None:
                pt_lines.append(f"| {k} | — | — | — |")
            else:
                eap_s = f"{eap_ov:.3f}" if eap_ov is not None else "—"
                pt_lines.append(f"| {k} | (overlap) | {ov:.3f} | {eap_s} |")
            continue
        eap_s = f"{eap:.3f}" if eap is not None else "—"
        pt_lines.append(
            f"| {k} | {ap_v:.3f} | {at_v:.3f} | {eap_s} |"
        )

    wall_eap = comp["wall_seconds"].get("eap_ig")
    wall_eap_s = f"{wall_eap:.1f}" if wall_eap is not None else "—"
    speedup = comp["wall_seconds"]["activation_patching"] / max(
        comp["wall_seconds"]["attribution_patching"], 1e-9
    )

    rho_eap = comp.get("spearman_ap_eap_ig")
    delta_rho = comp.get("eap_ig_delta_spearman")
    delta_p14 = comp.get("eap_ig_delta_precision14_heads")

    if v["primary"] == "GO":
        interpretation = (
            "Attribution patching meets both pre-registered bars: it ranks heads like "
            "activation patching (ρ≥0.80) and recovers at least half of the known IOI "
            f"heads in its top 14. At this scale AttrP is a **safe cheap proxy** "
            f"(≈{speedup:.0f}× faster than full AP). "
            + (
                "EAP-IG also met the improvement threshold versus AttrP."
                if v.get("eap_ig_helps")
                else "EAP-IG did **not** clear the pre-registered improvement threshold over AttrP; "
                "integrated gradients did not buy meaningful ranking fidelity here."
            )
        )
    elif v["primary"] == "PARTIAL":
        interpretation = (
            "Only one of the two AttrP bars passed. AttrP is a **partial** proxy: "
            "use it for coarse screening but confirm top heads with activation patching. "
            f"Weak layers (r<0.5): {weak_layers}. "
            + (
                "EAP-IG helped under the pre-registered rule."
                if v.get("eap_ig_helps")
                else "EAP-IG did not help under the pre-registered rule."
            )
        )
    else:
        interpretation = (
            "Attribution patching **fails** both pre-registered bars. Do not treat AttrP "
            "as a sole ranking tool for IOI heads at this setup. Inspect disagreement "
            f"localisation: weak layers = {weak_layers}. "
            + (
                "EAP-IG recovered enough to help under the pre-registered rule."
                if v.get("eap_ig_helps")
                else "EAP-IG also failed to clear the improvement threshold."
            )
        )

    text = TEMPLATE.format(
        today=date.today().isoformat(),
        n_prompts=meta["ap"].get("n_prompts", "?"),
        verdict=v["primary"],
        n_ig_steps=meta.get("eap", {}).get("n_steps", 5),
        clean_ld=comp["logit_diffs"]["clean"],
        corr_ld=comp["logit_diffs"]["corrupted"],
        wall_ap=comp["wall_seconds"]["activation_patching"],
        wall_attr=comp["wall_seconds"]["attribution_patching"],
        wall_eap=wall_eap_s,
        speedup=f"{speedup:.1f}×",
        rho_attr=v["spearman_ap_attr"],
        p14_attr=(
            v["precision_at_14_attr"]
            if v.get("precision_at_14_attr") is not None
            else v.get("topk_overlap14_attr", float("nan"))
        ),
        pass_rho=md_bool(v["attrp_safe_proxy"]),
        pass_p14=md_bool(
            v.get("attrp_recovers_circuit", v.get("attrp_recovers_topk", False))
        ),
        rho_eap=f"{rho_eap:.4f}" if rho_eap is not None else "—",
        delta_rho=f"{delta_rho:.4f}" if delta_rho is not None else "—",
        delta_p14=f"{delta_p14:.1f}" if delta_p14 is not None else "—",
        eap_helps=md_bool(bool(v.get("eap_ig_helps"))) if v.get("eap_ig_helps") is not None else "—",
        precision_table="\n".join(pt_lines),
        top_ap=", ".join(comp["top14"]["activation_patching"]),
        top_attr=", ".join(comp["top14"]["attribution_patching"]),
        top_eap=", ".join(comp["top14"].get("eap_ig", [])) or "—",
        known=", ".join(comp.get("known_positives") or []),
        weak_layers=weak_layers,
        mlp_r=mlp["pearson_r"],
        mlp_flag=mlp_flag,
        interpretation=interpretation,
    )

    out = args.out or (args.results / "WRITEUP.md")
    out.write_text(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
