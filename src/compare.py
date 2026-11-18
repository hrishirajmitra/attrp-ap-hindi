"""Compare AP vs AttrP vs EAP-IG: Spearman, top-k overlap, layer-wise localisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .known_circuit import POSITIVE_HEADS, head_label, positive_labels
from .registry import RESTRICTED_AP_ABS_THRESHOLD, TOPK_OVERLAP


def flatten_heads(scores: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """scores: [L, H] → flat vector + labels."""
    labels = []
    vals = []
    for l in range(scores.shape[0]):
        for h in range(scores.shape[1]):
            labels.append(head_label(l, h))
            vals.append(float(scores[l, h]))
    return np.asarray(vals, dtype=np.float64), labels


def precision_at_k(scores: np.ndarray, k: int, positives: set[str] | None = None) -> float:
    positives = positives or positive_labels()
    flat, labels = flatten_heads(scores)
    order = np.argsort(-flat)
    top = {labels[i] for i in order[:k]}
    return len(top & positives) / float(k)


def top_k_labels(scores: np.ndarray, k: int, by_abs: bool = False) -> list[str]:
    flat, labels = flatten_heads(scores)
    key = np.abs(flat) if by_abs else flat
    order = np.argsort(-key, kind="stable")
    return [labels[i] for i in order[:k]]


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int, by_abs: bool = True) -> float:
    """Fraction of top-k labels of a that also appear in top-k of b.

    by_abs=True (primary, matches the proposal): rank by |score|; False: rank by signed score.
    """
    top_a = set(top_k_labels(a, k, by_abs))
    top_b = set(top_k_labels(b, k, by_abs))
    return len(top_a & top_b) / float(k)


def effect_concentration(ap_heads: np.ndarray, k: int = 14) -> float:
    """Share of total |AP| head effect carried by the k largest-|AP| heads."""
    flat = np.abs(ap_heads.ravel())
    tot = float(flat.sum())
    if tot <= 0:
        return float("nan")
    return float(np.sort(flat)[::-1][:k].sum() / tot)


def spearman_heads(a: np.ndarray, b: np.ndarray) -> float:
    fa, _ = flatten_heads(a)
    fb, _ = flatten_heads(b)
    rho, _ = spearmanr(fa, fb)
    return float(rho)


def spearman_restricted(
    a: np.ndarray,
    b: np.ndarray,
    abs_threshold: float = RESTRICTED_AP_ABS_THRESHOLD,
) -> dict[str, Any]:
    """Spearman on heads with |a| > threshold (a = AP scores). A priori threshold."""
    fa, labels = flatten_heads(a)
    fb, _ = flatten_heads(b)
    mask = np.abs(fa) > abs_threshold
    n = int(mask.sum())
    if n < 3:
        return {
            "rho": float("nan"),
            "n_heads": n,
            "abs_threshold": abs_threshold,
            "labels": [labels[i] for i in np.where(mask)[0]],
        }
    rho, _ = spearmanr(fa[mask], fb[mask])
    return {
        "rho": float(rho),
        "n_heads": n,
        "abs_threshold": abs_threshold,
        "labels": [labels[i] for i in np.where(mask)[0]],
    }


def pearson_heads(a: np.ndarray, b: np.ndarray) -> float:
    fa, _ = flatten_heads(a)
    fb, _ = flatten_heads(b)
    r, _ = pearsonr(fa, fb)
    return float(r)


def layerwise_pearson(a: np.ndarray, b: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for l in range(a.shape[0]):
        r, p = pearsonr(a[l], b[l])
        rows.append({"layer": l, "pearson_r": float(r), "p_value": float(p), "flag_r_lt_0.5": bool(r < 0.5)})
    return rows


def mlp_pearson(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    r, p = pearsonr(a, b)
    return {"pearson_r": float(r), "p_value": float(p), "flag_r_lt_0.5": bool(r < 0.5)}


def compare_all(
    ap: dict[str, Any],
    attr: dict[str, Any],
    eap: dict[str, Any] | None = None,
    ks: tuple[int, ...] = TOPK_OVERLAP,
    task: str = "ioi",
    use_ioi_checklist: bool = True,
) -> dict[str, Any]:
    ap_h, attr_h = ap["head_scores"], attr["head_scores"]
    out: dict[str, Any] = {
        "spearman_ap_attr": spearman_heads(ap_h, attr_h),
        "pearson_ap_attr": pearson_heads(ap_h, attr_h),
        "restricted_spearman_ap_attr": spearman_restricted(ap_h, attr_h),
        "topk_overlap_ap_attr": {str(k): top_k_overlap(ap_h, attr_h, k) for k in ks},
        "topk_overlap_signed_ap_attr": {
            str(k): top_k_overlap(ap_h, attr_h, k, by_abs=False) for k in ks
        },
        "effect_concentration_top14": effect_concentration(ap_h, 14),
        "precision_at_k": {},
        "layerwise_pearson_ap_attr": layerwise_pearson(ap_h, attr_h),
        "mlp_pearson_ap_attr": mlp_pearson(ap["mlp_scores"], attr["mlp_scores"]),
        "wall_seconds": {
            "activation_patching": ap["wall_seconds"],
            "attribution_patching": attr["wall_seconds"],
        },
        "logit_diffs": {
            "clean": ap["clean_logit_diff"],
            "corrupted": ap["corrupted_logit_diff"],
        },
        "top14": {
            "activation_patching": top_k_labels(ap_h, 14),
            "attribution_patching": top_k_labels(attr_h, 14),
        },
        "direction": ap.get("direction", "denoising"),
        "task": task,
    }

    # IOI checklist precision (only meaningful for GPT-2-small IOI)
    if use_ioi_checklist and task == "ioi":
        out["known_positives"] = [head_label(l, h) for l, h in POSITIVE_HEADS]
        for k in ks:
            out["precision_at_k"][str(k)] = {
                "activation_patching": precision_at_k(ap_h, k),
                "attribution_patching": precision_at_k(attr_h, k),
            }
    else:
        out["known_positives"] = []
        for k in ks:
            out["precision_at_k"][str(k)] = {
                "activation_patching": None,
                "attribution_patching": None,
            }

    if eap is not None:
        eap_h = eap["head_scores"]
        out["spearman_ap_eap_ig"] = spearman_heads(ap_h, eap_h)
        out["pearson_ap_eap_ig"] = pearson_heads(ap_h, eap_h)
        out["restricted_spearman_ap_eap_ig"] = spearman_restricted(ap_h, eap_h)
        out["topk_overlap_ap_eap_ig"] = {str(k): top_k_overlap(ap_h, eap_h, k) for k in ks}
        out["topk_overlap_signed_ap_eap_ig"] = {
            str(k): top_k_overlap(ap_h, eap_h, k, by_abs=False) for k in ks
        }
        out["layerwise_pearson_ap_eap_ig"] = layerwise_pearson(ap_h, eap_h)
        out["mlp_pearson_ap_eap_ig"] = mlp_pearson(ap["mlp_scores"], eap["mlp_scores"])
        out["wall_seconds"]["eap_ig"] = eap["wall_seconds"]
        out["top14"]["eap_ig"] = top_k_labels(eap_h, 14)
        out["eap_n_steps"] = eap.get("n_steps")
        if use_ioi_checklist and task == "ioi":
            for k in ks:
                out["precision_at_k"][str(k)]["eap_ig"] = precision_at_k(eap_h, k)
            out["eap_ig_delta_precision14_heads"] = (
                out["precision_at_k"]["14"]["eap_ig"] * 14
                - out["precision_at_k"]["14"]["attribution_patching"] * 14
            )
        else:
            for k in ks:
                out["precision_at_k"][str(k)]["eap_ig"] = None
            out["eap_ig_delta_precision14_heads"] = None
        out["eap_ig_delta_spearman"] = out["spearman_ap_eap_ig"] - out["spearman_ap_attr"]

    # Verdicts — v2 primary uses rho + top-14 overlap with AP (not checklist)
    rho = out["spearman_ap_attr"]
    overlap14 = out["topk_overlap_ap_attr"]["14"]
    safe_proxy = rho >= 0.90  # H1 threshold in v2
    recovers = overlap14 >= 0.50
    if safe_proxy and recovers:
        verdict = "GO"
    elif safe_proxy or recovers:
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"

    eap_helps = None
    if eap is not None:
        mlp_attr_r = out["mlp_pearson_ap_attr"]["pearson_r"]
        mlp_eap_r = out["mlp_pearson_ap_eap_ig"]["pearson_r"]
        eap_helps = bool(
            out["eap_ig_delta_spearman"] >= 0.05
            or (mlp_attr_r < 0 and mlp_eap_r > 0.5)  # H4: repairs MLP
        )

    out["verdict"] = {
        "primary": verdict,
        "attrp_safe_proxy": safe_proxy,
        "attrp_recovers_topk": recovers,
        "spearman_ap_attr": rho,
        "topk_overlap14_attr": overlap14,
        "restricted_rho": out["restricted_spearman_ap_attr"]["rho"],
        "eap_ig_helps": eap_helps,
        "thresholds": {
            "spearman_ge": 0.90,
            "topk_overlap14_ge": 0.50,
            "restricted_ap_abs": RESTRICTED_AP_ABS_THRESHOLD,
            "eap_delta_spearman_ge": 0.05,
        },
        # Keep v1 fields for IOI back-compat reporting
        "precision_at_14_attr": (
            out["precision_at_k"]["14"]["attribution_patching"]
            if out["precision_at_k"]["14"]["attribution_patching"] is not None
            else None
        ),
    }
    return out


def plot_heatmaps(
    ap: np.ndarray,
    attr: np.ndarray,
    eap: np.ndarray | None,
    out_path: Path,
) -> None:
    mats = [("Activation patching", ap), ("Attribution patching", attr)]
    if eap is not None:
        mats.append(("EAP-IG", eap))
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, (title, m) in zip(axes[0], mats):
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-np.max(np.abs(m)), vmax=np.max(np.abs(m)))
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_results(
    out_dir: Path,
    ap: dict[str, Any],
    attr: dict[str, Any],
    comparison: dict[str, Any],
    eap: dict[str, Any] | None = None,
    eap_sweep: dict[int, dict[str, Any]] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ap_heads": ap["head_scores"],
        "ap_mlps": ap["mlp_scores"],
        "attr_heads": attr["head_scores"],
        "attr_mlps": attr["mlp_scores"],
    }
    if eap is not None:
        payload["eap_heads"] = eap["head_scores"]
        payload["eap_mlps"] = eap["mlp_scores"]
    if eap_sweep:
        for steps, res in eap_sweep.items():
            payload[f"eap{steps}_heads"] = res["head_scores"]
            payload[f"eap{steps}_mlps"] = res["mlp_scores"]
    np.savez(out_dir / "scores.npz", **payload)

    path = out_dir / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")

    meta = {
        "ap": {k: v for k, v in ap.items() if k not in ("head_scores", "mlp_scores")},
        "attr": {k: v for k, v in attr.items() if k not in ("head_scores", "mlp_scores")},
    }
    if eap is not None:
        meta["eap"] = {k: v for k, v in eap.items() if k not in ("head_scores", "mlp_scores")}
    if eap_sweep:
        meta["eap_sweep_steps"] = sorted(eap_sweep.keys())
        meta["eap_sweep"] = {
            str(s): {k: v for k, v in res.items() if k not in ("head_scores", "mlp_scores")}
            for s, res in eap_sweep.items()
        }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    plot_heatmaps(
        ap["head_scores"],
        attr["head_scores"],
        eap["head_scores"] if eap is not None else None,
        out_dir / "heatmaps.png",
    )
