#!/usr/bin/env python3
"""Aggregate pre-registered metrics across the multi-cell grid.

Supports:
  - Legacy v1 dirs: results/full_{abc,flip}_seed{n}/
  - v2 dirs: results/{model}_{task}_{lang}_{direction}_seed{n}/

Writes:
  results/seed_summary.{json,md}          (v1 back-compat if present)
  results/grid_summary.{json,md}          (v2 grid)
  results/h3_language_paired.{json,md}    (Hindi vs English paired test)
  results/grid_rho.png                    (heatmap if matplotlib available)
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
SEEDS = range(5)

CELL_RE = re.compile(
    r"^(?P<model>.+)_(?P<task>ioi|capitals|agreement|induction)_"
    r"(?P<lang>en|hi)_(?P<direction>denoising|noising)_seed(?P<seed>\d+)$"
)
LEGACY_RE = re.compile(r"^full_(?P<corr>abc|flip)_seed(?P<seed>\d+)$")


def ms(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {
        "mean": st.mean(xs),
        "sd": st.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        "n": len(xs),
    }


def load_comparison(d: Path) -> dict | None:
    f = d / "comparison.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def discover_v2_cells(root: Path) -> dict[tuple, list[tuple[int, dict]]]:
    """Map (model, task, lang, direction) -> [(seed, comparison), ...]"""
    cells: dict[tuple, list] = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = CELL_RE.match(d.name)
        if not m:
            continue
        comp = load_comparison(d)
        if comp is None:
            continue
        key = (m["model"], m["task"], m["lang"], m["direction"])
        cells[key].append((int(m["seed"]), comp))
    return cells


def aggregate_v1(root: Path) -> dict:
    summary = {}
    for cond in ("abc", "flip"):
        rows = []
        for s in SEEDS:
            c = load_comparison(root / f"full_{cond}_seed{s}")
            if c:
                rows.append(c)
        if not rows:
            continue
        g = lambda fn: [fn(c) for c in rows]
        summary[cond] = {
            "n_seeds": len(rows),
            "clean_logit_diff": ms(g(lambda c: c["logit_diffs"]["clean"])),
            "corrupted_logit_diff": ms(g(lambda c: c["logit_diffs"]["corrupted"])),
            "spearman_ap_attr": ms(g(lambda c: c["spearman_ap_attr"])),
            "spearman_ap_eap": ms(g(lambda c: c.get("spearman_ap_eap_ig"))),
            "p14_ap": ms(g(lambda c: c["precision_at_k"]["14"]["activation_patching"])),
            "p14_attr": ms(g(lambda c: c["precision_at_k"]["14"]["attribution_patching"])),
            "p14_eap": ms(g(lambda c: c["precision_at_k"]["14"].get("eap_ig"))),
            "mlp_r_attr": ms(g(lambda c: c["mlp_pearson_ap_attr"]["pearson_r"])),
            "mlp_r_eap": ms(g(lambda c: c.get("mlp_pearson_ap_eap_ig", {}).get("pearson_r"))),
            "speedup_attr": ms(
                g(
                    lambda c: c["wall_seconds"]["activation_patching"]
                    / c["wall_seconds"]["attribution_patching"]
                )
            ),
            "verdicts": {
                v: sum(1 for c in rows if c["verdict"]["primary"] == v)
                for v in ("GO", "PARTIAL", "NO-GO")
            },
            "eap_helps_count": sum(1 for c in rows if c["verdict"].get("eap_ig_helps")),
            "min_layer_r_attr": ms(
                g(lambda c: min(r["pearson_r"] for r in c["layerwise_pearson_ap_attr"]))
            ),
        }
    return summary


def aggregate_v2(cells: dict) -> dict:
    grid = {}
    for key, seed_rows in sorted(cells.items()):
        model, task, lang, direction = key
        rows = [c for _, c in sorted(seed_rows)]
        g = lambda fn: [fn(c) for c in rows]
        cell_id = f"{model}|{task}|{lang}|{direction}"
        grid[cell_id] = {
            "model": model,
            "task": task,
            "lang": lang,
            "direction": direction,
            "n_seeds": len(rows),
            "spearman_ap_attr": ms(g(lambda c: c["spearman_ap_attr"])),
            "restricted_rho": ms(
                g(lambda c: c.get("restricted_spearman_ap_attr", {}).get("rho"))
            ),
            "topk_overlap14": ms(g(lambda c: c.get("topk_overlap_ap_attr", {}).get("14"))),
            "spearman_ap_eap": ms(g(lambda c: c.get("spearman_ap_eap_ig"))),
            "mlp_r_attr": ms(g(lambda c: c["mlp_pearson_ap_attr"]["pearson_r"])),
            "mlp_r_eap": ms(g(lambda c: c.get("mlp_pearson_ap_eap_ig", {}).get("pearson_r"))),
            "clean_logit_diff": ms(g(lambda c: c["logit_diffs"]["clean"])),
            "corrupted_logit_diff": ms(g(lambda c: c["logit_diffs"]["corrupted"])),
            "speedup_attr": ms(
                g(
                    lambda c: c["wall_seconds"]["activation_patching"]
                    / max(c["wall_seconds"]["attribution_patching"], 1e-9)
                )
            ),
            "verdicts": {
                v: sum(1 for c in rows if c["verdict"]["primary"] == v)
                for v in ("GO", "PARTIAL", "NO-GO")
            },
            "eap_helps_count": sum(1 for c in rows if c["verdict"].get("eap_ig_helps")),
            "h1_pass": all(c["spearman_ap_attr"] >= 0.90 for c in rows),
            "h2_mlp_neg": ms(g(lambda c: c["mlp_pearson_ap_attr"]["pearson_r"]))["mean"] is not None
            and (ms(g(lambda c: c["mlp_pearson_ap_attr"]["pearson_r"]))["mean"] or 0) < 0,
        }
    return grid


def paired_bootstrap_h3(cells: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """For each (model, task, direction) with both en and hi, test mean(rho_hi - rho_en) < 0."""
    rng = random.Random(seed)
    # group by (model, task, direction, lang)
    by = defaultdict(dict)
    for (model, task, lang, direction), seed_rows in cells.items():
        rhos = [c["spearman_ap_attr"] for _, c in sorted(seed_rows)]
        by[(model, task, direction)][lang] = rhos

    out = {"pairs": [], "n_boot": n_boot}
    for key, langs in sorted(by.items()):
        if "en" not in langs or "hi" not in langs:
            continue
        en, hi = langs["en"], langs["hi"]
        n = min(len(en), len(hi))
        if n == 0:
            continue
        en, hi = en[:n], hi[:n]
        diffs = [hi[i] - en[i] for i in range(n)]
        mean_diff = st.mean(diffs)
        # bootstrap CI on mean diff
        boots = []
        for _ in range(n_boot):
            sample = [diffs[rng.randrange(n)] for _ in range(n)]
            boots.append(st.mean(sample))
        boots.sort()
        lo = boots[int(0.025 * n_boot)]
        hi_ci = boots[int(0.975 * n_boot)]
        # one-sided P(mean_diff >= 0) under bootstrap for H3 (rho_hi < rho_en)
        p_ge0 = sum(1 for b in boots if b >= 0) / n_boot
        out["pairs"].append(
            {
                "model": key[0],
                "task": key[1],
                "direction": key[2],
                "n_seeds": n,
                "mean_rho_en": st.mean(en),
                "mean_rho_hi": st.mean(hi),
                "mean_diff_hi_minus_en": mean_diff,
                "ci95": [lo, hi_ci],
                "bootstrap_p_diff_ge_0": p_ge0,
                "h3_supported": mean_diff < 0 and hi_ci < 0,
            }
        )
    return out


def fmt(d):
    if not d or d.get("mean") is None:
        return "—"
    return f"{d['mean']:.3f} ± {d['sd']:.3f} [{d['min']:.2f}, {d['max']:.2f}]"


def write_v1_md(summary: dict, path: Path) -> None:
    if not summary:
        return
    lines = ["| Metric | ABC | Role-flip |", "|---|---|---|"]
    for k, label in [
        ("clean_logit_diff", "Clean logit-diff"),
        ("corrupted_logit_diff", "Corrupted logit-diff"),
        ("spearman_ap_attr", "rho(AP, AttrP)"),
        ("spearman_ap_eap", "rho(AP, EAP-IG)"),
        ("p14_ap", "Precision@14 AP"),
        ("p14_attr", "Precision@14 AttrP"),
        ("p14_eap", "Precision@14 EAP-IG"),
        ("mlp_r_attr", "MLP Pearson r (AttrP)"),
        ("mlp_r_eap", "MLP Pearson r (EAP-IG)"),
        ("min_layer_r_attr", "Worst per-layer r (AttrP)"),
        ("speedup_attr", "AttrP speedup (x)"),
    ]:
        a = summary.get("abc", {}).get(k)
        b = summary.get("flip", {}).get(k)
        if a is None and b is None:
            continue
        lines.append(f"| {label} | {fmt(a) if a else '—'} | {fmt(b) if b else '—'} |")
    if "abc" in summary:
        lines.append(
            f"| Verdicts (GO/PARTIAL/NO-GO) | {summary['abc']['verdicts']} | "
            f"{summary.get('flip', {}).get('verdicts', {})} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_grid_md(grid: dict, path: Path) -> None:
    lines = [
        "# Grid summary (AttrP vs AP)",
        "",
        "| Model | Task | Lang | Dir | n | rho(AP,AttrP) | restr. rho | top14 overlap | MLP r AttrP | MLP r EAP | Verdicts | H1 |",
        "|---|---|:---:|:---:|---:|---|---|---|---|---|---|:---:|",
    ]
    for cell_id, g in sorted(grid.items()):
        lines.append(
            f"| `{g['model']}` | {g['task']} | {g['lang']} | {g['direction']} | {g['n_seeds']} | "
            f"{fmt(g['spearman_ap_attr'])} | {fmt(g['restricted_rho'])} | {fmt(g['topk_overlap14'])} | "
            f"{fmt(g['mlp_r_attr'])} | {fmt(g['mlp_r_eap'])} | {g['verdicts']} | {g['h1_pass']} |"
        )
    if not grid:
        lines.append("")
        lines.append("_No v2 result dirs found yet. Run the grid on Ada first._")
    path.write_text("\n".join(lines) + "\n")


def write_h3_md(h3: dict, path: Path) -> None:
    lines = [
        "# H3: paired Hindi vs English AttrP–AP agreement",
        "",
        "| Model | Task | Dir | mean ρ_en | mean ρ_hi | Δ (hi−en) | 95% CI | H3 supported |",
        "|---|---|:---:|---:|---:|---:|---|:---:|",
    ]
    for p in h3.get("pairs", []):
        lines.append(
            f"| `{p['model']}` | {p['task']} | {p['direction']} | {p['mean_rho_en']:.3f} | "
            f"{p['mean_rho_hi']:.3f} | {p['mean_diff_hi_minus_en']:.3f} | "
            f"[{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}] | {p['h3_supported']} |"
        )
    if not h3.get("pairs"):
        lines.append("")
        lines.append("_No paired en/hi cells with results yet._")
    path.write_text("\n".join(lines) + "\n")


def try_plot_grid(grid: dict, path: Path) -> None:
    if not grid:
        return
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    # Build matrix for denoising primary: rows=model|task, cols=lang
    denoise = {k: v for k, v in grid.items() if v["direction"] == "denoising"}
    if not denoise:
        denoise = grid
    row_keys = sorted({(v["model"], v["task"]) for v in denoise.values()})
    langs = ["en", "hi"]
    mat = np.full((len(row_keys), len(langs)), np.nan)
    for i, (model, task) in enumerate(row_keys):
        for j, lang in enumerate(langs):
            for v in denoise.values():
                if v["model"] == model and v["task"] == task and v["lang"] == lang:
                    m = v["spearman_ap_attr"].get("mean")
                    if m is not None:
                        mat[i, j] = m
    fig, ax = plt.subplots(figsize=(4, max(3, 0.35 * len(row_keys) + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{m.split('/')[-1]}|{t}" for m, t in row_keys], fontsize=8)
    ax.set_title("ρ(AP, AttrP) mean over seeds")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    v1 = aggregate_v1(results_root)
    if v1:
        (results_root / "seed_summary.json").write_text(json.dumps(v1, indent=2) + "\n")
        write_v1_md(v1, results_root / "seed_summary.md")
        print("=== v1 seed summary ===")
        print((results_root / "seed_summary.md").read_text())

    cells = discover_v2_cells(results_root)
    grid = aggregate_v2(cells)
    (results_root / "grid_summary.json").write_text(json.dumps(grid, indent=2) + "\n")
    write_grid_md(grid, results_root / "grid_summary.md")
    print("=== v2 grid summary ===")
    print((results_root / "grid_summary.md").read_text())

    h3 = paired_bootstrap_h3(cells)
    (results_root / "h3_language_paired.json").write_text(json.dumps(h3, indent=2) + "\n")
    write_h3_md(h3, results_root / "h3_language_paired.md")
    print("=== H3 paired language ===")
    print((results_root / "h3_language_paired.md").read_text())

    try_plot_grid(grid, results_root / "grid_rho.png")
    print(f"Wrote summaries under {results_root}")


if __name__ == "__main__":
    main()
