#!/usr/bin/env python3
"""Stage 2: compare important-head profiles across English vs Hindi.

For each model × task with both en and hi result dirs:
  - Spearman / Pearson of AP head-score vectors (en vs hi)
  - Split-half ceiling within each language (seed 0 vs seed 1 if present; else
    random half-split of heads is NOT used — we need two runs)
  - Different-task floor: correlate en-taskA with en-taskB when available
  - Top-k overlap
  - Layer profile of |AP| mass

Only reports cells; does not claim "circuits".
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # offline / no venv
    np = None  # type: ignore
    pearsonr = spearmanr = None  # type: ignore
CELL_RE = re.compile(
    r"^(?P<model>.+)_(?P<task>ioi|capitals|agreement|agreement_ne|induction)_"
    r"(?P<lang>en|hi|hi_roman)_(?P<direction>denoising|noising)_seed(?P<seed>\d+)$"
)


def load_ap_heads(d: Path):
    if np is None:
        return None
    f = d / "scores.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if "ap_heads" not in z:
        return None
    return z["ap_heads"]


def discover(root: Path) -> dict:
    """(model, task, lang, direction) -> {seed: heads}"""
    out: dict = defaultdict(dict)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = CELL_RE.match(d.name)
        if not m:
            continue
        heads = load_ap_heads(d)
        if heads is None:
            continue
        key = (m["model"], m["task"], m["lang"], m["direction"])
        out[key][int(m["seed"])] = heads
    return out


def mean_over_seeds(seed_map: dict):
    mats = [seed_map[s] for s in sorted(seed_map)]
    return np.mean(np.stack(mats, axis=0), axis=0)


def top_k_overlap(a, b, k: int) -> float:
    ia = np.argsort(-a.ravel())[:k]
    ib = set(np.argsort(-b.ravel())[:k].tolist())
    return len([i for i in ia if i in ib]) / float(k)


def layer_mass(heads) -> list[float]:
    """Fraction of total |AP| mass per layer."""
    abs_h = np.abs(heads)
    total = float(abs_h.sum()) + 1e-12
    return (abs_h.sum(axis=1) / total).tolist()


def corr(a, b) -> dict:
    fa, fb = a.ravel(), b.ravel()
    rho, rp = spearmanr(fa, fb)
    r, pp = pearsonr(fa, fb)
    return {
        "spearman": float(rho),
        "spearman_p": float(rp),
        "pearson": float(r),
        "pearson_p": float(pp),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--k", type=int, default=14)
    p.add_argument("--direction", default="denoising")
    args = p.parse_args()
    out_dir = args.out or args.results
    out_dir.mkdir(parents=True, exist_ok=True)

    if np is None:
        payload = {
            "direction": args.direction,
            "n_pairs": 0,
            "pairs": [],
            "cross_task_floors": [],
            "status": "skipped",
            "reason": "numpy/scipy not installed (run on Ada venv)",
            "note": "Important-head profiles under AP, not full circuits.",
        }
        (out_dir / "language_compare.json").write_text(json.dumps(payload, indent=2) + "\n")
        md = (
            "# Language comparison (AP head profiles)\n\n"
            "_Skipped locally (no numpy). Run on Ada after stage 1._\n"
        )
        (out_dir / "language_compare.md").write_text(md)
        print(md)
        return

    cells = discover(args.results)
    # Index by (model, task, direction) -> lang -> seed_map
    by: dict = defaultdict(dict)
    for (model, task, lang, direction), seed_map in cells.items():
        if direction != args.direction:
            continue
        by[(model, task, direction)][lang] = seed_map

    pairs = []
    for (model, task, direction), langs in sorted(by.items()):
        if "en" not in langs or "hi" not in langs:
            continue
        en_mean = mean_over_seeds(langs["en"])
        hi_mean = mean_over_seeds(langs["hi"])
        row = {
            "model": model,
            "task": task,
            "direction": direction,
            "n_seeds_en": len(langs["en"]),
            "n_seeds_hi": len(langs["hi"]),
            "en_vs_hi": corr(en_mean, hi_mean),
            f"topk_overlap_{args.k}": top_k_overlap(en_mean, hi_mean, args.k),
            "layer_mass_en": layer_mass(en_mean),
            "layer_mass_hi": layer_mass(hi_mean),
        }
        # Split-half ceiling: seed 0 vs seed 1 within language
        for lang in ("en", "hi"):
            sm = langs[lang]
            if 0 in sm and 1 in sm:
                row[f"ceiling_{lang}_seed0_vs_1"] = corr(sm[0], sm[1])
            else:
                seeds = sorted(sm)
                if len(seeds) >= 2:
                    row[f"ceiling_{lang}_seed{seeds[0]}_vs_{seeds[1]}"] = corr(
                        sm[seeds[0]], sm[seeds[1]]
                    )
                else:
                    row[f"ceiling_{lang}"] = None
        # Romanized stretch
        if "hi_roman" in langs:
            rom = mean_over_seeds(langs["hi_roman"])
            row["en_vs_hi_roman"] = corr(en_mean, rom)
            row["hi_vs_hi_roman"] = corr(hi_mean, rom)
        pairs.append(row)

    # Different-task floor: for each model, correlate en heads across tasks
    floors = []
    tasks_by_model: dict = defaultdict(dict)
    for (model, task, direction), langs in by.items():
        if "en" in langs:
            tasks_by_model[model][task] = mean_over_seeds(langs["en"])
    for model, task_map in sorted(tasks_by_model.items()):
        tnames = sorted(task_map)
        for i, t1 in enumerate(tnames):
            for t2 in tnames[i + 1 :]:
                floors.append(
                    {
                        "model": model,
                        "task_a": t1,
                        "task_b": t2,
                        "en_cross_task": corr(task_map[t1], task_map[t2]),
                    }
                )

    # H5-style verdict helpers
    for row in pairs:
        ceil_en = row.get("ceiling_en_seed0_vs_1") or row.get(
            next((k for k in row if k.startswith("ceiling_en_seed")), ""), None
        )
        # floor: mean of cross-task correlations for this model
        model_floors = [
            f["en_cross_task"]["spearman"]
            for f in floors
            if f["model"] == row["model"] and f["en_cross_task"]["spearman"] is not None
        ]
        floor_mean = float(np.mean(model_floors)) if model_floors else None
        rho = row["en_vs_hi"]["spearman"]
        ceil_rho = ceil_en["spearman"] if isinstance(ceil_en, dict) else None
        row["h5"] = {
            "rho_en_hi": rho,
            "ceiling_en": ceil_rho,
            "floor_cross_task_mean": floor_mean,
            "near_ceiling": (
                ceil_rho is not None and rho >= 0.9 * ceil_rho if ceil_rho else None
            ),
            "above_floor": (floor_mean is not None and rho > floor_mean),
        }
        # H6: early/late residual — compare layer mass L1 distance in first 2 + last 2 vs mid
        le, lh = np.asarray(row["layer_mass_en"]), np.asarray(row["layer_mass_hi"])
        L = len(le)
        if L >= 4:
            edge = list(range(2)) + list(range(L - 2, L))
            mid = list(range(2, L - 2))
            row["h6"] = {
                "l1_edge_layers": float(np.abs(le[edge] - lh[edge]).sum()),
                "l1_mid_layers": float(np.abs(le[mid] - lh[mid]).sum()) if mid else None,
                "edge_gt_mid": (
                    float(np.abs(le[edge] - lh[edge]).sum())
                    > float(np.abs(le[mid] - lh[mid]).sum())
                    if mid
                    else None
                ),
            }

    payload = {
        "direction": args.direction,
        "k": args.k,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "cross_task_floors": floors,
        "note": (
            "Important-head profiles under AP, not full circuits. "
            "H5: shared if en–hi near within-lang ceiling and above cross-task floor."
        ),
    }
    (out_dir / "language_compare.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Language comparison (AP head profiles)",
        "",
        "| Model | Task | ρ(en,hi) | top-k overlap | ceiling_en | floor | near_ceil | above_floor |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in pairs:
        h5 = row.get("h5") or {}
        lines.append(
            f"| `{row['model']}` | {row['task']} | {row['en_vs_hi']['spearman']:.3f} | "
            f"{row[f'topk_overlap_{args.k}']:.3f} | "
            f"{h5.get('ceiling_en') if h5.get('ceiling_en') is not None else '—'} | "
            f"{h5.get('floor_cross_task_mean') if h5.get('floor_cross_task_mean') is not None else '—'} | "
            f"{h5.get('near_ceiling')} | {h5.get('above_floor')} |"
        )
    if not pairs:
        lines.append("")
        lines.append("_No paired en/hi AP results yet. Run stage 1 on Ada first._")
    (out_dir / "language_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
