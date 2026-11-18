#!/usr/bin/env python3
"""Reproduce the locked v1 GPT-2 IOI study with the new engines (5 seeds, abc corruption).

Compares fresh scores against results/full_abc_seed{0..4}/scores.npz and reports the v1 headline
numbers (head rho ~0.989 across seeds, MLP r ~ -0.93).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.activation_patch import activation_patch_heads_and_mlps
from src.attribution_patch import attribution_patch_heads_and_mlps
from src.data import generate_batch
from src.modelload import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--micro-batch", type=int, default=50)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "validate_v1.json")
    args = p.parse_args()

    model = load_model("gpt2", "cuda")
    rows = []
    for s in args.seeds:
        old = np.load(ROOT / "results" / f"full_abc_seed{s}" / "scores.npz")
        b = generate_batch(model, "ioi", "en", n=args.n_prompts, seed=s, device="cuda", corruption="abc")
        ap = activation_patch_heads_and_mlps(model, b, micro_batch=args.micro_batch)
        at = attribution_patch_heads_and_mlps(model, b, micro_batch=args.micro_batch)
        rho, _ = spearmanr(ap["head_scores"].ravel(), at["head_scores"].ravel())
        mlp_r, _ = pearsonr(ap["mlp_scores"], at["mlp_scores"])
        rows.append(
            {
                "seed": s,
                "rho_new": float(rho),
                "mlp_r_new": float(mlp_r),
                "max_abs_diff_ap_heads_vs_v1": float(np.abs(ap["head_scores"] - old["ap_heads"]).max()),
                "max_abs_diff_attr_heads_vs_v1": float(np.abs(at["head_scores"] - old["attr_heads"]).max()),
                "max_abs_diff_ap_mlps_vs_v1": float(np.abs(ap["mlp_scores"] - old["ap_mlps"]).max()),
                "max_abs_diff_attr_mlps_vs_v1": float(np.abs(at["mlp_scores"] - old["attr_mlps"]).max()),
                "fast_mode": ap["fast_mode"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "mean_rho": float(np.mean([r["rho_new"] for r in rows])),
        "mean_mlp_r": float(np.mean([r["mlp_r_new"] for r in rows])),
        "worst_abs_diff": float(
            max(max(v for k, v in r.items() if k.startswith("max_abs")) for r in rows)
        ),
        "rows": rows,
    }
    # v1 locked: rho ~0.989, MLP r ~ -0.93.  Scores should match v1 to numerical precision.
    summary["pass"] = bool(
        abs(summary["mean_rho"] - 0.989) < 0.01
        and summary["mean_mlp_r"] < -0.85
        and summary["worst_abs_diff"] < 5e-3
    )
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print("VALIDATE_V1", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    print("VALIDATE_OK" if summary["pass"] else "VALIDATE_FAILED")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
