#!/usr/bin/env python3
"""Arm 0 only: full activation patching baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.activation_patch import activation_patch_heads_and_mlps
from src.compare import plot_heatmaps
from src.data import generate_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--task", default="ioi", choices=["ioi", "capitals", "agreement", "induction"])
    p.add_argument("--lang", default="en", choices=["en", "hi"])
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption", default=None)
    p.add_argument("--direction", choices=["denoising", "noising"], default="denoising")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()

    batch = generate_batch(
        model,
        task=args.task,
        lang=args.lang,
        n=args.n_prompts,
        seed=args.seed,
        device=device,
        corruption=args.corruption,
    )
    print(f"Batch: {batch.clean_tokens.shape}, AP direction={args.direction}")
    ap = activation_patch_heads_and_mlps(model, batch, direction=args.direction)

    np.savez(args.out / "ap_scores.npz", heads=ap["head_scores"], mlps=ap["mlp_scores"])
    meta = {k: v for k, v in ap.items() if k not in ("head_scores", "mlp_scores")}
    (args.out / "ap_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    plot_heatmaps(ap["head_scores"], ap["head_scores"], None, args.out / "ap_heatmap.png")

    print(json.dumps(meta, indent=2))
    print(f"Saved baseline to {args.out}")


if __name__ == "__main__":
    main()
