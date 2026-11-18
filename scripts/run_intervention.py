#!/usr/bin/env python3
"""Arms 1–2 + comparison. Loads AP from --ap-dir if present, else recomputes."""

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
from src.attribution_patch import attribution_patch_heads_and_mlps
from src.compare import compare_all, save_results
from src.data import generate_batch
from src.eap_ig import eap_ig_heads_and_mlps


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
    p.add_argument("--ap-dir", type=Path, default=None, help="Directory with ap_scores.npz")
    p.add_argument("--n-ig-steps", type=int, default=5)
    p.add_argument("--skip-eap", action="store_true")
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

    if args.ap_dir and (args.ap_dir / "ap_scores.npz").exists():
        print(f"Loading AP from {args.ap_dir}")
        z = np.load(args.ap_dir / "ap_scores.npz")
        meta = json.loads((args.ap_dir / "ap_meta.json").read_text())
        ap = {
            "method": "activation_patching",
            "head_scores": z["heads"],
            "mlp_scores": z["mlps"],
            **meta,
        }
    else:
        print("Running activation patching...")
        ap = activation_patch_heads_and_mlps(model, batch, direction=args.direction)

    print("Running attribution patching...")
    attr = attribution_patch_heads_and_mlps(model, batch, direction=args.direction)

    eap = None
    if not args.skip_eap:
        print(f"Running EAP-IG (n_steps={args.n_ig_steps})...")
        eap = eap_ig_heads_and_mlps(
            model, batch, n_steps=args.n_ig_steps, direction=args.direction
        )

    comparison = compare_all(ap, attr, eap, task=args.task)
    save_results(args.out, ap, attr, comparison, eap)
    print(json.dumps(comparison["verdict"], indent=2))
    print(f"Saved intervention results to {args.out}")


if __name__ == "__main__":
    main()
