#!/usr/bin/env python3
"""Full pipeline for one (model, task, lang, direction, seed) cell: AP + AttrP + EAP-IG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import cell_name
from src.runner import load_model, run_cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--task",
        default="ioi",
        choices=["ioi", "capitals", "agreement", "agreement_ne", "induction"],
    )
    p.add_argument("--lang", default="en", choices=["en", "hi", "hi_roman"])
    p.add_argument("--agreement-source", default="multiblimp", choices=["multiblimp", "hand"])
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption", default=None, help="IOI only: abc|flip")
    p.add_argument("--direction", choices=["denoising", "noising"], default="denoising")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-ig-steps", type=int, default=5)
    p.add_argument("--ig-sweep", type=str, default="", help="e.g. 5,10,20")
    p.add_argument("--skip-eap", action="store_true")
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    out = args.out or ROOT / "results" / cell_name(
        args.model, args.task, args.lang, args.seed, args.direction
    )
    model = load_model(args.model, args.device)
    sweep = [int(x) for x in args.ig_sweep.split(",") if x.strip()]
    run_cell(
        model,
        args.model,
        args.task,
        args.lang,
        args.direction,
        args.seed,
        out,
        args.device,
        n_prompts=args.n_prompts,
        micro_batch=args.micro_batch,
        n_ig_steps=args.n_ig_steps,
        ig_sweep=sweep,
        skip_eap=args.skip_eap,
        agreement_source=args.agreement_source,
        corruption=args.corruption,
        show_progress=args.progress,
    )
    print(f"Results -> {out}")


if __name__ == "__main__":
    main()
