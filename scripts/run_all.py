#!/usr/bin/env python3
"""Full pipeline: AP + AttrP + EAP-IG (+ optional step sweep) for one (model, task, lang) cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.stats import pearsonr, spearmanr
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.activation_patch import activation_patch_heads_and_mlps
from src.attribution_patch import attribution_patch_heads_and_mlps
from src.compare import compare_all, save_results
from src.data import generate_batch
from src.eap_ig import eap_ig_heads_and_mlps
from src.registry import cell_name


def load_model(name: str, device: str) -> HookedTransformer:
    print(f"Loading {name} on {device}...")
    try:
        model = HookedTransformer.from_pretrained(name, device=device)
    except Exception as e:
        # Some hub ids need dtype=float16 on 11GB cards
        print(f"  retry with dtype=float16 after: {e}")
        model = HookedTransformer.from_pretrained(name, device=device, dtype=torch.float16)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument(
        "--task",
        default="ioi",
        choices=["ioi", "capitals", "agreement", "agreement_ne", "induction"],
    )
    p.add_argument("--lang", default="en", choices=["en", "hi", "hi_roman"])
    p.add_argument(
        "--agreement-source",
        default="multiblimp",
        choices=["multiblimp", "hand"],
        help="For task=agreement: MultiBLiMP (default) or hand templates",
    )
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption", default=None, help="IOI only: abc|flip")
    p.add_argument("--direction", choices=["denoising", "noising"], default="denoising")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-ig-steps", type=int, default=5)
    p.add_argument(
        "--ig-sweep",
        type=str,
        default="",
        help="Comma-separated EAP-IG step counts to sweep, e.g. 5,10,20",
    )
    p.add_argument("--skip-eap", action="store_true")
    args = p.parse_args()

    if args.out is None:
        args.out = ROOT / "results" / cell_name(
            args.model, args.task, args.lang, args.seed, args.direction
        )
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.device)
    batch = generate_batch(
        model,
        task=args.task,
        lang=args.lang,
        n=args.n_prompts,
        seed=args.seed,
        device=args.device,
        corruption=args.corruption,
        agreement_source=args.agreement_source,
    )
    print(
        f"task={args.task} lang={args.lang} dir={args.direction} "
        f"n={args.n_prompts} seq_len={batch.clean_tokens.shape[1]} "
        f"single_tok_rate={batch.meta.get('single_token_answer_rate')}"
    )

    print("=== Arm 0: activation patching ===")
    ap = activation_patch_heads_and_mlps(model, batch, direction=args.direction)
    print(
        f"  clean_ld={ap['clean_logit_diff']:.4f} "
        f"corr_ld={ap['corrupted_logit_diff']:.4f} "
        f"wall={ap['wall_seconds']:.1f}s"
    )

    print("=== Arm 1: attribution patching ===")
    attr = attribution_patch_heads_and_mlps(model, batch, direction=args.direction)
    print(f"  wall={attr['wall_seconds']:.1f}s")

    eap = None
    eap_sweep: dict[int, dict] = {}
    sweep_steps = []
    if args.ig_sweep.strip():
        sweep_steps = sorted({int(x) for x in args.ig_sweep.split(",") if x.strip()})
    elif not args.skip_eap:
        sweep_steps = [args.n_ig_steps]

    for steps in sweep_steps:
        print(f"=== Arm 2: EAP-IG (steps={steps}) ===")
        res = eap_ig_heads_and_mlps(model, batch, n_steps=steps, direction=args.direction)
        print(f"  wall={res['wall_seconds']:.1f}s")
        eap_sweep[steps] = res
        if steps == args.n_ig_steps or eap is None:
            eap = res

    if args.skip_eap:
        eap = None
        eap_sweep = {}

    comparison = compare_all(ap, attr, eap, task=args.task)
    comparison["cell"] = {
        "model": args.model,
        "task": args.task,
        "lang": args.lang,
        "seed": args.seed,
        "direction": args.direction,
        "corruption": batch.corruption,
        "batch_meta": batch.meta,
    }
    if eap_sweep and len(eap_sweep) > 1:
        comparison["eap_sweep"] = {}
        for s, res in eap_sweep.items():
            rho_s, _ = spearmanr(ap["head_scores"].ravel(), res["head_scores"].ravel())
            r_m, _ = pearsonr(ap["mlp_scores"], res["mlp_scores"])
            comparison["eap_sweep"][str(s)] = {
                "spearman_ap_eap_ig": float(rho_s),
                "mlp_pearson_r": float(r_m),
                "wall_seconds": res["wall_seconds"],
            }

    save_results(args.out, ap, attr, comparison, eap, eap_sweep=eap_sweep or None)

    print("=== Verdict ===")
    print(json.dumps(comparison["verdict"], indent=2))
    print(f"Spearman(AP, AttrP) = {comparison['spearman_ap_attr']:.4f}")
    print(f"Top-14 overlap AttrP = {comparison['topk_overlap_ap_attr']['14']:.4f}")
    if eap is not None:
        print(f"Spearman(AP, EAP-IG) = {comparison['spearman_ap_eap_ig']:.4f}")
    print(f"Results → {args.out}")


if __name__ == "__main__":
    main()
