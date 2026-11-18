#!/usr/bin/env python3
"""Sanity-check our AP/AttrP against each other (and optionally hannamw/EAP-IG).

Reproduces a tiny GPT-2 IOI run and reports Spearman(AP, AttrP).
If eap is installed, also reports Spearman(our AttrP, eap-lib EAP).

Usage (Ada):
  $VENV/bin/python scripts/validate_engine.py --device cuda --n-prompts 20 \\
      --out results/engine_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "engine_check")
    p.add_argument("--try-eap-lib", action="store_true")
    args = p.parse_args()

    try:
        import torch
        from scipy.stats import spearmanr
        from transformer_lens import HookedTransformer
    except ImportError as e:
        args.out.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "skipped",
            "reason": f"missing deps: {e}",
            "engine_decision": "keep_ours_crosscheck_optional",
            "see": "results/ENGINE_DECISION.md",
        }
        (args.out / "validate_engine.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    from src.activation_patch import activation_patch_heads_and_mlps
    from src.attribution_patch import attribution_patch_heads_and_mlps
    from src.data import generate_batch
    from src.eap_lib_adapter import eap_available, node_attrp_from_eap_lib

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable; using CPU")

    print(f"Loading {args.model} on {device}...")
    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    batch = generate_batch(
        model, task="ioi", lang="en", n=args.n_prompts, seed=args.seed, device=device
    )

    print("AP...")
    ap = activation_patch_heads_and_mlps(model, batch, show_progress=True)
    print("AttrP...")
    attr = attribution_patch_heads_and_mlps(model, batch)

    rho, _ = spearmanr(ap["head_scores"].ravel(), attr["head_scores"].ravel())
    out = {
        "status": "ok",
        "model": args.model,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "clean_logit_diff": ap["clean_logit_diff"],
        "corrupted_logit_diff": ap["corrupted_logit_diff"],
        "spearman_ap_attr": float(rho),
        "engine_decision": "keep_ours_crosscheck_optional",
        "eap_lib_installed": eap_available(),
        "eap_lib": None,
    }
    print(f"Spearman(AP, AttrP) = {rho:.4f}")

    if args.try_eap_lib and eap_available():
        print("EAP-IG lib node EAP...")
        try:
            eap = node_attrp_from_eap_lib(model, batch, method="EAP")
            rho_e, _ = spearmanr(attr["head_scores"].ravel(), eap["head_scores"].ravel())
            out["eap_lib"] = {
                "spearman_our_attr_vs_eap_lib": float(rho_e),
                "method": eap["method"],
            }
            print(f"Spearman(our AttrP, eap-lib EAP) = {rho_e:.4f}")
        except Exception as e:
            out["eap_lib"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"EAP-lib failed: {e}")

    (args.out / "validate_engine.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out / 'validate_engine.json'}")


if __name__ == "__main__":
    main()
