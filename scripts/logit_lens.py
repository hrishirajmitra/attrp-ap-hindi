#!/usr/bin/env python3
"""Layer-wise logit lens for English / Hindi / romanized-Hindi prompts.

Adapted in spirit from epfl-dlab/llm-latent-language and AI4Bharat/Romanlens:
apply the unembedding to each residual stream (post-block) and track
logit(correct) − logit(distractor) and top-1 token by layer.

Usage:
  python scripts/logit_lens.py --model gpt2 --task capitals --lang en \\
      --n-prompts 8 --device cpu --out results/logit_lens_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--task", default="capitals")
    p.add_argument("--lang", default="en", choices=["en", "hi", "hi_roman"])
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        import torch
        from transformer_lens import HookedTransformer
    except ImportError as e:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "logit_lens.json").write_text(
            json.dumps({"status": "skipped", "reason": str(e)}, indent=2) + "\n"
        )
        print(f"skipped: {e}")
        return

    from src.data import generate_batch

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = HookedTransformer.from_pretrained(args.model, device=device)
    model.eval()
    # capitals/ioi support hi_roman only for ioi; map hi_roman→hi for others
    task = args.task
    lang = args.lang
    if task != "ioi" and lang == "hi_roman":
        print("hi_roman only implemented for ioi; using hi")
        lang = "hi"

    batch = generate_batch(
        model, task=task, lang=lang, n=args.n_prompts, seed=args.seed, device=device
    )

    # Cache residual stream after each block (+ embed)
    hook_names = ["blocks.0.hook_resid_pre"] + [
        f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)
    ]
    # resid_pre of layer 0 ≈ embed+pos; then resid_post of each layer
    layer_labels = ["embed"] + [f"L{l}" for l in range(model.cfg.n_layers)]

    with torch.no_grad():
        _, cache = model.run_with_cache(batch.clean_tokens, names_filter=lambda n: n in hook_names)

    # For each layer representation, apply unembed and measure LD
    W_U = model.W_U  # [d_model, d_vocab]
    b_U = model.b_U
    ld_by_layer = []
    top1_agree = []
    for name in hook_names:
        resid = cache[name][:, -1, :]  # [B, d_model] final position
        logits = resid @ W_U + b_U  # [B, V]
        # fake [B,1,V] for gather helper
        logits_3d = logits.unsqueeze(1)
        from src.metrics import logit_diff_from_logits

        ld = logit_diff_from_logits(
            logits_3d, batch.pos_token_ids, batch.neg_token_ids
        )
        ld_by_layer.append(float(ld.mean().item()))
        pred = logits.argmax(dim=-1)
        top1_agree.append(float((pred == batch.pos_token_ids).float().mean().item()))

    payload = {
        "model": args.model,
        "task": task,
        "lang": lang,
        "n_prompts": args.n_prompts,
        "seed": args.seed,
        "layers": layer_labels,
        "mean_logit_diff": ld_by_layer,
        "top1_accuracy": top1_agree,
        "final_clean_ld": ld_by_layer[-1],
        "note": (
            "Logit lens on residual stream; not a causal intervention. "
            "Inspired by Wendler et al. / RomanLens; no external dependency."
        ),
    }
    (args.out / "logit_lens.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Tiny plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(len(layer_labels)), ld_by_layer, marker="o", label="logit-diff")
        ax.plot(range(len(layer_labels)), top1_agree, marker="s", label="top1 acc")
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Logit lens: {args.model} {task}/{lang}")
        ax.legend()
        ax.axhline(0, color="gray", lw=0.5)
        fig.tight_layout()
        fig.savefig(args.out / "logit_lens.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        payload["plot_error"] = str(e)

    print(json.dumps({k: payload[k] for k in ("model", "task", "lang", "final_clean_ld")}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
