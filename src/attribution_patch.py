"""Arm 1: attribution patching (Syed et al. / Nanda) for head and MLP outputs."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .metrics import logit_diff_from_logits


def attribution_patch_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    direction: str = "denoising",
) -> dict[str, Any]:
    """
    Denoising: AttrP(c) = (a_clean - a_corrupted) · ∇_{a_corrupted} metric
    Noising:   AttrP(c) = (a_corrupted - a_clean) · ∇_{a_clean} metric
    Both normalised by (metric_clean - metric_corrupted).
    """
    assert direction in ("denoising", "noising")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()

    with torch.no_grad():
        other_tokens = batch.clean_tokens if direction == "denoising" else batch.corrupted_tokens
        other_logits, other_cache = model.run_with_cache(other_tokens)
        other_ld = logit_diff_from_logits(
            other_logits, batch.pos_token_ids, batch.neg_token_ids
        ).mean()

    # Gradients at the base run (corrupted for denoising, clean for noising)
    base_tokens = batch.corrupted_tokens if direction == "denoising" else batch.clean_tokens
    retained: dict[str, torch.Tensor] = {}

    def retain_hook(name: str):
        def hook(act: torch.Tensor, hook):
            act.retain_grad()
            retained[name] = act
            return act

        return hook

    fwd_hooks = []
    for layer in range(n_layers):
        fwd_hooks.append((f"blocks.{layer}.attn.hook_z", retain_hook(f"z.{layer}")))
        fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", retain_hook(f"mlp.{layer}")))

    base_logits = model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)
    base_ld = logit_diff_from_logits(
        base_logits, batch.pos_token_ids, batch.neg_token_ids
    ).mean()

    if direction == "denoising":
        clean_ld, corr_ld = other_ld, base_ld
    else:
        clean_ld, corr_ld = base_ld, other_ld
    denom = (clean_ld - corr_ld).item()
    if abs(denom) < 1e-8:
        raise RuntimeError("Degenerate clean/corrupted gap for attribution patching.")

    base_ld.backward()

    head_attr = np.zeros((n_layers, n_heads), dtype=np.float64)
    mlp_attr = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        z_base = retained[f"z.{layer}"]
        z_other = other_cache["z", layer]
        grad = z_base.grad
        if direction == "denoising":
            # other=clean, base=corr → delta = clean - corr
            delta = (z_other - z_base.detach()).to(grad.dtype)
        else:
            # other=corr, base=clean → delta = corr - clean
            delta = (z_other - z_base.detach()).to(grad.dtype)
        per_head = (delta * grad).sum(dim=(0, 1, 3))
        head_attr[layer, :] = (per_head / denom).detach().float().cpu().numpy()

        mlp_base = retained[f"mlp.{layer}"]
        mlp_other = other_cache["mlp_out", layer]
        g = mlp_base.grad
        delta_m = (mlp_other - mlp_base.detach()).to(g.dtype)
        mlp_attr[layer] = float((delta_m * g).sum().item() / denom)

    elapsed = time.perf_counter() - t0

    model.zero_grad(set_to_none=True)
    del other_cache, retained
    if batch.clean_tokens.is_cuda:
        torch.cuda.empty_cache()
    if was_training:
        model.train()

    return {
        "method": "attribution_patching",
        "direction": direction,
        "head_scores": head_attr,
        "mlp_scores": mlp_attr,
        "clean_logit_diff": float(clean_ld.item()),
        "corrupted_logit_diff": float(corr_ld.item()),
        "denom": denom,
        "wall_seconds": elapsed,
        "n_prompts": int(batch.clean_tokens.shape[0]),
    }
