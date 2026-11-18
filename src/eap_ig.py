"""Arm 2: attribution patching with integrated gradients (EAP-IG style)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .metrics import logit_diff_from_logits


def eap_ig_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    n_steps: int = 5,
    direction: str = "denoising",
) -> dict[str, Any]:
    """
    Path-integrated attribution.

    Denoising: interpolate embeddings corrupted → clean; Attr = (a_clean - a_corr) · mean ∇
    Noising:   interpolate embeddings clean → corrupted; Attr = (a_corr - a_clean) · mean ∇
    Both normalised by (metric_clean - metric_corrupted).
    """
    assert direction in ("denoising", "noising")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    t0 = time.perf_counter()
    model.eval()

    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(batch.clean_tokens)
        corr_logits, corr_cache = model.run_with_cache(batch.corrupted_tokens)
        clean_ld = logit_diff_from_logits(
            clean_logits, batch.pos_token_ids, batch.neg_token_ids
        ).mean()
        corr_ld = logit_diff_from_logits(
            corr_logits, batch.pos_token_ids, batch.neg_token_ids
        ).mean()
        denom = (clean_ld - corr_ld).item()
        if abs(denom) < 1e-8:
            raise RuntimeError("Degenerate clean/corrupted gap for EAP-IG.")

    grad_z_sum = {l: torch.zeros_like(clean_cache["z", l]) for l in range(n_layers)}
    grad_mlp_sum = {l: torch.zeros_like(clean_cache["mlp_out", l]) for l in range(n_layers)}
    clean_emb = clean_cache["hook_embed"]
    corr_emb = corr_cache["hook_embed"]

    # Path start → end
    if direction == "denoising":
        start_emb, end_emb = corr_emb, clean_emb
        base_tokens = batch.corrupted_tokens
    else:
        start_emb, end_emb = clean_emb, corr_emb
        base_tokens = batch.clean_tokens

    alphas = [(k + 0.5) / n_steps for k in range(n_steps)]
    for alpha in alphas:
        retained: dict[str, torch.Tensor] = {}

        def retain(name: str):
            def hook(act, hook):
                act.retain_grad()
                retained[name] = act
                return act

            return hook

        def emb_hook(act, hook, alpha=alpha):
            return start_emb + alpha * (end_emb - start_emb)

        fwd_hooks = [("hook_embed", emb_hook)]
        for layer in range(n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_z", retain(f"z.{layer}")))
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", retain(f"mlp.{layer}")))

        logits = model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)
        metric = logit_diff_from_logits(logits, batch.pos_token_ids, batch.neg_token_ids).mean()
        metric.backward()
        for layer in range(n_layers):
            grad_z_sum[layer] += retained[f"z.{layer}"].grad.detach()
            grad_mlp_sum[layer] += retained[f"mlp.{layer}"].grad.detach()
        model.zero_grad(set_to_none=True)

    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    mlp_scores = np.zeros(n_layers, dtype=np.float64)

    for layer in range(n_layers):
        mean_gz = grad_z_sum[layer] / n_steps
        if direction == "denoising":
            delta_z = clean_cache["z", layer] - corr_cache["z", layer]
        else:
            delta_z = corr_cache["z", layer] - clean_cache["z", layer]
        per_head = (delta_z * mean_gz).sum(dim=(0, 1, 3))
        head_scores[layer, :] = (per_head / denom).float().cpu().numpy()

        mean_gm = grad_mlp_sum[layer] / n_steps
        if direction == "denoising":
            delta_m = clean_cache["mlp_out", layer] - corr_cache["mlp_out", layer]
        else:
            delta_m = corr_cache["mlp_out", layer] - clean_cache["mlp_out", layer]
        mlp_scores[layer] = float((delta_m * mean_gm).sum().item() / denom)

    elapsed = time.perf_counter() - t0
    del clean_cache, corr_cache
    if batch.clean_tokens.is_cuda:
        torch.cuda.empty_cache()

    return {
        "method": "eap_ig",
        "direction": direction,
        "n_steps": n_steps,
        "head_scores": head_scores,
        "mlp_scores": mlp_scores,
        "clean_logit_diff": float(clean_ld.item()),
        "corrupted_logit_diff": float(corr_ld.item()),
        "denom": denom,
        "wall_seconds": elapsed,
        "n_prompts": int(batch.clean_tokens.shape[0]),
    }
