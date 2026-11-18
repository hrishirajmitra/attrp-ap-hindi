"""Arm 0: full activation patching of head outputs and MLP outputs."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .batch import TaskBatch
from .metrics import logit_diff_from_logits


@torch.no_grad()
def activation_patch_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    show_progress: bool = True,
    direction: str = "denoising",
) -> dict[str, Any]:
    """
    Denoising: patch *clean* activations into a *corrupted* forward pass.
      score = (metric(patched) - metric(corrupted)) / (metric(clean) - metric(corrupted))
    Noising: patch *corrupted* activations into a *clean* forward pass.
      score = (metric(clean) - metric(patched)) / (metric(clean) - metric(corrupted))
    """
    assert direction in ("denoising", "noising")
    device = batch.clean_tokens.device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    t0 = time.perf_counter()

    clean_logits, clean_cache = model.run_with_cache(batch.clean_tokens)
    corr_logits, corr_cache = model.run_with_cache(batch.corrupted_tokens)

    clean_ld = logit_diff_from_logits(clean_logits, batch.pos_token_ids, batch.neg_token_ids).mean()
    corr_ld = logit_diff_from_logits(corr_logits, batch.pos_token_ids, batch.neg_token_ids).mean()
    denom = (clean_ld - corr_ld).item()
    if abs(denom) < 1e-8:
        raise RuntimeError(
            f"Clean/corrupted logit-diff nearly equal ({clean_ld:.4f} vs {corr_ld:.4f}); "
            "task may be broken for this batch."
        )

    src_cache = clean_cache if direction == "denoising" else corr_cache
    base_tokens = batch.corrupted_tokens if direction == "denoising" else batch.clean_tokens
    base_ld = corr_ld if direction == "denoising" else clean_ld

    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    mlp_scores = np.zeros(n_layers, dtype=np.float64)

    def make_head_hook(layer: int, head: int):
        src_z = src_cache["z", layer]

        def hook(z: torch.Tensor, hook: HookPoint):
            z = z.clone()
            z[:, :, head, :] = src_z[:, :, head, :]
            return z

        return hook

    def make_mlp_hook(layer: int):
        src_mlp = src_cache["mlp_out", layer]

        def hook(mlp_out: torch.Tensor, hook: HookPoint):
            return src_mlp

        return hook

    iterator = range(n_layers)
    if show_progress:
        iterator = tqdm(iterator, desc=f"AP heads ({direction})")
    for layer in iterator:
        for head in range(n_heads):
            logits = model.run_with_hooks(
                base_tokens,
                fwd_hooks=[(f"blocks.{layer}.attn.hook_z", make_head_hook(layer, head))],
            )
            ld = logit_diff_from_logits(logits, batch.pos_token_ids, batch.neg_token_ids).mean()
            if direction == "denoising":
                head_scores[layer, head] = (ld.item() - base_ld.item()) / denom
            else:
                head_scores[layer, head] = (base_ld.item() - ld.item()) / denom

    for layer in tqdm(range(n_layers), desc=f"AP MLPs ({direction})", disable=not show_progress):
        logits = model.run_with_hooks(
            base_tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_mlp_out", make_mlp_hook(layer))],
        )
        ld = logit_diff_from_logits(logits, batch.pos_token_ids, batch.neg_token_ids).mean()
        if direction == "denoising":
            mlp_scores[layer] = (ld.item() - base_ld.item()) / denom
        else:
            mlp_scores[layer] = (base_ld.item() - ld.item()) / denom

    elapsed = time.perf_counter() - t0

    del clean_cache, corr_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "method": "activation_patching",
        "direction": direction,
        "head_scores": head_scores,
        "mlp_scores": mlp_scores,
        "clean_logit_diff": float(clean_ld.item()),
        "corrupted_logit_diff": float(corr_ld.item()),
        "denom": denom,
        "wall_seconds": elapsed,
        "n_prompts": int(batch.clean_tokens.shape[0]),
    }
