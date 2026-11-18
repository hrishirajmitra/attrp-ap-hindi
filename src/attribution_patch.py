"""Arm 1: attribution patching (Syed et al. / Nanda) for head and MLP outputs.

Micro-batched: the metric is the batch-mean logit difference, i.e. a sum over examples divided
by N. Each chunk back-propagates (sum over chunk) / N, so the accumulated delta . grad equals
the single-batch value exactly.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .chunking import cache_filter, freeze_params, iter_chunks, sync
from .metrics import logit_diff_from_logits


def attribution_patch_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    direction: str = "denoising",
    micro_batch: int | None = None,
) -> dict[str, Any]:
    """
    Denoising: AttrP(c) = (a_clean - a_corrupted) . grad_{a_corrupted} metric
    Noising:   AttrP(c) = (a_corrupted - a_clean) . grad_{a_clean} metric
    Both normalised by (metric_clean - metric_corrupted).
    """
    assert direction in ("denoising", "noising")
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = batch.clean_tokens.device
    n = int(batch.clean_tokens.shape[0])

    freeze_params(model)
    model.eval()
    sync(device)
    t0 = time.perf_counter()

    head_acc = torch.zeros((n_layers, n_heads), dtype=torch.float64, device=device)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64, device=device)
    other_sum = torch.zeros((), dtype=torch.float64, device=device)
    base_sum = torch.zeros((), dtype=torch.float64, device=device)

    for ct, rt, pos, neg in iter_chunks(batch, micro_batch):
        other_tokens = ct if direction == "denoising" else rt
        base_tokens = rt if direction == "denoising" else ct

        with torch.no_grad():
            other_logits, other_cache = model.run_with_cache(
                other_tokens, names_filter=cache_filter
            )
            other_sum += logit_diff_from_logits(other_logits, pos, neg).sum().double()
            del other_logits

        retained: dict[str, torch.Tensor] = {}

        def embed_hook(act: torch.Tensor, hook):
            # Params are frozen, so make the embedding a grad-requiring leaf to get
            # activation gradients downstream.
            return act.detach().requires_grad_(True)

        def retain_hook(name: str):
            def hook(act: torch.Tensor, hook):
                act.retain_grad()
                retained[name] = act
                return act

            return hook

        fwd_hooks = [("hook_embed", embed_hook)]
        for layer in range(n_layers):
            fwd_hooks.append((f"blocks.{layer}.attn.hook_z", retain_hook(f"z.{layer}")))
            fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", retain_hook(f"mlp.{layer}")))

        base_logits = model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)
        base_ld = logit_diff_from_logits(base_logits, pos, neg)
        base_sum += base_ld.detach().sum().double()
        (base_ld.sum() / n).backward()

        with torch.no_grad():
            for layer in range(n_layers):
                z_base = retained[f"z.{layer}"]
                delta = other_cache["z", layer] - z_base.detach()
                head_acc[layer] += (delta * z_base.grad).sum(dim=(0, 1, 3)).double()
                m_base = retained[f"mlp.{layer}"]
                delta_m = other_cache["mlp_out", layer] - m_base.detach()
                mlp_acc[layer] += (delta_m * m_base.grad).sum().double()

        del other_cache, retained, base_logits, base_ld
        if device.type == "cuda":
            torch.cuda.empty_cache()

    other_ld = (other_sum / n).item()
    base_ld_mean = (base_sum / n).item()
    if direction == "denoising":
        clean_ld, corr_ld = other_ld, base_ld_mean
    else:
        clean_ld, corr_ld = base_ld_mean, other_ld
    denom = clean_ld - corr_ld
    if abs(denom) < 1e-8:
        raise RuntimeError("Degenerate clean/corrupted gap for attribution patching.")

    head_attr = (head_acc / denom).cpu().numpy()
    mlp_attr = (mlp_acc / denom).cpu().numpy()

    sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "method": "attribution_patching",
        "direction": direction,
        "head_scores": head_attr.astype(np.float64),
        "mlp_scores": mlp_attr.astype(np.float64),
        "clean_logit_diff": float(clean_ld),
        "corrupted_logit_diff": float(corr_ld),
        "denom": float(denom),
        "wall_seconds": elapsed,
        "n_prompts": n,
        "micro_batch": int(micro_batch) if micro_batch else n,
    }
