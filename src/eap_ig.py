"""Arm 2: attribution patching with integrated gradients (EAP-IG style).

Micro-batched and gradient-accumulating: the mean-over-steps gradient is linear, so each
(chunk, step) adds delta . grad / n_steps directly to the per-component accumulators.
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


def eap_ig_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    n_steps: int = 5,
    direction: str = "denoising",
    micro_batch: int | None = None,
) -> dict[str, Any]:
    """
    Path-integrated attribution.

    Denoising: interpolate embeddings corrupted -> clean; Attr = (a_clean - a_corr) . mean grad
    Noising:   interpolate embeddings clean -> corrupted; Attr = (a_corr - a_clean) . mean grad
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
    clean_sum = torch.zeros((), dtype=torch.float64, device=device)
    corr_sum = torch.zeros((), dtype=torch.float64, device=device)

    alphas = [(k + 0.5) / n_steps for k in range(n_steps)]

    for ct, rt, pos, neg in iter_chunks(batch, micro_batch):
        with torch.no_grad():
            clean_logits, clean_cache = model.run_with_cache(ct, names_filter=cache_filter)
            corr_logits, corr_cache = model.run_with_cache(rt, names_filter=cache_filter)
            clean_sum += logit_diff_from_logits(clean_logits, pos, neg).sum().double()
            corr_sum += logit_diff_from_logits(corr_logits, pos, neg).sum().double()
            del clean_logits, corr_logits
            clean_emb = clean_cache["hook_embed"]
            corr_emb = corr_cache["hook_embed"]

        if direction == "denoising":
            start_emb, end_emb, base_tokens = corr_emb, clean_emb, rt
            src, base = clean_cache, corr_cache
        else:
            start_emb, end_emb, base_tokens = clean_emb, corr_emb, ct
            src, base = corr_cache, clean_cache

        for alpha in alphas:
            retained: dict[str, torch.Tensor] = {}

            def retain(name: str):
                def hook(act, hook):
                    act.retain_grad()
                    retained[name] = act
                    return act

                return hook

            def emb_hook(act, hook, alpha=alpha):
                return (start_emb + alpha * (end_emb - start_emb)).detach().requires_grad_(True)

            fwd_hooks = [("hook_embed", emb_hook)]
            for layer in range(n_layers):
                fwd_hooks.append((f"blocks.{layer}.attn.hook_z", retain(f"z.{layer}")))
                fwd_hooks.append((f"blocks.{layer}.hook_mlp_out", retain(f"mlp.{layer}")))

            logits = model.run_with_hooks(base_tokens, fwd_hooks=fwd_hooks)
            ld = logit_diff_from_logits(logits, pos, neg)
            (ld.sum() / n).backward()

            with torch.no_grad():
                for layer in range(n_layers):
                    # delta = (clean - corr) for denoising, (corr - clean) for noising
                    delta_z = src["z", layer] - base["z", layer]
                    gz = retained[f"z.{layer}"].grad
                    head_acc[layer] += (delta_z * gz).sum(dim=(0, 1, 3)).double() / n_steps
                    delta_m = src["mlp_out", layer] - base["mlp_out", layer]
                    gm = retained[f"mlp.{layer}"].grad
                    mlp_acc[layer] += (delta_m * gm).sum().double() / n_steps
            del retained, logits, ld

        del clean_cache, corr_cache, src, base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clean_ld = (clean_sum / n).item()
    corr_ld = (corr_sum / n).item()
    denom = clean_ld - corr_ld
    if abs(denom) < 1e-8:
        raise RuntimeError("Degenerate clean/corrupted gap for EAP-IG.")

    # noising: delta.grad estimates (patched - clean) = -(effect), so flip to the AP convention
    sgn = 1.0 if direction == "denoising" else -1.0
    head_scores = (sgn * head_acc / denom).cpu().numpy()
    mlp_scores = (sgn * mlp_acc / denom).cpu().numpy()

    sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "method": "eap_ig",
        "direction": direction,
        "n_steps": n_steps,
        "head_scores": head_scores.astype(np.float64),
        "mlp_scores": mlp_scores.astype(np.float64),
        "clean_logit_diff": float(clean_ld),
        "corrupted_logit_diff": float(corr_ld),
        "denom": float(denom),
        "wall_seconds": elapsed,
        "n_prompts": n,
        "micro_batch": int(micro_batch) if micro_batch else n,
    }
