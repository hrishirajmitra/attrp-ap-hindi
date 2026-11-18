"""Arm 0: full activation patching of head outputs and MLP outputs.

Exactly equivalent to the original single-batch implementation, but
  * micro-batched (the mean logit-diff is a sum over examples, so per-chunk sums add up), and
  * optionally starts each patched forward pass at the patched layer from a cached residual
    ("fast" mode). Fast mode is self-checked against the full forward pass and disabled
    automatically if the two disagree.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from .batch import TaskBatch
from .chunking import cache_filter, iter_chunks, sync
from .metrics import logit_diff_from_logits

FAST_CHECK_TOL = 1e-3  # max abs logit-diff difference tolerated between fast and full forward


@torch.no_grad()
def activation_patch_heads_and_mlps(
    model: HookedTransformer,
    batch: TaskBatch,
    show_progress: bool = False,
    direction: str = "denoising",
    micro_batch: int | None = None,
    fast: bool = True,
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
    n = int(batch.clean_tokens.shape[0])

    sync(device)
    t0 = time.perf_counter()

    clean_sum = torch.zeros((), dtype=torch.float64, device=device)
    corr_sum = torch.zeros((), dtype=torch.float64, device=device)
    head_acc = torch.zeros((n_layers, n_heads), dtype=torch.float64, device=device)
    mlp_acc = torch.zeros(n_layers, dtype=torch.float64, device=device)

    fast_ok = bool(fast)
    fast_check_diff = None
    first_chunk = True

    for ct, rt, pos, neg in iter_chunks(batch, micro_batch):
        clean_logits, clean_cache = model.run_with_cache(ct, names_filter=cache_filter)
        corr_logits, corr_cache = model.run_with_cache(rt, names_filter=cache_filter)
        clean_sum += logit_diff_from_logits(clean_logits, pos, neg).sum().double()
        corr_sum += logit_diff_from_logits(corr_logits, pos, neg).sum().double()
        del clean_logits, corr_logits

        if direction == "denoising":
            src_cache, base_cache, base_tokens = clean_cache, corr_cache, rt
        else:
            src_cache, base_cache, base_tokens = corr_cache, clean_cache, ct

        def run_patched(layer: int, hook_name: str, hook_fn):
            if fast_ok:
                return model.run_with_hooks(
                    base_cache["resid_pre", layer],
                    fwd_hooks=[(hook_name, hook_fn)],
                    start_at_layer=layer,
                )
            return model.run_with_hooks(base_tokens, fwd_hooks=[(hook_name, hook_fn)])

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

        if first_chunk and fast_ok:
            # Self-check: fast (start_at_layer) vs full forward for a few sites.
            worst = 0.0
            probes = sorted({0, n_layers // 2, n_layers - 1})
            for layer in probes:
                for kind in ("head", "mlp"):
                    if kind == "head":
                        hn, hf = f"blocks.{layer}.attn.hook_z", make_head_hook(layer, 0)
                    else:
                        hn, hf = f"blocks.{layer}.hook_mlp_out", make_mlp_hook(layer)
                    full = model.run_with_hooks(base_tokens, fwd_hooks=[(hn, hf)])
                    fastl = model.run_with_hooks(
                        base_cache["resid_pre", layer], fwd_hooks=[(hn, hf)], start_at_layer=layer
                    )
                    d = (
                        logit_diff_from_logits(full, pos, neg)
                        - logit_diff_from_logits(fastl, pos, neg)
                    ).abs().max().item()
                    worst = max(worst, d)
            fast_check_diff = worst
            if worst > FAST_CHECK_TOL:
                print(f"  [AP] fast mode check FAILED (max diff {worst:.2e}); using full forward")
                fast_ok = False
            else:
                print(f"  [AP] fast mode check ok (max diff {worst:.2e})")
        first_chunk = False

        iterator = range(n_layers)
        if show_progress:
            iterator = tqdm(iterator, desc=f"AP heads ({direction})")
        for layer in iterator:
            for head in range(n_heads):
                logits = run_patched(
                    layer, f"blocks.{layer}.attn.hook_z", make_head_hook(layer, head)
                )
                head_acc[layer, head] += logit_diff_from_logits(logits, pos, neg).sum().double()
        for layer in range(n_layers):
            logits = run_patched(layer, f"blocks.{layer}.hook_mlp_out", make_mlp_hook(layer))
            mlp_acc[layer] += logit_diff_from_logits(logits, pos, neg).sum().double()

        del clean_cache, corr_cache, src_cache, base_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clean_ld = (clean_sum / n).item()
    corr_ld = (corr_sum / n).item()
    denom = clean_ld - corr_ld
    if abs(denom) < 1e-8:
        raise RuntimeError(
            f"Clean/corrupted logit-diff nearly equal ({clean_ld:.4f} vs {corr_ld:.4f}); "
            "task may be broken for this batch."
        )
    base_ld = corr_ld if direction == "denoising" else clean_ld

    patched_heads = (head_acc / n).cpu().numpy()
    patched_mlps = (mlp_acc / n).cpu().numpy()
    if direction == "denoising":
        head_scores = (patched_heads - base_ld) / denom
        mlp_scores = (patched_mlps - base_ld) / denom
    else:
        head_scores = (base_ld - patched_heads) / denom
        mlp_scores = (base_ld - patched_mlps) / denom

    sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "method": "activation_patching",
        "direction": direction,
        "head_scores": head_scores.astype(np.float64),
        "mlp_scores": mlp_scores.astype(np.float64),
        "clean_logit_diff": float(clean_ld),
        "corrupted_logit_diff": float(corr_ld),
        "denom": float(denom),
        "wall_seconds": elapsed,
        "n_prompts": n,
        "micro_batch": int(micro_batch) if micro_batch else n,
        "fast_mode": bool(fast_ok),
        "fast_check_max_diff": fast_check_diff,
    }
