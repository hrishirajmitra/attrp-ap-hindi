"""Micro-batching helpers shared by the AP / AttrP / EAP-IG engines.

All three engines score components with the *mean* logit difference over the batch. That metric
is linear in the examples, so splitting the batch into chunks and summing per-chunk
contributions gives exactly the same scores as one big batch, while fitting in 11 GB.
"""

from __future__ import annotations

from typing import Iterator

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch

CACHE_NAMES = ("hook_z", "hook_mlp_out", "hook_resid_pre", "hook_embed")


def cache_filter(name: str) -> bool:
    return name.endswith(CACHE_NAMES)


def iter_chunks(
    batch: TaskBatch, micro_batch: int | None
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    n = int(batch.clean_tokens.shape[0])
    mb = n if not micro_batch or micro_batch <= 0 else min(int(micro_batch), n)
    for s in range(0, n, mb):
        e = min(n, s + mb)
        yield (
            batch.clean_tokens[s:e],
            batch.corrupted_tokens[s:e],
            batch.pos_token_ids[s:e],
            batch.neg_token_ids[s:e],
        )


def freeze_params(model: HookedTransformer) -> None:
    """No parameter gradients are needed (only activation gradients): saves ~2x memory/time."""
    for p in model.parameters():
        p.requires_grad_(False)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
