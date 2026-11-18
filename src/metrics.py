"""Binary logit-difference metric for any TaskBatch."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch


def logit_diff_from_logits(
    logits: torch.Tensor,
    pos_token_ids: torch.Tensor,
    neg_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    logits: [B, T, V] — use final position.
    Returns per-example logit(pos) - logit(neg), shape [B].
    """
    final = logits[:, -1, :]
    pos_logits = final.gather(1, pos_token_ids[:, None]).squeeze(1)
    neg_logits = final.gather(1, neg_token_ids[:, None]).squeeze(1)
    return pos_logits - neg_logits


def mean_logit_diff(model: HookedTransformer, batch: TaskBatch, which: str = "clean") -> float:
    tokens = batch.clean_tokens if which == "clean" else batch.corrupted_tokens
    logits = model(tokens)
    ld = logit_diff_from_logits(logits, batch.pos_token_ids, batch.neg_token_ids)
    return float(ld.mean().item())


def accuracy_from_logits(
    logits: torch.Tensor,
    pos_token_ids: torch.Tensor,
    neg_token_ids: torch.Tensor,
) -> float:
    """Fraction of examples where logit(pos) > logit(neg)."""
    ld = logit_diff_from_logits(logits, pos_token_ids, neg_token_ids)
    return float((ld > 0).float().mean().item())


def metric_fn_factory(batch: TaskBatch):
    """Return a function logits -> scalar mean logit-diff (for attribution patching)."""

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return logit_diff_from_logits(logits, batch.pos_token_ids, batch.neg_token_ids).mean()

    return metric
