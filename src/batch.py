"""Generic task batch used by all patching arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TaskBatch:
    """Clean/corrupted prompt pairs with a binary logit-diff metric.

    Metric = logit(pos) - logit(neg) at the final token.
    For IOI: pos=IO, neg=S. For capitals: pos=correct capital, neg=distractor.
    """

    clean_tokens: torch.Tensor
    corrupted_tokens: torch.Tensor
    pos_token_ids: torch.Tensor
    neg_token_ids: torch.Tensor
    prompts_clean: list[str]
    prompts_corrupted: list[str]
    task: str = "ioi"
    lang: str = "en"
    corruption: str = "abc"
    meta: dict[str, Any] = field(default_factory=dict)

    # Back-compat aliases used by the original IOI-only code paths
    @property
    def io_token_ids(self) -> torch.Tensor:
        return self.pos_token_ids

    @property
    def s_token_ids(self) -> torch.Tensor:
        return self.neg_token_ids
