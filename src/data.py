"""Factory: build a TaskBatch for (task, lang, corruption)."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .data_agreement import generate_agreement_batch, generate_ne_perfective_batch
from .data_capitals import generate_capitals_batch
from .data_induction import generate_induction_batch
from .data_ioi import generate_ioi_batch
from .data_multiblimp import generate_multiblimp_batch


def generate_batch(
    model: HookedTransformer,
    task: str,
    lang: str = "en",
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    corruption: str | None = None,
    allow_multi_token_answers: bool = True,
    agreement_source: str = "multiblimp",
) -> TaskBatch:
    """
    agreement_source:
      - "multiblimp" (default): MultiBLiMP hin/eng pairs
      - "hand": small hand-written templates (smoke / fallback)
    """
    task = task.lower()
    lang = lang.lower()
    if task == "ioi":
        corr = corruption or "abc"
        return generate_ioi_batch(
            model,
            n=n,
            seed=seed,
            device=device,
            corruption=corr,
            lang=lang,
            allow_multi_token_answers=allow_multi_token_answers,
        )
    if task == "capitals":
        return generate_capitals_batch(
            model,
            n=n,
            seed=seed,
            device=device,
            lang=lang,
            allow_multi_token_answers=allow_multi_token_answers,
        )
    if task == "agreement":
        if agreement_source == "hand":
            return generate_agreement_batch(
                model,
                n=n,
                seed=seed,
                device=device,
                lang=lang if lang in ("en", "hi") else "hi",
                allow_multi_token_answers=allow_multi_token_answers,
            )
        # Default: MultiBLiMP (en / hi only)
        blang = "en" if lang.startswith("en") else "hi"
        return generate_multiblimp_batch(
            model,
            n=n,
            seed=seed,
            device=device,
            lang=blang,
            allow_multi_token_answers=allow_multi_token_answers,
        )
    if task in ("agreement_ne", "ne_perfective"):
        return generate_ne_perfective_batch(
            model,
            n=n,
            seed=seed,
            device=device,
            allow_multi_token_answers=allow_multi_token_answers,
        )
    if task == "induction":
        return generate_induction_batch(
            model,
            n=n,
            seed=seed,
            device=device,
            lang=lang if lang in ("en", "hi") else "en",
            allow_multi_token_answers=allow_multi_token_answers,
        )
    raise ValueError(f"Unknown task: {task}")
