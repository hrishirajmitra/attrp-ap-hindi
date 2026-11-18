"""In-context copy / induction control (language-neutral structure).

Clean:   "A B C A B"  → predict C  (repeated bigram induction)
Corr:    "A B C X Y"  → same answer tokens C vs distractor, but pattern broken.

We use short letter/digit or Hindi akshara tokens depending on language, filtered
to single-token items per model.
"""

from __future__ import annotations

import random
import string

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .tokenize import try_single_or_first

LETTERS_EN = list(string.ascii_uppercase)
# Devanagari consonants that are often single tokens in multilingual tokenizers
AKSHARA_HI = list("कखगघचछजझटठडढतथदधनपफबभमयरलवशषसह")


def generate_induction_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    lang: str = "en",
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    assert lang in ("en", "hi")
    rng = random.Random(seed)
    pool = LETTERS_EN if lang == "en" else AKSHARA_HI
    lead = False  # bare tokens, no leading space for both (letters as tokens)
    device = device or model.cfg.device

    usable = []
    for ch in pool:
        try:
            tid, ntok = try_single_or_first(
                model, ch, leading_space=lead, allow_multi=allow_multi_token_answers
            )
            if ntok == 1:
                usable.append((ch, tid))
        except ValueError:
            continue
    # Also try with leading space for English BPE
    if len(usable) < 8 and lang == "en":
        for ch in pool:
            try:
                tid, ntok = try_single_or_first(model, ch, leading_space=True, allow_multi=False)
                usable.append((ch, tid))
                lead = True
            except ValueError:
                continue
    # Deduplicate by token id
    seen = set()
    uniq = []
    for ch, tid in usable:
        if tid not in seen:
            seen.add(tid)
            uniq.append((ch, tid))
    usable = uniq
    if len(usable) < 6:
        raise RuntimeError(f"Too few single-token induction symbols for lang={lang}")

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError(f"Could not generate enough induction prompts (lang={lang})")
        (a, _), (b, _), (c, c_id), (x, _), (y, _), (d, d_id) = rng.sample(usable, 6)
        if c_id == d_id:
            continue
        # Pattern: A B C A B → C
        sep = " "
        clean = sep.join([a, b, c, a, b])
        corr = sep.join([a, b, c, x, y])  # break the repeat
        ct = model.to_tokens(clean)[0]
        rt = model.to_tokens(corr)[0]
        if ct.numel() != rt.numel():
            continue
        if target_len is None:
            target_len = ct.numel()
        if ct.numel() != target_len:
            continue
        clean_s.append(clean)
        corr_s.append(corr)
        clean_t.append(ct)
        corr_t.append(rt)
        pos_ids.append(c_id)
        neg_ids.append(d_id)

    return TaskBatch(
        clean_tokens=torch.stack(clean_t).to(device),
        corrupted_tokens=torch.stack(corr_t).to(device),
        pos_token_ids=torch.tensor(pos_ids, device=device),
        neg_token_ids=torch.tensor(neg_ids, device=device),
        prompts_clean=clean_s,
        prompts_corrupted=corr_s,
        task="induction",
        lang=lang,
        corruption="break_repeat",
        meta={
            "single_token_answer_rate": 1.0,
            "n_symbols": len(usable),
            "metric": "single_token_logit_diff",
            "leading_space": lead,
        },
    )
