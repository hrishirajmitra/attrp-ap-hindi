"""Factual recall: country → capital (English and Hindi).

Metric: logit(correct_capital) − logit(distractor_capital) at the final token.
Corruption: swap the country name for a different country (answer tokens stay
the original capital vs distractor, so corrupted logit-diff collapses).
"""

from __future__ import annotations

import random

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .tokenize import try_single_or_first

# (country, capital) pairs — capitals chosen to be often single-token in English BPE.
PAIRS_EN = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("Russia", "Moscow"),
    ("Egypt", "Cairo"),
    ("Canada", "Ottawa"),
    ("Australia", "Sydney"),  # commonly predicted though Canberra is correct; keep Ottawa-style
    ("India", "Delhi"),
    ("Brazil", "Brasilia"),
    ("Mexico", "Mexico"),  # "Mexico City" multi-token — use short distractor set
    ("Greece", "Athens"),
    ("Turkey", "Ankara"),
    ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Portugal", "Lisbon"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    ("Chile", "Santiago"),
    ("Peru", "Lima"),
    ("Cuba", "Havana"),
    ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"),
    ("Afghanistan", "Kabul"),
    ("Pakistan", "Islamabad"),
]

# Fix Australia → Canberra (more correct)
PAIRS_EN = [(c, cap) if c != "Australia" else ("Australia", "Canberra") for c, cap in PAIRS_EN]
# Fix Mexico → use a short name that models often treat as capital cue; skip if multi-token later
PAIRS_EN = [(c, cap) if c != "Mexico" else ("Mexico", "Mexico") for c, cap in PAIRS_EN]

PAIRS_HI = [
    ("फ़्रांस", "पेरिस"),
    ("जर्मनी", "बर्लिन"),
    ("इटली", "रोम"),
    ("स्पेन", "मैड्रिड"),
    ("जापान", "टोक्यो"),
    ("चीन", "बीजिंग"),
    ("रूस", "मास्को"),
    ("मिस्र", "काहिरा"),
    ("कनाडा", "ओटावा"),
    ("ऑस्ट्रेलिया", "कैनबरा"),
    ("भारत", "दिल्ली"),
    ("ब्राज़ील", "ब्रासीलिया"),
    ("ग्रीस", "एथेंस"),
    ("तुर्की", "अंकारा"),
    ("पोलैंड", "वारसॉ"),
    ("स्वीडन", "स्टॉकहोम"),
    ("नॉर्वे", "ओस्लो"),
    ("पुर्तगाल", "लिस्बन"),
    ("थाईलैंड", "बैंकॉक"),
    ("वियतनाम", "हनोई"),
    ("ईरान", "तेहरान"),
    ("इराक", "बगदाद"),
    ("चिली", "सैंटियागो"),
    ("पेरू", "लीमा"),
    ("क्यूबा", "हवाना"),
    ("केन्या", "नैरोबी"),
    ("नाइजीरिया", "अबुजा"),
    ("अफ़ग़ानिस्तान", "काबुल"),
    ("पाकिस्तान", "इस्लामाबाद"),
    ("नेपाल", "काठमांडू"),
]

TEMPLATES_EN = [
    "The capital of {COUNTRY} is",
    "Q: What is the capital city of {COUNTRY}? A:",
    "{COUNTRY}'s capital city is",
]

TEMPLATES_HI = [
    "{COUNTRY} की राजधानी है",
    "प्रश्न: {COUNTRY} की राजधानी क्या है? उत्तर:",
    "{COUNTRY} की राजधानी शहर है",
]


def generate_capitals_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    lang: str = "en",
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    assert lang in ("en", "hi")
    rng = random.Random(seed)
    pairs = PAIRS_EN if lang == "en" else PAIRS_HI
    templates = TEMPLATES_EN if lang == "en" else TEMPLATES_HI
    lead = lang == "en"
    device = device or model.cfg.device

    # Pre-filter pairs whose capital has a usable token
    usable = []
    for country, capital in pairs:
        try:
            tid, ntok = try_single_or_first(
                model, capital, leading_space=lead, allow_multi=allow_multi_token_answers
            )
            usable.append((country, capital, tid, ntok))
        except ValueError:
            continue
    if len(usable) < 4:
        raise RuntimeError(f"Too few usable capital tokens for lang={lang}")

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks = []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError(f"Could not generate enough capitals prompts (lang={lang})")
        (country, capital, pos_id, n_pos), (d_country, d_capital, neg_id, n_neg) = rng.sample(usable, 2)
        if pos_id == neg_id:
            continue
        tmpl = rng.choice(templates)
        clean = tmpl.format(COUNTRY=country)
        corr = tmpl.format(COUNTRY=d_country)
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
        pos_ids.append(pos_id)
        neg_ids.append(neg_id)
        n_toks.append((n_pos, n_neg))

    single_rate = sum(1 for a, b in n_toks if a == 1 and b == 1) / max(len(n_toks), 1)
    return TaskBatch(
        clean_tokens=torch.stack(clean_t).to(device),
        corrupted_tokens=torch.stack(corr_t).to(device),
        pos_token_ids=torch.tensor(pos_ids, device=device),
        neg_token_ids=torch.tensor(neg_ids, device=device),
        prompts_clean=clean_s,
        prompts_corrupted=corr_s,
        task="capitals",
        lang=lang,
        corruption="country_swap",
        meta={
            "single_token_answer_rate": single_rate,
            "n_pairs": len(usable),
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
        },
    )
