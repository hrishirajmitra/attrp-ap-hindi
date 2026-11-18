"""Synthetic IOI prompt pairs (English and Hindi).

Canonical setup (Wang et al. 2022): ABBA / BABA templates, answer = IO.
Corruptions:
  - "abc":  replace all three names with fresh, distinct names.
  - "flip": swap the IO and S roles.
"""

from __future__ import annotations

import random

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .tokenize import filter_single_token, first_token_id, try_single_or_first

# Back-compat alias
IOIBatch = TaskBatch

NAMES_EN = [
    "Mary", "John", "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace",
    "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yvonne", "Zack", "Amy", "Tom", "Anna", "James", "Laura", "Mark", "Susan",
    "Peter", "Julia", "Adam", "Emily", "Daniel",
]

# Common Hindi given names; filtered to single-/first-token per model at generation time.
NAMES_HI = [
    "राम", "सीता", "अमित", "नीता", "राज", "गीता", "अनन्या", "आरव",
    "कबीर", "मीरा", "रोहन", "प्रिया", "अजय", "नेहा", "विवेक", "पूजा",
    "अरुण", "दीपा", "मोहन", "सुनीता", "विकास", "कविता", "संजय", "अनीता",
    "राहुल", "कमल", "लता", "सुरेश", "माया", "दीपक", "रीना", "अनिल",
]

PLACES_EN = ["store", "park", "restaurant", "school", "hospital", "office", "garden", "library"]
OBJECTS_EN = ["drink", "book", "ring", "kiss", "bone", "letter", "gift", "necklace"]

PLACES_HI = ["बाज़ार", "पार्क", "रेस्तरां", "स्कूल", "अस्पताल", "दफ़्तर", "बगीचा", "पुस्तकालय"]
OBJECTS_HI = ["पानी", "किताब", "अंगूठी", "उपहार", "पत्र", "हार", "फूल", "मिठाई"]

TEMPLATES_EN = [
    "Then, {X} and {Y} went to the {PLACE}. {Z} gave a {OBJ} to",
    "When {X} and {Y} got a {OBJ} at the {PLACE}, {Z} decided to give it to",
    "After {X} and {Y} went to the {PLACE}, {Z} gave a {OBJ} to",
    "Then, {X} and {Y} had a lot of fun at the {PLACE}. {Z} gave a {OBJ} to",
]

# Hindi IOI: verb-final. Full sentence would be
#   "{X} और {Y} {PLACE} गए। {Z} ने {OBJ} {IO} को दिया।"
# Prompt ends immediately before the recipient name so the model predicts IO.
# Native-speaker review required before citing results.
TEMPLATES_HI = [
    "फिर, {X} और {Y} {PLACE} गए। {Z} ने {OBJ}",
    "जब {X} और {Y} {PLACE} पहुँचे, तो {Z} ने {OBJ}",
    "{X} और {Y} के {PLACE} जाने के बाद, {Z} ने {OBJ}",
    "फिर, {X} और {Y} ने {PLACE} में मज़ा किया। {Z} ने {OBJ}",
]

# Romanized Hindi (stretch / H7): same structure, Latin script.
TEMPLATES_HI_ROMAN = [
    "Phir, {X} aur {Y} {PLACE} gaye. {Z} ne {OBJ}",
    "Jab {X} aur {Y} {PLACE} pahunche, to {Z} ne {OBJ}",
    "{X} aur {Y} ke {PLACE} jaane ke baad, {Z} ne {OBJ}",
    "Phir, {X} aur {Y} ne {PLACE} mein maza kiya. {Z} ne {OBJ}",
]
NAMES_HI_ROMAN = [
    "Ram", "Sita", "Amit", "Neeta", "Raj", "Geeta", "Ananya", "Arav",
    "Kabir", "Meera", "Rohan", "Priya", "Ajay", "Neha", "Vivek", "Pooja",
]
PLACES_HI_ROMAN = ["bazaar", "park", "school", "hospital", "daftar", "bagicha"]
OBJECTS_HI_ROMAN = ["paani", "kitaab", "uphaar", "patra", "haar", "mithai"]


def _single_token_id(model: HookedTransformer, word: str) -> int:
    """English IOI: require a true single token (with leading space)."""
    tid, n = try_single_or_first(model, word, leading_space=True, allow_multi=False)
    _ = n
    return tid


def single_token_names(model: HookedTransformer, lang: str = "en") -> list[str]:
    if lang == "en":
        names = NAMES_EN
        lead = True
    elif lang == "hi_roman":
        names = NAMES_HI_ROMAN
        lead = True
    else:
        names = NAMES_HI
        lead = True
    if lang in ("en", "hi_roman"):
        good = filter_single_token(model, names, leading_space=lead)
        if len(good) < 8 and lang == "hi_roman":
            # Relax to first-subword
            good = []
            for n in names:
                try:
                    try_single_or_first(model, n, leading_space=True, allow_multi=True)
                    good.append(n)
                except ValueError:
                    continue
    else:
        # Prefer single-token; fall back to first-subword Devanagari names
        good = []
        for n in names:
            try:
                tid, ntok = try_single_or_first(model, n, leading_space=True, allow_multi=True)
                _ = tid
                if ntok == 1:
                    good.append(n)
            except ValueError:
                continue
        if len(good) < 8:
            for n in names:
                try:
                    try_single_or_first(model, n, leading_space=True, allow_multi=True)
                    if n not in good:
                        good.append(n)
                except ValueError:
                    continue
    if len(good) < 8:
        raise RuntimeError(f"Too few usable names for lang={lang} (got {len(good)})")
    return good


def generate_ioi_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    corruption: str = "abc",
    lang: str = "en",
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    assert corruption in ("abc", "flip")
    assert lang in ("en", "hi", "hi_roman")
    rng = random.Random(seed)
    names = single_token_names(model, lang=lang)
    device = device or model.cfg.device
    if lang == "en":
        templates, places, objects, lead = TEMPLATES_EN, PLACES_EN, OBJECTS_EN, True
    elif lang == "hi_roman":
        templates, places, objects, lead = (
            TEMPLATES_HI_ROMAN,
            PLACES_HI_ROMAN,
            OBJECTS_HI_ROMAN,
            True,
        )
    else:
        templates, places, objects, lead = TEMPLATES_HI, PLACES_HI, OBJECTS_HI, True

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks_answers = []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError(f"Could not generate enough equal-length IOI prompts (lang={lang})")
        io, s, c1, c2, c3 = rng.sample(names, 5)
        place, obj = rng.choice(places), rng.choice(objects)
        tmpl = rng.choice(templates)
        abba = rng.random() < 0.5
        x, y = (io, s) if abba else (s, io)
        clean = tmpl.format(X=x, Y=y, Z=s, PLACE=place, OBJ=obj)
        if corruption == "abc":
            corr = tmpl.format(X=c1, Y=c2, Z=c3, PLACE=place, OBJ=obj)
        else:
            corr = tmpl.format(X=y, Y=x, Z=io, PLACE=place, OBJ=obj)
        ct = model.to_tokens(clean)[0]
        rt = model.to_tokens(corr)[0]
        if ct.numel() != rt.numel():
            continue
        if target_len is None:
            target_len = ct.numel()
        if ct.numel() != target_len:
            continue
        try:
            io_id, n_io = try_single_or_first(
                model, io, leading_space=lead, allow_multi=allow_multi_token_answers
            )
            s_id, n_s = try_single_or_first(
                model, s, leading_space=lead, allow_multi=allow_multi_token_answers
            )
        except ValueError:
            continue
        clean_s.append(clean)
        corr_s.append(corr)
        clean_t.append(ct)
        corr_t.append(rt)
        pos_ids.append(io_id)
        neg_ids.append(s_id)
        n_toks_answers.append((n_io, n_s))

    single_rate = sum(1 for a, b in n_toks_answers if a == 1 and b == 1) / max(len(n_toks_answers), 1)
    return TaskBatch(
        clean_tokens=torch.stack(clean_t).to(device),
        corrupted_tokens=torch.stack(corr_t).to(device),
        pos_token_ids=torch.tensor(pos_ids, device=device),
        neg_token_ids=torch.tensor(neg_ids, device=device),
        prompts_clean=clean_s,
        prompts_corrupted=corr_s,
        task="ioi",
        lang=lang,
        corruption=corruption,
        meta={
            "single_token_answer_rate": single_rate,
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
            "n_names": len(names),
        },
    )


# Keep old helper name used elsewhere
def _single_token_id_legacy(model: HookedTransformer, word: str) -> int:
    return first_token_id(model, word, leading_space=True)
