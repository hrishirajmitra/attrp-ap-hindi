"""Morphological agreement tasks.

Hindi: gender agreement on the past participle / adjective-like verb form.
  e.g. "लड़का गया" (masc) vs "लड़की गई" (fem).
English parallel: subject–verb number agreement.
  e.g. "The boy walks" vs "The boys walk".

Metric: logit(agreeing_form) − logit(disagreeing_form).
Corruption: swap the subject gender/number so the original agreeing form becomes wrong.
"""

from __future__ import annotations

import random

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .tokenize import try_single_or_first

# Hindi: (masc_subject, fem_subject, masc_verb, fem_verb, template)
# Template ends just before the verb so the model predicts the verb form.
# Prefer MultiBLiMP (src/data_multiblimp.py) for the main agreement analysis;
# these hand templates remain as a small fallback / smoke set.
HI_ITEMS = [
    ("लड़का", "लड़की", "गया", "गई", "{SUBJ} स्कूल"),
    ("आदमी", "औरत", "आया", "आई", "{SUBJ} बाज़ार"),
    ("कुत्ता", "बिल्ली", "बैठा", "बैठी", "{SUBJ} घर में"),
    ("छात्र", "छात्रा", "पढ़ा", "पढ़ी", "{SUBJ} किताब"),
    ("बेटा", "बेटी", "हँसा", "हँसी", "{SUBJ} ज़ोर से"),
    ("भाई", "बहन", "सोया", "सोई", "{SUBJ} कमरे में"),
    ("नौकर", "नौकरानी", "रुका", "रुकी", "{SUBJ} द्वार पर"),
    ("डॉक्टर", "नर्स", "लौटा", "लौटी", "{SUBJ} अस्पताल से"),
    ("किसान", "महिला", "बोया", "बोई", "{SUBJ} खेत में बीज"),
    ("शेर", "शेरनी", "दौड़ा", "दौड़ी", "{SUBJ} जंगल में"),
    ("राजा", "रानी", "बैठा", "बैठी", "{SUBJ} सिंहासन पर"),
    ("गुरु", "गुरुमाँ", "बोला", "बोली", "{SUBJ} धीरे से"),
]

# Stretch (H7): ergative ne-perfective — verb agrees with OBJECT, not subject.
# (subj_masc, subj_fem, obj_masc, obj_fem, verb_masc, verb_fem, template)
# Template ends before the verb; we vary the object gender while keeping subject fixed.
HI_NE_PERFECTIVE = [
    ("लड़के", "लड़की", "पत्र", "किताब", "लिखा", "लिखी", "{SUBJ} ने {OBJ} "),
    ("राम", "सीता", "पत्र", "चिट्ठी", "पढ़ा", "पढ़ी", "{SUBJ} ने {OBJ} "),
    ("आदमी", "औरत", "दरवाज़ा", "खिड़की", "खोला", "खोली", "{SUBJ} ने {OBJ} "),
    ("छात्र", "छात्रा", "प्रश्न", "कविता", "पूछा", "पूछी", "{SUBJ} ने {OBJ} "),
    ("बेटे", "बेटी", "खेल", "कहानी", "शुरू किया", "शुरू की", "{SUBJ} ने {OBJ} "),
    ("भाई", "बहन", "गाना", "कविता", "गाया", "गायी", "{SUBJ} ने {OBJ} "),
]

# English number agreement: (sg_subj, pl_subj, sg_verb, pl_verb, template)
EN_ITEMS = [
    ("boy", "boys", "walks", "walk", "The {SUBJ}"),
    ("girl", "girls", "runs", "run", "The {SUBJ}"),
    ("dog", "dogs", "barks", "bark", "The {SUBJ}"),
    ("cat", "cats", "sleeps", "sleep", "The {SUBJ}"),
    ("teacher", "teachers", "writes", "write", "The {SUBJ}"),
    ("student", "students", "reads", "read", "The {SUBJ}"),
    ("bird", "birds", "flies", "fly", "The {SUBJ}"),
    ("child", "children", "plays", "play", "The {SUBJ}"),
    ("man", "men", "drives", "drive", "The {SUBJ}"),
    ("woman", "women", "sings", "sing", "The {SUBJ}"),
    ("doctor", "doctors", "works", "work", "The {SUBJ}"),
    ("nurse", "nurses", "helps", "help", "The {SUBJ}"),
]


def generate_agreement_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    lang: str = "en",
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    assert lang in ("en", "hi")
    rng = random.Random(seed)
    items = EN_ITEMS if lang == "en" else HI_ITEMS
    lead = lang == "en"
    device = device or model.cfg.device

    usable = []
    for a, b, va, vb, tmpl in items:
        try:
            id_a, na = try_single_or_first(model, va, leading_space=lead, allow_multi=allow_multi_token_answers)
            id_b, nb = try_single_or_first(model, vb, leading_space=lead, allow_multi=allow_multi_token_answers)
            if id_a == id_b:
                continue
            usable.append((a, b, va, vb, tmpl, id_a, id_b, na, nb))
        except ValueError:
            continue
    if len(usable) < 4:
        raise RuntimeError(f"Too few usable agreement items for lang={lang}")

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks = []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError(f"Could not generate enough agreement prompts (lang={lang})")
        a, b, va, vb, tmpl, id_a, id_b, na, nb = rng.choice(usable)
        use_a = rng.random() < 0.5  # True → masculine/singular is correct
        if use_a:
            clean = tmpl.format(SUBJ=a)
            corr = tmpl.format(SUBJ=b)
            pos_id, neg_id = id_a, id_b
            n_pos, n_neg = na, nb
        else:
            clean = tmpl.format(SUBJ=b)
            corr = tmpl.format(SUBJ=a)
            pos_id, neg_id = id_b, id_a
            n_pos, n_neg = nb, na
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
        task="agreement",
        lang=lang,
        corruption="feature_swap",
        meta={
            "single_token_answer_rate": single_rate,
            "n_items": len(usable),
            "feature": "number" if lang == "en" else "gender",
            "source": "hand_templates",
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
        },
    )


def generate_ne_perfective_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    """Stretch H7: Hindi ergative ne-perfective (verb agrees with object)."""
    rng = random.Random(seed)
    device = device or model.cfg.device
    usable = []
    for subj_m, subj_f, obj_m, obj_f, vm, vf, tmpl in HI_NE_PERFECTIVE:
        try:
            id_m, nm = try_single_or_first(
                model, vm, leading_space=False, allow_multi=allow_multi_token_answers
            )
            id_f, nf = try_single_or_first(
                model, vf, leading_space=False, allow_multi=allow_multi_token_answers
            )
        except ValueError:
            continue
        if id_m == id_f:
            continue
        usable.append((subj_m, subj_f, obj_m, obj_f, vm, vf, tmpl, id_m, id_f, nm, nf))
    if len(usable) < 3:
        raise RuntimeError("Too few usable ne-perfective items")

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks = []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError("Could not generate enough ne-perfective prompts")
        subj_m, subj_f, obj_m, obj_f, vm, vf, tmpl, id_m, id_f, nm, nf = rng.choice(usable)
        subj = rng.choice([subj_m, subj_f])  # subject gender should not matter
        use_m = rng.random() < 0.5
        if use_m:
            clean = tmpl.format(SUBJ=subj, OBJ=obj_m)
            corr = tmpl.format(SUBJ=subj, OBJ=obj_f)
            pos_id, neg_id, n_pos, n_neg = id_m, id_f, nm, nf
        else:
            clean = tmpl.format(SUBJ=subj, OBJ=obj_f)
            corr = tmpl.format(SUBJ=subj, OBJ=obj_m)
            pos_id, neg_id, n_pos, n_neg = id_f, id_m, nf, nm
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
        task="agreement_ne",
        lang="hi",
        corruption="object_gender_swap",
        meta={
            "single_token_answer_rate": single_rate,
            "n_items": len(usable),
            "feature": "object_gender_ergative",
            "source": "hand_ne_perfective",
            "stretch": True,
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
        },
    )
