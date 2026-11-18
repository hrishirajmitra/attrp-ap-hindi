"""MultiBLiMP subject–verb agreement loader (English + Hindi).

Data: HF jumelet/multiblimp (local copies under data/multiblimp/{eng,hin}_data.tsv).
Hindi phenomena include SV-G (gender), SV-P (person), SV-# (number).

Clean prompt = MultiBLiMP `prefix` (text before the agreeing verb).
Metric = logit(correct_verb) − logit(swap_head) at the final token.
Corruption = another example's prefix of equal token length (context swap),
keeping the original answer tokens so corrupted logit-diff collapses when the
model tracks the subject features.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from .batch import TaskBatch
from .tokenize import try_single_or_first

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "multiblimp"

MIN_SINGLE_POOL = 150
LANG_FILE = {"en": "eng_data.tsv", "hi": "hin_data.tsv"}


def _load_rows(lang: str, data_dir: Path) -> list[dict]:
    path = data_dir / LANG_FILE[lang]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download with:\n"
            f"  curl -L -o {path} "
            f"https://huggingface.co/datasets/jumelet/multiblimp/resolve/main/"
            f"{'eng' if lang == 'en' else 'hin'}/data.tsv"
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _divergence(sen: str, wrong: str) -> tuple[str, str, str] | None:
    """Return (shared_prefix, correct_word, wrong_word) at the first differing word."""
    a, b = sen.split(), wrong.split()
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if i == 0 or i >= min(len(a), len(b)):
        return None
    return " ".join(a[:i]), a[i], b[i]


def generate_multiblimp_batch(
    model: HookedTransformer,
    n: int = 100,
    seed: int = 0,
    device: str | torch.device | None = None,
    lang: str = "en",
    phenomenon: str | None = None,
    data_dir: Path | None = None,
    allow_multi_token_answers: bool = True,
) -> TaskBatch:
    """Build a TaskBatch from MultiBLiMP prefixes + verb/swap_head tokens."""
    assert lang in ("en", "hi")
    rng = random.Random(seed)
    device = device or model.cfg.device
    data_dir = Path(data_dir) if data_dir else DEFAULT_DIR
    rows = _load_rows(lang, data_dir)
    if phenomenon:
        rows = [r for r in rows if r.get("phenomenon") == phenomenon]
    if not rows:
        raise RuntimeError(f"No MultiBLiMP rows for lang={lang} phenomenon={phenomenon}")

    lead = True  # next word follows a space in running text, for both scripts
    usable: list[dict] = []
    for r in rows:
        # NOTE: the TSV `verb` / `swap_head` columns are NOT reliably the word
        # right after `prefix` (checked: 197/200 mismatches in hin). Derive the
        # contrast from the first word where `sen` and `wrong_sen` diverge.
        pair = _divergence(r.get("sen") or "", r.get("wrong_sen") or "")
        if pair is None:
            continue
        prefix, verb, swap = pair
        # strip clause punctuation glued to the word (Hindi 'देगा,' vs 'दिए,')
        verb = verb.rstrip(",.;:!?।\"”’)")
        swap = swap.rstrip(",.;:!?।\"”’)")
        if not verb or not swap or not prefix or verb == swap:
            continue
        # English: skip contractions / non-alphabetic contrasts ("is" vs "'m") -- the first
        # token of the wrong form is an apostrophe, so the logit difference is meaningless.
        if lang == "en" and not (re.fullmatch(r"[A-Za-z]+", verb) and re.fullmatch(r"[A-Za-z]+", swap)):
            continue
        try:
            pos_id, n_pos = try_single_or_first(
                model, verb, leading_space=lead, allow_multi=allow_multi_token_answers
            )
            neg_id, n_neg = try_single_or_first(
                model, swap, leading_space=lead, allow_multi=allow_multi_token_answers
            )
        except ValueError:
            continue
        if pos_id == neg_id:
            continue
        usable.append(
            {
                **r,
                "prefix": prefix,
                "pos_id": pos_id,
                "neg_id": neg_id,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "verb": verb,
                "swap": swap,
            }
        )
    if len(usable) < 8:
        raise RuntimeError(
            f"Too few usable MultiBLiMP items for lang={lang} (got {len(usable)})"
        )
    # Prefer pairs whose contrast words are both single tokens (so the logit difference is a
    # clean one-token contrast). If fewer than MIN_SINGLE_POOL such pairs exist the full pool is
    # used and the batch single-token rate will (correctly) fail the competence gate.
    pool_single_rate = sum(1 for u in usable if u["n_pos"] == 1 and u["n_neg"] == 1) / len(usable)
    single_pool = [u for u in usable if u["n_pos"] == 1 and u["n_neg"] == 1]
    pool_filtered = len(single_pool) >= MIN_SINGLE_POOL
    if pool_filtered:
        usable = single_pool

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks = []
    metas = []
    target_len = None
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 400:
            raise RuntimeError(
                f"Could not build enough equal-length MultiBLiMP pairs (lang={lang})"
            )
        a, b = rng.sample(usable, 2)
        if a["pos_id"] == b["pos_id"] and a["prefix"] == b["prefix"]:
            continue
        ct = model.to_tokens(a["prefix"])[0]
        rt = model.to_tokens(b["prefix"])[0]
        if ct.numel() != rt.numel():
            continue
        if target_len is None:
            target_len = ct.numel()
        if ct.numel() != target_len:
            continue
        clean_s.append(a["prefix"])
        corr_s.append(b["prefix"])
        clean_t.append(ct)
        corr_t.append(rt)
        pos_ids.append(a["pos_id"])
        neg_ids.append(a["neg_id"])
        n_toks.append((a["n_pos"], a["n_neg"]))
        metas.append(
            {
                "phenomenon": a.get("phenomenon"),
                "grammatical_feature": a.get("grammatical_feature"),
                "verb": a["verb"],
                "swap": a["swap"],
            }
        )

    single_rate = sum(1 for x, y in n_toks if x == 1 and y == 1) / max(len(n_toks), 1)
    phen_counts: dict[str, int] = {}
    for m in metas:
        phen_counts[m["phenomenon"] or "?"] = phen_counts.get(m["phenomenon"] or "?", 0) + 1

    return TaskBatch(
        clean_tokens=torch.stack(clean_t).to(device),
        corrupted_tokens=torch.stack(corr_t).to(device),
        pos_token_ids=torch.tensor(pos_ids, device=device),
        neg_token_ids=torch.tensor(neg_ids, device=device),
        prompts_clean=clean_s,
        prompts_corrupted=corr_s,
        task="agreement",
        lang=lang,
        corruption="prefix_swap",
        meta={
            "source": "multiblimp",
            "single_token_answer_rate": single_rate,
            "n_pool": len(usable),
            "pool_single_token_rate": pool_single_rate,
            "pool_filtered_to_single_token": pool_filtered,
            "phenomenon_counts": phen_counts,
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
            "items": metas,
        },
    )
