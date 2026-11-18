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

MIN_SINGLE_POOL = 100
MIN_WINDOW, MAX_WINDOW = 8, 24
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
    cand: list[dict] = []
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
        cand.append(
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
    if not cand:
        raise RuntimeError(f"No MultiBLiMP candidates for lang={lang}")
    # Descriptor of the whole candidate pool (before any filtering): how often is the contrast a
    # clean one-token-vs-one-token pair in this model's tokenizer?
    pool_single_rate = sum(1 for u in cand if u["n_pos"] == 1 and u["n_neg"] == 1) / len(cand)
    # A first-token contrast with identical first tokens is meaningless -> drop.
    usable = [u for u in cand if u["pos_id"] != u["neg_id"]]
    # Prefer pairs whose contrast words are both single tokens. If fewer than MIN_SINGLE_POOL
    # exist the (first-token) pool is used and the batch single-token rate fails the gate.
    single_pool = [u for u in usable if u["n_pos"] == 1 and u["n_neg"] == 1]
    pool_filtered = len(single_pool) >= MIN_SINGLE_POOL
    if pool_filtered:
        usable = single_pool
    if len(usable) < 8:
        raise RuntimeError(
            f"Too few usable MultiBLiMP items for lang={lang} (got {len(usable)} of {len(cand)})"
        )

    # Fixed-length window: every prompt is [first token] + the last (K-1) tokens of its prefix,
    # so clean/corrupted prompts and all prompts in the batch share one length (no padding).
    # K = min(MAX_WINDOW, 30th percentile of prefix lengths in the pool), >= MIN_WINDOW.
    toks_all = [model.to_tokens(u["prefix"])[0] for u in usable]
    lens = sorted(int(t.numel()) for t in toks_all)
    K = max(MIN_WINDOW, min(MAX_WINDOW, lens[int(0.3 * (len(lens) - 1))]))
    keep = [(u, t) for u, t in zip(usable, toks_all) if t.numel() >= K]
    if len(keep) < 8:
        raise RuntimeError(f"Too few MultiBLiMP prefixes with >= {K} tokens (lang={lang})")

    def _window(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t[:1], t[-(K - 1):]])

    clean_s, corr_s, pos_ids, neg_ids = [], [], [], []
    clean_t, corr_t = [], []
    n_toks = []
    metas = []
    seen_clean: set[tuple] = set()
    tries = 0
    while len(clean_s) < n:
        tries += 1
        if tries > n * 400:
            raise RuntimeError(f"Could not build enough MultiBLiMP pairs (lang={lang})")
        (a, ta), (b, tb) = rng.sample(keep, 2)
        ct, rt = _window(ta), _window(tb)
        if torch.equal(ct, rt) or a["pos_id"] == b["pos_id"] and a["neg_id"] == b["neg_id"]:
            continue
        key = tuple(ct.tolist())
        if key in seen_clean and len(keep) >= n:
            continue
        seen_clean.add(key)
        clean_s.append(model.to_string(ct[1:]))
        corr_s.append(model.to_string(rt[1:]))
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
            "n_pool": len(keep),
            "n_candidates": len(cand),
            "window_tokens": K,
            "pool_single_token_rate": pool_single_rate,
            "pool_filtered_to_single_token": pool_filtered,
            "phenomenon_counts": phen_counts,
            "metric": "first_token_logit_diff" if single_rate < 1.0 else "single_token_logit_diff",
            "items": metas,
        },
    )
