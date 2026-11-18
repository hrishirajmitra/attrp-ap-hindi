"""Tokenization helpers: single-token checks, first-token fallback, tokens-per-word."""

from __future__ import annotations

from typing import Any

from transformer_lens import HookedTransformer


def token_ids_for_word(
    model: HookedTransformer,
    word: str,
    *,
    leading_space: bool = True,
) -> list[int]:
    """Tokenize a word, optionally with a leading space (GPT-2 BPE convention)."""
    text = (" " + word) if leading_space else word
    toks = model.to_tokens(text, prepend_bos=False)[0]
    return [int(t) for t in toks.tolist()]


def single_token_id(
    model: HookedTransformer,
    word: str,
    *,
    leading_space: bool = True,
) -> int:
    ids = token_ids_for_word(model, word, leading_space=leading_space)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not a single token ({len(ids)} toks)")
    return ids[0]


def first_token_id(
    model: HookedTransformer,
    word: str,
    *,
    leading_space: bool = True,
) -> int:
    """First subword id — used when Hindi/multilingual words are multi-token."""
    ids = token_ids_for_word(model, word, leading_space=leading_space)
    if not ids:
        raise ValueError(f"{word!r} tokenizes to empty")
    return ids[0]


def try_single_or_first(
    model: HookedTransformer,
    word: str,
    *,
    leading_space: bool = True,
    allow_multi: bool = True,
) -> tuple[int, int]:
    """Return (token_id, n_tokens). Prefer single-token; else first token if allow_multi."""
    ids = token_ids_for_word(model, word, leading_space=leading_space)
    if len(ids) == 1:
        return ids[0], 1
    if allow_multi and ids:
        return ids[0], len(ids)
    raise ValueError(f"{word!r} is multi-token ({len(ids)}) and multi not allowed")


def filter_single_token(
    model: HookedTransformer,
    words: list[str],
    *,
    leading_space: bool = True,
) -> list[str]:
    good = []
    for w in words:
        try:
            single_token_id(model, w, leading_space=leading_space)
            good.append(w)
        except ValueError:
            continue
    return good


def tokens_per_word_stats(
    model: HookedTransformer,
    words: list[str],
    *,
    leading_space: bool = True,
) -> dict[str, Any]:
    lengths = [len(token_ids_for_word(model, w, leading_space=leading_space)) for w in words]
    n = len(lengths) or 1
    return {
        "n_words": len(words),
        "mean_tokens": sum(lengths) / n,
        "min_tokens": min(lengths) if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
        "frac_single_token": sum(1 for L in lengths if L == 1) / n,
        "per_word": {w: L for w, L in zip(words, lengths)},
    }


def hindi_leading_space(model: HookedTransformer) -> bool:
    """Devanagari often should not use a leading English space; try both."""
    # Heuristic: if model is GPT-2-like English BPE, leading space helps English
    # but hurts Hindi. Prefer no leading space for scripts outside Latin.
    name = (model.cfg.model_name or "").lower()
    if any(k in name for k in ("gpt2", "pythia", "opt-")):
        return False  # English space before Devanagari rarely helps
    return False
