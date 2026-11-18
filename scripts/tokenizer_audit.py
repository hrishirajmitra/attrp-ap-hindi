#!/usr/bin/env python3
"""Audit TransformerLens loadability and Hindi/English tokenization per model.

Writes results/tokenizer_audit.json and results/tokenizer_audit.md.
Run on Ada (needs HF downloads + TL). Offline fallback writes expected notes.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import MODELS


def _probe_sets() -> dict:
    from src.data_agreement import EN_ITEMS, HI_ITEMS
    from src.data_capitals import PAIRS_EN, PAIRS_HI
    from src.data_induction import AKSHARA_HI, LETTERS_EN
    from src.data_ioi import NAMES_EN, NAMES_HI

    return {
        "names_en": NAMES_EN,
        "names_hi": NAMES_HI,
        "capitals_en": [c for _, c in PAIRS_EN],
        "capitals_hi": [c for _, c in PAIRS_HI],
        "verbs_en": [v for a, b, va, vb, t in EN_ITEMS for v in (va, vb)],
        "verbs_hi": [v for a, b, va, vb, t in HI_ITEMS for v in (va, vb)],
        "letters_en": LETTERS_EN,
        "akshara_hi": AKSHARA_HI,
    }


def audit_model(model_name: str, device: str) -> dict:
    row: dict = {"model": model_name, "loaded": False, "error": None, "cfg": {}, "token_stats": {}}
    try:
        import torch
        from transformer_lens import HookedTransformer

        try:
            model = HookedTransformer.from_pretrained(model_name, device=device)
        except Exception:
            model = HookedTransformer.from_pretrained(
                model_name, device=device, dtype=torch.float16
            )
        row["loaded"] = True
        from src.tokenize import tokens_per_word_stats

        row["cfg"] = {
            "n_layers": model.cfg.n_layers,
            "n_heads": model.cfg.n_heads,
            "d_model": model.cfg.d_model,
            "n_ctx": model.cfg.n_ctx,
            "tokenizer_name": getattr(model.cfg, "tokenizer_name", None),
            "model_name": model.cfg.model_name,
        }
        for key, words in _probe_sets().items():
            lead = key.endswith("_en") and "letter" not in key and "akshara" not in key
            if "letter" in key or "akshara" in key:
                lead = False
            try:
                row["token_stats"][key] = tokens_per_word_stats(
                    model, words, leading_space=lead
                )
            except Exception as e:
                row["token_stats"][key] = {"error": str(e)}
        # Hindi tokens-per-word covariate on a fixed sentence
        hi_sent = "भारत की राजधानी दिल्ली है"
        en_sent = "The capital of India is Delhi"
        row["sentence_tokens"] = {
            "hi": int(model.to_tokens(hi_sent).shape[-1]),
            "en": int(model.to_tokens(en_sent).shape[-1]),
            "hi_text": hi_sent,
            "en_text": en_sent,
        }
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-2000:]
    return row


def expected_offline_notes() -> list[dict]:
    """Literature / prior-knowledge notes when models cannot be loaded locally."""
    notes = []
    for name, meta in MODELS.items():
        notes.append(
            {
                "model": name,
                "loaded": None,
                "offline_note": True,
                "family": meta["family"],
                "expected_langs": meta["expected_langs"],
                "tl_note": meta.get("tl_note", "Expect TL 2.x/3.x support for GPT-2/Pythia; Qwen2.5 and Llama-3.2 commonly supported; Gemma-3 / Sarvam may need fallback"),
                "tokenization_expectation": (
                    "English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). "
                    "Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; "
                    "still filter for single-token answers at generation time."
                    if "hi" in meta["expected_langs"]
                    else "English-only tokenizer: Hindi tasks expected to fail single-token gate."
                ),
            }
        )
    return notes


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "# Tokenizer / TransformerLens audit",
        "",
        "| Model | Loaded | Layers | Heads | HI names single-tok frac | EN names single-tok frac | HI sent toks | EN sent toks | Notes |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("offline_note"):
            lines.append(
                f"| `{r['model']}` | — | — | — | — | — | — | — | {r.get('tokenization_expectation','')} |"
            )
            continue
        cfg = r.get("cfg") or {}
        ts = r.get("token_stats") or {}
        hi = ts.get("names_hi") or {}
        en = ts.get("names_en") or {}
        sent = r.get("sentence_tokens") or {}
        note = r.get("error") or ""
        lines.append(
            f"| `{r['model']}` | {r.get('loaded')} | {cfg.get('n_layers','')} | {cfg.get('n_heads','')} | "
            f"{hi.get('frac_single_token','')} | {en.get('frac_single_token','')} | "
            f"{sent.get('hi','')} | {sent.get('en','')} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--offline", action="store_true", help="Write expected notes only (no HF load)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = args.models or list(MODELS.keys())

    if args.offline:
        rows = expected_offline_notes()
    else:
        rows = [audit_model(m, args.device) for m in models]

    (args.out_dir / "tokenizer_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out_dir / "tokenizer_audit.md").write_text(to_markdown(rows))
    print(to_markdown(rows))
    print(f"Wrote {args.out_dir / 'tokenizer_audit.json'}")


if __name__ == "__main__":
    main()
