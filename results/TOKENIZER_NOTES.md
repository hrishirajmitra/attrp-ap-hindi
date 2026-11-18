# Tokenizer / TL support notes (planning draft)

Full machine-readable audit: `tokenizer_audit.json` / `tokenizer_audit.md`.
Re-run on Ada without `--offline` to fill measured tokens-per-word.

## Expectations by family

| Family | TL 2.x/3.x | Hindi tokenization | Gate implication |
|--------|------------|--------------------|------------------|
| GPT-2 | Yes (anchor) | Devanagari heavily multi-token | Hindi cells fail single-token / competence gate (control) |
| Pythia | Yes | Same English BPE | Hindi expected fail |
| Qwen2.5 | Usually yes | Stronger multilingual coverage | Hindi may pass if answers single-token |
| Llama-3.2 | Usually yes | Multilingual | Same |
| Gemma-3 | TBD — may need fallback | Multilingual | Document load failure; do not silently substitute |
| Sarvam-1 | TBD; optional | Indic-focused | Drop first if VRAM or TL fails |

## Metric rule

Prefer single-token answer tokens. If multi-token after filtering, use **first-subword** logit-diff and record `single_token_answer_rate` in batch meta (gated ≥ 0.90).
