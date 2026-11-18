# Tokenizer / TransformerLens audit

| Model | Loaded | Layers | Heads | HI names single-tok frac | EN names single-tok frac | HI sent toks | EN sent toks | Notes |
|---|:---:|---:|---:|---:|---:|---:|---:|---|
| `gpt2` | — | — | — | — | — | — | — | English-only tokenizer: Hindi tasks expected to fail single-token gate. |
| `EleutherAI/pythia-160m` | — | — | — | — | — | — | — | English-only tokenizer: Hindi tasks expected to fail single-token gate. |
| `EleutherAI/pythia-410m` | — | — | — | — | — | — | — | English-only tokenizer: Hindi tasks expected to fail single-token gate. |
| `Qwen/Qwen2.5-0.5B` | — | — | — | — | — | — | — | English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; still filter for single-token answers at generation time. |
| `Qwen/Qwen2.5-1.5B` | — | — | — | — | — | — | — | English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; still filter for single-token answers at generation time. |
| `meta-llama/Llama-3.2-1B` | — | — | — | — | — | — | — | English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; still filter for single-token answers at generation time. |
| `google/gemma-3-1b-pt` | — | — | — | — | — | — | — | English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; still filter for single-token answers at generation time. |
| `sarvamai/sarvam-1` | — | — | — | — | — | — | — | English BPE: Hindi Devanagari heavily multi-token (often 2–6+ toks/word). Multilingual tokenizers (Qwen, Llama-3.2, Gemma, Sarvam): better Hindi coverage; still filter for single-token answers at generation time. |
