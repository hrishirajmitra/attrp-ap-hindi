# Competence gate coverage

Thresholds: clean_acc≥0.8, clean_ld≥0.5, corr_ld≤0.25, single_tok≥0.9, N=100.

| Model | Task | Lang | Pass | Clean acc | Clean LD | Corr LD | Single-tok | Reasons |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| `gpt2` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | ioi | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | capitals | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | agreement | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `gpt2` | induction | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | ioi | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | capitals | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | agreement | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-160m` | induction | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | ioi | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | capitals | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | agreement | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `EleutherAI/pythia-410m` | induction | hi | expected=False | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | ioi | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | capitals | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | agreement | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-0.5B` | induction | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | ioi | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | capitals | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | agreement | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `Qwen/Qwen2.5-1.5B` | induction | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | ioi | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | capitals | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | agreement | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `meta-llama/Llama-3.2-1B` | induction | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | ioi | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | capitals | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | agreement | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `google/gemma-3-1b-pt` | induction | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | ioi | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | ioi | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | capitals | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | capitals | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | agreement | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | agreement | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | induction | en | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |
| `sarvamai/sarvam-1` | induction | hi | expected=True | — | — | — | — | offline expectation only — run scripts/gate.py on Ada to lock |

Passed cells: 0 / 64
Expected (offline) pass cells: 52 / 64
