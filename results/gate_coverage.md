# Competence gate coverage

Thresholds: clean_acc>=0.8, clean_ld>=0.5, corr_ld<=max(0.25, 0.5*clean_ld), single_tok>=0.9, N=100.

| Model | Task | Lang | Tier | Clean acc | Clean LD | Corr LD | Single-tok | Tok/word | Reasons |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|---|
| `EleutherAI/pythia-160m` | ioi | en | strict | 0.940 | 4.908 | 0.063 | 1.000 | 1.077 | ok |
| `EleutherAI/pythia-160m` | capitals | en | strict | 1.000 | 4.425 | -4.179 | 1.000 | 1.040 | ok |
| `EleutherAI/pythia-160m` | agreement | en | strict | 0.820 | 2.591 | 0.503 | 1.000 | 1.213 | ok |
| `EleutherAI/pythia-160m` | induction | en | strict | 0.880 | 1.915 | 0.511 | 1.000 | 1.000 | ok |
| `EleutherAI/pythia-410m` | ioi | en | strict | 1.000 | 4.333 | -0.011 | 1.000 | 1.077 | ok |
| `EleutherAI/pythia-410m` | capitals | en | strict | 1.000 | 7.620 | -7.730 | 1.000 | 1.040 | ok |
| `EleutherAI/pythia-410m` | agreement | en | strict | 0.830 | 3.359 | 0.632 | 1.000 | 1.213 | ok |
| `EleutherAI/pythia-410m` | induction | en | strict | 0.920 | 2.646 | 0.167 | 1.000 | 1.000 | ok |
| `Qwen/Qwen2.5-0.5B` | ioi | en | strict | 1.000 | 5.763 | -0.411 | 1.000 | 1.077 | ok |
| `Qwen/Qwen2.5-0.5B` | ioi | hi | fail | 0.300 | 0.005 | 0.015 | 0.000 | 4.386 | clean_acc 0.300 < 0.8; clean_ld 0.005 < 0.5; single_tok_rate 0.000 < 0.9 |
| `Qwen/Qwen2.5-0.5B` | capitals | en | strict | 1.000 | 9.400 | -8.697 | 1.000 | 1.068 | ok |
| `Qwen/Qwen2.5-0.5B` | capitals | hi | fail | 0.480 | -0.134 | 0.110 | 0.000 | 4.384 | clean_acc 0.480 < 0.8; clean_ld -0.134 < 0.5; single_tok_rate 0.000 < 0.9 |
| `Qwen/Qwen2.5-0.5B` | agreement | en | strict | 0.830 | 3.450 | 0.941 | 1.000 | 1.171 | ok |
| `Qwen/Qwen2.5-0.5B` | agreement | hi | fail |  |  |  |  |  | exception: RuntimeError: Too few usable MultiBLiMP items for lang=hi (got 0 of 1439) |
| `Qwen/Qwen2.5-0.5B` | induction | en | strict | 0.960 | 3.475 | 0.006 | 1.000 | 1.000 | ok |
| `Qwen/Qwen2.5-0.5B` | induction | hi | fail | 0.620 | 0.689 | 0.318 | 1.000 | 1.800 | clean_acc 0.620 < 0.8 |
| `Qwen/Qwen2.5-0.5B` | ioi | hi_roman | fail | 0.500 | -0.078 | -0.067 | 0.000 | 1.815 | clean_acc 0.500 < 0.8; clean_ld -0.078 < 0.5; single_tok_rate 0.000 < 0.9 |
| `Qwen/Qwen2.5-1.5B` | ioi | en | strict | 0.970 | 5.176 | -0.055 | 1.000 | 1.077 | ok |
| `Qwen/Qwen2.5-1.5B` | ioi | hi | fail | 0.270 | -0.006 | -0.042 | 0.000 | 4.386 | clean_acc 0.270 < 0.8; clean_ld -0.006 < 0.5; single_tok_rate 0.000 < 0.9 |
| `Qwen/Qwen2.5-1.5B` | capitals | en | strict | 1.000 | 10.783 | -10.299 | 1.000 | 1.068 | ok |
| `Qwen/Qwen2.5-1.5B` | capitals | hi | fail | 0.460 | -0.149 | 0.040 | 0.000 | 4.384 | clean_acc 0.460 < 0.8; clean_ld -0.149 < 0.5; single_tok_rate 0.000 < 0.9 |
| `Qwen/Qwen2.5-1.5B` | agreement | en | strict | 0.870 | 3.852 | 1.082 | 1.000 | 1.171 | ok |
| `Qwen/Qwen2.5-1.5B` | agreement | hi | fail |  |  |  |  |  | exception: RuntimeError: Too few usable MultiBLiMP items for lang=hi (got 0 of 1439) |
| `Qwen/Qwen2.5-1.5B` | induction | en | strict | 0.940 | 3.911 | -0.203 | 1.000 | 1.000 | ok |
| `Qwen/Qwen2.5-1.5B` | induction | hi | fail | 0.750 | 1.210 | 0.492 | 1.000 | 1.800 | clean_acc 0.750 < 0.8 |
| `Qwen/Qwen2.5-1.5B` | ioi | hi_roman | fail | 0.450 | -0.024 | -0.152 | 0.000 | 1.815 | clean_acc 0.450 < 0.8; clean_ld -0.024 < 0.5; single_tok_rate 0.000 < 0.9 |
| `bigscience/bloom-560m` | ioi | en | strict | 1.000 | 4.725 | -0.258 | 1.000 | 1.077 | ok |
| `bigscience/bloom-560m` | ioi | hi | fail | 0.540 | 0.343 | 0.387 | 1.000 | 1.320 | clean_acc 0.540 < 0.8; clean_ld 0.343 < 0.5; corr_ld 0.387 > max(0.25, 0.5*clean_ld) |
| `bigscience/bloom-560m` | capitals | en | strict | 0.940 | 7.704 | -7.725 | 1.000 | 1.037 | ok |
| `bigscience/bloom-560m` | capitals | hi | fail | 0.510 | 0.052 | -0.288 | 0.000 | 1.274 | clean_acc 0.510 < 0.8; clean_ld 0.052 < 0.5; single_tok_rate 0.000 < 0.9 |
| `bigscience/bloom-560m` | agreement | en | strict | 0.870 | 3.262 | 1.331 | 1.000 | 1.178 | ok |
| `bigscience/bloom-560m` | agreement | hi | strict | 0.900 | 4.791 | 1.193 | 1.000 | 1.203 | ok |
| `bigscience/bloom-560m` | induction | en | strict | 0.990 | 4.164 | 0.721 | 1.000 | 1.000 | ok |
| `bigscience/bloom-560m` | induction | hi | strict | 0.980 | 3.104 | 0.636 | 1.000 | 1.000 | ok |
| `bigscience/bloom-560m` | ioi | hi_roman | fail | 0.400 | -0.285 | 0.095 | 0.000 | 1.801 | clean_acc 0.400 < 0.8; clean_ld -0.285 < 0.5; single_tok_rate 0.000 < 0.9 |
| `gpt2` | ioi | en | strict | 0.990 | 3.147 | -0.120 | 1.000 | 1.077 | ok |
| `gpt2` | capitals | en | strict | 0.990 | 4.996 | -4.592 | 1.000 | 1.062 | ok |
| `gpt2` | agreement | en | strict | 0.880 | 3.180 | 0.628 | 1.000 | 1.149 | ok |
| `gpt2` | induction | en | strict | 0.860 | 2.229 | -0.092 | 1.000 | 1.000 | ok |
| `meta-llama/Llama-3.2-1B` | ioi | en | strict | 1.000 | 5.045 | -0.303 | 1.000 | 1.077 | ok |
| `meta-llama/Llama-3.2-1B` | ioi | hi | fail | 0.320 | -1.185 | -1.367 | 0.000 | 2.532 | clean_acc 0.320 < 0.8; clean_ld -1.185 < 0.5; single_tok_rate 0.000 < 0.9 |
| `meta-llama/Llama-3.2-1B` | capitals | en | strict | 1.000 | 10.953 | -10.128 | 1.000 | 1.068 | ok |
| `meta-llama/Llama-3.2-1B` | capitals | hi | first_token | 0.910 | 3.217 | -3.129 | 0.000 | 2.814 | single_tok_rate 0.000 < 0.9 |
| `meta-llama/Llama-3.2-1B` | agreement | en | strict | 0.920 | 3.311 | 0.874 | 1.000 | 1.203 | ok |
| `meta-llama/Llama-3.2-1B` | agreement | hi | fail | 0.690 | 1.317 | -0.937 | 0.020 | 2.522 | clean_acc 0.690 < 0.8; single_tok_rate 0.020 < 0.9 |
| `meta-llama/Llama-3.2-1B` | induction | en | strict | 0.910 | 3.513 | 0.011 | 1.000 | 1.000 | ok |
| `meta-llama/Llama-3.2-1B` | induction | hi | strict | 0.990 | 3.776 | 0.422 | 1.000 | 1.000 | ok |
| `meta-llama/Llama-3.2-1B` | ioi | hi_roman | fail | 0.600 | 0.432 | -0.091 | 0.000 | 1.815 | clean_acc 0.600 < 0.8; clean_ld 0.432 < 0.5; single_tok_rate 0.000 < 0.9 |

Strict: 31; first_token tier: 1; failed: 16; total 48
