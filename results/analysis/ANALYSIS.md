# Analysis (rules: PREREGISTRATION_v2.md, Amendment 1)

## 0. Competence gate coverage

43 strict-pass, 1 first-token tier (single-token criterion waived), 22 failed, of 66 cells.

| model | task | lang | tier | acc | clean_ld | single_tok | tok/word | why |
|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-160m | ioi | en | strict | 0.940 | 4.908 | 1.000 | 1.077 |  |
| EleutherAI/pythia-160m | capitals | en | strict | 1.000 | 4.425 | 1.000 | 1.040 |  |
| EleutherAI/pythia-160m | agreement | en | strict | 0.820 | 2.591 | 1.000 | 1.213 |  |
| EleutherAI/pythia-160m | induction | en | strict | 0.880 | 1.915 | 1.000 | 1.000 |  |
| EleutherAI/pythia-410m | ioi | en | strict | 1.000 | 4.333 | 1.000 | 1.077 |  |
| EleutherAI/pythia-410m | capitals | en | strict | 1.000 | 7.620 | 1.000 | 1.040 |  |
| EleutherAI/pythia-410m | agreement | en | strict | 0.830 | 3.359 | 1.000 | 1.213 |  |
| EleutherAI/pythia-410m | induction | en | strict | 0.920 | 2.646 | 1.000 | 1.000 |  |
| Qwen/Qwen2.5-0.5B | ioi | en | strict | 1.000 | 5.763 | 1.000 | 1.077 |  |
| Qwen/Qwen2.5-0.5B | ioi | hi | fail | 0.300 | 0.005 | 0.000 | 4.386 | clean_acc 0.300 < 0.8; clean_ld 0.005 < 0.5; single_tok_rate 0.000 < 0.9 |
| Qwen/Qwen2.5-0.5B | capitals | en | strict | 1.000 | 9.400 | 1.000 | 1.068 |  |
| Qwen/Qwen2.5-0.5B | capitals | hi | fail | 0.480 | -0.134 | 0.000 | 4.384 | clean_acc 0.480 < 0.8; clean_ld -0.134 < 0.5; single_tok_rate 0.000 < 0.9 |
| Qwen/Qwen2.5-0.5B | agreement | en | strict | 0.830 | 3.450 | 1.000 | 1.171 |  |
| Qwen/Qwen2.5-0.5B | agreement | hi | fail | nan | nan | nan | nan | exception: RuntimeError: Too few usable MultiBLiMP items for lang=hi (got 0 of 1439) |
| Qwen/Qwen2.5-0.5B | induction | en | strict | 0.960 | 3.475 | 1.000 | 1.000 |  |
| Qwen/Qwen2.5-0.5B | induction | hi | fail | 0.620 | 0.689 | 1.000 | 1.800 | clean_acc 0.620 < 0.8 |
| Qwen/Qwen2.5-0.5B | ioi | hi_roman | fail | 0.500 | -0.078 | 0.000 | 1.815 | clean_acc 0.500 < 0.8; clean_ld -0.078 < 0.5; single_tok_rate 0.000 < 0.9 |
| Qwen/Qwen2.5-1.5B | ioi | en | strict | 0.970 | 5.176 | 1.000 | 1.077 |  |
| Qwen/Qwen2.5-1.5B | ioi | hi | fail | 0.270 | -0.006 | 0.000 | 4.386 | clean_acc 0.270 < 0.8; clean_ld -0.006 < 0.5; single_tok_rate 0.000 < 0.9 |
| Qwen/Qwen2.5-1.5B | capitals | en | strict | 1.000 | 10.783 | 1.000 | 1.068 |  |
| Qwen/Qwen2.5-1.5B | capitals | hi | fail | 0.460 | -0.149 | 0.000 | 4.384 | clean_acc 0.460 < 0.8; clean_ld -0.149 < 0.5; single_tok_rate 0.000 < 0.9 |
| Qwen/Qwen2.5-1.5B | agreement | en | strict | 0.870 | 3.852 | 1.000 | 1.171 |  |
| Qwen/Qwen2.5-1.5B | agreement | hi | fail | nan | nan | nan | nan | exception: RuntimeError: Too few usable MultiBLiMP items for lang=hi (got 0 of 1439) |
| Qwen/Qwen2.5-1.5B | induction | en | strict | 0.940 | 3.911 | 1.000 | 1.000 |  |
| Qwen/Qwen2.5-1.5B | induction | hi | fail | 0.750 | 1.210 | 1.000 | 1.800 | clean_acc 0.750 < 0.8 |
| Qwen/Qwen2.5-1.5B | ioi | hi_roman | fail | 0.450 | -0.024 | 0.000 | 1.815 | clean_acc 0.450 < 0.8; clean_ld -0.024 < 0.5; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-1b1 | ioi | en | strict | 1.000 | 4.070 | 1.000 | 1.077 |  |
| bigscience/bloom-1b1 | ioi | hi | fail | 0.560 | 0.272 | 1.000 | 1.320 | clean_acc 0.560 < 0.8; clean_ld 0.272 < 0.5 |
| bigscience/bloom-1b1 | capitals | en | strict | 1.000 | 10.216 | 1.000 | 1.037 |  |
| bigscience/bloom-1b1 | capitals | hi | fail | 0.630 | 0.731 | 0.000 | 1.274 | clean_acc 0.630 < 0.8; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-1b1 | agreement | en | strict | 0.830 | 3.335 | 1.000 | 1.178 |  |
| bigscience/bloom-1b1 | agreement | hi | strict | 0.900 | 4.779 | 1.000 | 1.203 |  |
| bigscience/bloom-1b1 | induction | en | strict | 0.970 | 3.927 | 1.000 | 1.000 |  |
| bigscience/bloom-1b1 | induction | hi | strict | 0.920 | 2.018 | 1.000 | 1.000 |  |
| bigscience/bloom-1b1 | ioi | hi_roman | fail | 0.300 | -0.935 | 0.000 | 1.801 | clean_acc 0.300 < 0.8; clean_ld -0.935 < 0.5; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-1b7 | ioi | en | strict | 1.000 | 3.119 | 1.000 | 1.077 |  |
| bigscience/bloom-1b7 | ioi | hi | fail | 0.640 | 0.558 | 1.000 | 1.320 | clean_acc 0.640 < 0.8; corr_ld 0.352 > max(0.25, 0.5*clean_ld) |
| bigscience/bloom-1b7 | capitals | en | strict | 1.000 | 10.136 | 1.000 | 1.037 |  |
| bigscience/bloom-1b7 | capitals | hi | fail | 0.580 | 0.328 | 0.000 | 1.274 | clean_acc 0.580 < 0.8; clean_ld 0.328 < 0.5; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-1b7 | agreement | en | strict | 0.840 | 3.360 | 1.000 | 1.178 |  |
| bigscience/bloom-1b7 | agreement | hi | strict | 0.910 | 5.169 | 1.000 | 1.203 |  |
| bigscience/bloom-1b7 | induction | en | strict | 0.980 | 4.210 | 1.000 | 1.000 |  |
| bigscience/bloom-1b7 | induction | hi | strict | 0.910 | 2.158 | 1.000 | 1.000 |  |
| bigscience/bloom-1b7 | ioi | hi_roman | fail | 0.330 | -0.702 | 0.000 | 1.801 | clean_acc 0.330 < 0.8; clean_ld -0.702 < 0.5; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-560m | ioi | en | strict | 1.000 | 4.725 | 1.000 | 1.077 |  |
| bigscience/bloom-560m | ioi | hi | fail | 0.540 | 0.343 | 1.000 | 1.320 | clean_acc 0.540 < 0.8; clean_ld 0.343 < 0.5; corr_ld 0.387 > max(0.25, 0.5*clean_ld) |
| bigscience/bloom-560m | capitals | en | strict | 0.940 | 7.704 | 1.000 | 1.037 |  |
| bigscience/bloom-560m | capitals | hi | fail | 0.510 | 0.052 | 0.000 | 1.274 | clean_acc 0.510 < 0.8; clean_ld 0.052 < 0.5; single_tok_rate 0.000 < 0.9 |
| bigscience/bloom-560m | agreement | en | strict | 0.870 | 3.262 | 1.000 | 1.178 |  |
| bigscience/bloom-560m | agreement | hi | strict | 0.900 | 4.791 | 1.000 | 1.203 |  |
| bigscience/bloom-560m | induction | en | strict | 0.990 | 4.164 | 1.000 | 1.000 |  |
| bigscience/bloom-560m | induction | hi | strict | 0.980 | 3.104 | 1.000 | 1.000 |  |
| bigscience/bloom-560m | ioi | hi_roman | fail | 0.400 | -0.285 | 0.000 | 1.801 | clean_acc 0.400 < 0.8; clean_ld -0.285 < 0.5; single_tok_rate 0.000 < 0.9 |
| gpt2 | ioi | en | strict | 0.990 | 3.147 | 1.000 | 1.077 |  |
| gpt2 | capitals | en | strict | 0.990 | 4.996 | 1.000 | 1.062 |  |
| gpt2 | agreement | en | strict | 0.880 | 3.180 | 1.000 | 1.149 |  |
| gpt2 | induction | en | strict | 0.860 | 2.229 | 1.000 | 1.000 |  |
| meta-llama/Llama-3.2-1B | ioi | en | strict | 1.000 | 5.045 | 1.000 | 1.077 |  |
| meta-llama/Llama-3.2-1B | ioi | hi | fail | 0.320 | -1.185 | 0.000 | 2.532 | clean_acc 0.320 < 0.8; clean_ld -1.185 < 0.5; single_tok_rate 0.000 < 0.9 |
| meta-llama/Llama-3.2-1B | capitals | en | strict | 1.000 | 10.953 | 1.000 | 1.068 |  |
| meta-llama/Llama-3.2-1B | capitals | hi | first_token | 0.910 | 3.217 | 0.000 | 2.814 | single_tok_rate 0.000 < 0.9 |
| meta-llama/Llama-3.2-1B | agreement | en | strict | 0.920 | 3.311 | 1.000 | 1.203 |  |
| meta-llama/Llama-3.2-1B | agreement | hi | fail | 0.690 | 1.317 | 0.020 | 2.522 | clean_acc 0.690 < 0.8; single_tok_rate 0.020 < 0.9 |
| meta-llama/Llama-3.2-1B | induction | en | strict | 0.910 | 3.513 | 1.000 | 1.000 |  |
| meta-llama/Llama-3.2-1B | induction | hi | strict | 0.990 | 3.776 | 1.000 | 1.000 |  |
| meta-llama/Llama-3.2-1B | ioi | hi_roman | fail | 0.600 | 0.432 | 0.000 | 1.815 | clean_acc 0.600 < 0.8; clean_ld 0.432 < 0.5; single_tok_rate 0.000 < 0.9 |

## 0b. First-token-tier cells (sensitivity set, not in main tests)

| model | task | lang | direction | rho | top14 | mlp_r | clean_ld | tokens_per_word | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.2-1B | capitals | hi | denoising | 0.895 | 0.857 | 0.368 | 3.146 | 2.802 | 5 |

## 1. Agreement per core cell (denoising, mean over seeds)

| model | task | lang | n_seeds | rho | rho_sd | rho_restricted | top14 | mlp_r | clean_ld | tokens_per_word |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | agreement | en | 5 | 0.858 | 0.026 | 1.000 | 0.600 | 0.542 | 3.370 | 1.179 |
| Qwen/Qwen2.5-0.5B | capitals | en | 5 | 0.924 | 0.006 | 1.000 | 0.829 | 0.915 | 9.383 | 1.058 |
| Qwen/Qwen2.5-0.5B | ioi | en | 5 | 0.963 | 0.008 | 0.997 | 1.000 | -0.866 | 4.634 | 1.131 |
| Qwen/Qwen2.5-1.5B | agreement | en | 5 | 0.884 | 0.023 | 0.900 | 0.714 | 0.807 | 3.620 | 1.179 |
| Qwen/Qwen2.5-1.5B | capitals | en | 5 | 0.977 | 0.003 | 0.960 | 0.829 | 0.906 | 10.791 | 1.058 |
| Qwen/Qwen2.5-1.5B | ioi | en | 5 | 0.964 | 0.010 | 0.936 | 0.943 | 0.980 | 4.788 | 1.131 |
| bigscience/bloom-1b1 | agreement | en | 5 | 0.966 | 0.007 | 0.750 | 0.857 | 0.958 | 3.436 | 1.185 |
| bigscience/bloom-1b1 | agreement | hi | 5 | 0.959 | 0.028 | nan | 0.929 | 0.961 | 5.543 | 1.202 |
| bigscience/bloom-1b1 | capitals | en | 5 | 0.990 | 0.006 | 0.860 | 1.000 | 0.890 | 9.873 | 1.039 |
| bigscience/bloom-1b1 | ioi | en | 5 | 0.990 | 0.004 | 0.997 | 1.000 | 0.997 | 4.501 | 1.136 |
| bigscience/bloom-1b7 | agreement | en | 5 | 0.937 | 0.021 | 0.750 | 0.814 | 0.981 | 3.426 | 1.185 |
| bigscience/bloom-1b7 | agreement | hi | 5 | 0.978 | 0.005 | nan | 1.000 | 0.990 | 5.799 | 1.202 |
| bigscience/bloom-1b7 | capitals | en | 5 | 0.992 | 0.004 | 0.980 | 1.000 | 0.955 | 10.212 | 1.039 |
| bigscience/bloom-1b7 | ioi | en | 5 | 0.993 | 0.004 | 0.998 | 1.000 | 0.997 | 3.764 | 1.136 |
| bigscience/bloom-560m | agreement | en | 5 | 0.974 | 0.016 | 0.750 | 0.943 | 0.981 | 3.306 | 1.185 |
| bigscience/bloom-560m | agreement | hi | 5 | 0.966 | 0.021 | 1.000 | 0.886 | 0.982 | 5.559 | 1.202 |
| bigscience/bloom-560m | capitals | en | 5 | 0.990 | 0.003 | 0.840 | 0.986 | 0.899 | 7.669 | 1.039 |
| bigscience/bloom-560m | ioi | en | 5 | 0.988 | 0.004 | 0.997 | 0.971 | 0.998 | 4.851 | 1.136 |
| gpt2 | agreement | en | 5 | 0.930 | 0.015 | 0.750 | 0.829 | 0.246 | 3.118 | 1.162 |
| gpt2 | capitals | en | 5 | 0.995 | 0.003 | 1.000 | 0.971 | 0.992 | 4.954 | 1.054 |
| gpt2 | ioi | en | 5 | 0.989 | 0.007 | 0.998 | 0.986 | -0.928 | 3.522 | 1.131 |
| meta-llama/Llama-3.2-1B | agreement | en | 5 | 0.910 | 0.019 | 1.000 | 0.829 | 0.850 | 3.413 | 1.163 |
| meta-llama/Llama-3.2-1B | capitals | en | 5 | 0.982 | 0.005 | 0.957 | 0.929 | 0.367 | 10.904 | 1.058 |
| meta-llama/Llama-3.2-1B | ioi | en | 5 | 0.977 | 0.007 | 0.927 | 0.857 | 0.062 | 4.405 | 1.131 |

## 2. H1 and H2

- H1: 92% of 24 core cells have seed-mean head rho >= 0.90 (DOES NOT HOLD in all cells).

- H2: MLP r(AP, AttrP) < 0 in 8% of cells (DOES NOT HOLD).

Most common verdict per cell:

| model | task | lang | verdict |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B | agreement | en | PARTIAL |
| Qwen/Qwen2.5-0.5B | capitals | en | GO |
| Qwen/Qwen2.5-0.5B | ioi | en | GO |
| Qwen/Qwen2.5-1.5B | agreement | en | PARTIAL |
| Qwen/Qwen2.5-1.5B | capitals | en | GO |
| Qwen/Qwen2.5-1.5B | ioi | en | GO |
| bigscience/bloom-1b1 | agreement | en | GO |
| bigscience/bloom-1b1 | agreement | hi | GO |
| bigscience/bloom-1b1 | capitals | en | GO |
| bigscience/bloom-1b1 | ioi | en | GO |
| bigscience/bloom-1b7 | agreement | en | GO |
| bigscience/bloom-1b7 | agreement | hi | GO |
| bigscience/bloom-1b7 | capitals | en | GO |
| bigscience/bloom-1b7 | ioi | en | GO |
| bigscience/bloom-560m | agreement | en | GO |
| bigscience/bloom-560m | agreement | hi | GO |
| bigscience/bloom-560m | capitals | en | GO |
| bigscience/bloom-560m | ioi | en | GO |
| gpt2 | agreement | en | GO |
| gpt2 | capitals | en | GO |
| gpt2 | ioi | en | GO |
| meta-llama/Llama-3.2-1B | agreement | en | GO |
| meta-llama/Llama-3.2-1B | capitals | en | GO |
| meta-llama/Llama-3.2-1B | ioi | en | GO |

## 3. H3: Hindi vs English (paired over seeds, core, denoising)

| model | task | mean_rho_hi_minus_en | ci90_lo | ci90_hi | n_seeds |
|---|---|---|---|---|---|
| bigscience/bloom-1b1 | agreement | -0.007 | -0.031 | 0.015 | 5 |
| bigscience/bloom-1b7 | agreement | 0.041 | 0.020 | 0.054 | 5 |
| bigscience/bloom-560m | agreement | -0.008 | -0.027 | 0.013 | 5 |

Pooled mean(rho_hi - rho_en) = +0.009; 90% CI [-0.014, +0.034]; 95% CI [-0.017, +0.040] over 3 pairs.

**Verdict: EQUIVALENT within +/-0.05.**

## 4. H4: tokenizer fertility and clean logit difference

```json
{
 "rho_vs_tokens_per_word": {
  "rho": -0.5086569257267163,
  "ci_lo": -0.7303777152460472,
  "ci_hi": -0.15072175268014715,
  "n": 24
 },
 "rho_vs_clean_ld": {
  "rho": 0.4452173913043478,
  "ci_lo": 0.0722647370726422,
  "ci_hi": 0.7201957303585109,
  "n": 24
 }
}
```

Interpretation guide: in the paired OLS the intercept is the Hindi-minus-English difference in rho that remains at equal fertility and equal clean logit difference.

## 5. Script vs language (IOI)

```json
{}
```

## 6. H5: EAP-IG vs first-order AttrP

```json
{
 "mean_mlp_r_attrp": 0.6859025976343348,
 "mean_mlp_r_eap": 0.6324930251335259,
 "mlp_r_gain_mean": -0.05340957250080883,
 "mlp_r_gain_ci95": [
  -0.20870985156209093,
  0.1045514867148733
 ],
 "head_rho_gain_mean": -0.35584381599494946,
 "head_rho_gain_ci95": [
  -0.38859672312573645,
  -0.32224896984744966
 ],
 "time_ratio_eap_over_attrp": 3.4640429289700085,
 "time_ratio_ap_over_attrp": 48.70011210711196,
 "holds": false,
 "step_sweep_seed0": {
  "5": {
   "mean_head_rho": 0.5864222393626731,
   "mean_mlp_r": 0.6409826070630681,
   "n": 24
  },
  "10": {
   "mean_head_rho": 0.5832068352682388,
   "mean_mlp_r": 0.6616232621766891,
   "n": 24
  },
  "20": {
   "mean_head_rho": 0.58685849944201,
   "mean_mlp_r": 0.6476692046300757,
   "n": 24
  }
 }
}
```

## 7. Noising direction (core, seed mean)
| model | task | lang | direction | rho | top14 | mlp_r | n_seeds |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B | agreement | en | noising | 0.758 | 0.514 | 0.208 | 5 |
| Qwen/Qwen2.5-0.5B | capitals | en | noising | 0.919 | 0.829 | 0.916 | 5 |
| Qwen/Qwen2.5-0.5B | ioi | en | noising | 0.917 | 0.829 | 0.038 | 5 |
| Qwen/Qwen2.5-1.5B | agreement | en | noising | 0.833 | 0.443 | 0.140 | 5 |
| Qwen/Qwen2.5-1.5B | capitals | en | noising | 0.979 | 0.843 | 0.909 | 5 |
| Qwen/Qwen2.5-1.5B | ioi | en | noising | 0.866 | 0.871 | 0.719 | 5 |
| bigscience/bloom-1b1 | agreement | en | noising | 0.954 | 0.814 | 0.659 | 5 |
| bigscience/bloom-1b1 | agreement | hi | noising | 0.898 | 0.871 | 0.372 | 5 |
| bigscience/bloom-1b1 | capitals | en | noising | 0.992 | 1.000 | 0.888 | 5 |
| bigscience/bloom-1b1 | ioi | en | noising | 0.962 | 0.971 | 0.846 | 5 |
| bigscience/bloom-1b7 | agreement | en | noising | 0.934 | 0.871 | 0.892 | 5 |
| bigscience/bloom-1b7 | agreement | hi | noising | 0.947 | 0.871 | 0.752 | 5 |
| bigscience/bloom-1b7 | capitals | en | noising | 0.988 | 1.000 | 0.955 | 5 |
| bigscience/bloom-1b7 | ioi | en | noising | 0.984 | 0.957 | 0.869 | 5 |
| bigscience/bloom-560m | agreement | en | noising | 0.953 | 0.843 | 0.917 | 5 |
| bigscience/bloom-560m | agreement | hi | noising | 0.933 | 0.886 | 0.437 | 5 |
| bigscience/bloom-560m | capitals | en | noising | 0.988 | 1.000 | 0.902 | 5 |
| bigscience/bloom-560m | ioi | en | noising | 0.958 | 0.914 | 0.916 | 5 |
| gpt2 | agreement | en | noising | 0.882 | 0.671 | -0.029 | 5 |
| gpt2 | capitals | en | noising | 0.994 | 0.986 | 0.992 | 5 |
| gpt2 | ioi | en | noising | 0.915 | 0.829 | -0.664 | 5 |
| meta-llama/Llama-3.2-1B | agreement | en | noising | 0.855 | 0.614 | -0.146 | 5 |
| meta-llama/Llama-3.2-1B | capitals | en | noising | 0.980 | 0.957 | 0.364 | 5 |
| meta-llama/Llama-3.2-1B | ioi | en | noising | 0.908 | 0.886 | 0.609 | 5 |

## 8. Appendix: English robustness (Pythia, induction)
| model | task | lang | direction | rho | top14 | mlp_r | clean_ld | n_seeds |
|---|---|---|---|---|---|---|---|---|
| EleutherAI/pythia-160m | agreement | en | denoising | 0.843 | 0.729 | 0.629 | 2.590 | 5 |
| EleutherAI/pythia-160m | capitals | en | denoising | 0.931 | 0.986 | 0.996 | 4.457 | 5 |
| EleutherAI/pythia-160m | induction | en | denoising | 0.926 | 0.929 | -0.571 | 2.005 | 5 |
| EleutherAI/pythia-160m | ioi | en | denoising | 0.951 | 0.943 | -0.887 | 4.620 | 5 |
| EleutherAI/pythia-410m | agreement | en | denoising | 0.921 | 0.786 | 0.466 | 3.369 | 5 |
| EleutherAI/pythia-410m | capitals | en | denoising | 0.983 | 0.943 | 0.970 | 7.768 | 5 |
| EleutherAI/pythia-410m | induction | en | denoising | 0.967 | 0.943 | 0.187 | 2.749 | 5 |
| EleutherAI/pythia-410m | ioi | en | denoising | 0.984 | 0.986 | -0.826 | 3.941 | 5 |
| Qwen/Qwen2.5-0.5B | induction | en | denoising | 0.924 | 0.886 | 0.020 | 3.614 | 5 |
| Qwen/Qwen2.5-1.5B | induction | en | denoising | 0.930 | 0.757 | 0.862 | 3.941 | 5 |
| bigscience/bloom-1b1 | induction | en | denoising | 0.991 | 0.957 | 0.906 | 3.851 | 5 |
| bigscience/bloom-1b1 | induction | hi | denoising | 0.995 | 1.000 | 0.903 | 2.146 | 5 |
| bigscience/bloom-1b7 | induction | en | denoising | 0.987 | 0.943 | 0.952 | 4.190 | 5 |
| bigscience/bloom-1b7 | induction | hi | denoising | 0.991 | 0.929 | 0.958 | 2.087 | 5 |
| bigscience/bloom-560m | induction | en | denoising | 0.988 | 0.943 | 0.984 | 4.160 | 5 |
| bigscience/bloom-560m | induction | hi | denoising | 0.991 | 0.900 | 0.958 | 3.121 | 5 |
| gpt2 | induction | en | denoising | 0.962 | 0.900 | -0.162 | 2.519 | 5 |
| meta-llama/Llama-3.2-1B | induction | en | denoising | 0.957 | 0.886 | 0.245 | 3.688 | 5 |
| meta-llama/Llama-3.2-1B | induction | hi | denoising | 0.969 | 0.929 | 0.414 | 3.529 | 5 |
