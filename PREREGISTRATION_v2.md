# Pre-registration v2: AttrP vs AP across Hindi/English, tasks, and models

**Project:** A extension (multi-cell grid)  
**Date:** 2026-10-25  
**Status:** Locked before grid analysis. Leaves [PREREGISTRATION.md](PREREGISTRATION.md) (v1, GPT-2 IOI) untouched.

## One-line question

Does attribution patching reproduce activation-patching component rankings across English and Hindi, several templated tasks, and several ≤2B models — and where does the linear approximation fail?

## Motivation

v1 found high AttrP–AP head agreement on GPT-2 IOI but MLP anti-correlation. That result is one task × one model. Circuit-discovery work increasingly uses AttrP on multilingual models; we measure agreement on a gated (model, task, language) grid and test whether Hindi systematically worsens the approximation.

## Unit of analysis

A **cell** = `(model, task, lang)` under a fixed direction (`denoising` primary; `noising` secondary). Only cells that pass the competence gate enter the main analysis. Failed cells appear in the coverage table.

## Models (≤2B / 11GB 2080 Ti)

| Model | Role | Expected langs |
|-------|------|----------------|
| `gpt2` | Anchor (v1 replication) | en |
| `EleutherAI/pythia-160m`, `EleutherAI/pythia-410m` | English-only controls | en |
| `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` | Multilingual | en, hi |
| `meta-llama/Llama-3.2-1B` | Multilingual | en, hi |
| `google/gemma-3-1b-pt` | Multilingual (TL support TBD) | en, hi |
| `sarvamai/sarvam-1` | Indic-focused, optional (drop if VRAM/TL fails) | en, hi |

Load via TransformerLens `<4`. Unsupported models: document failure in tokenizer audit; do not silently substitute.

## Tasks (templated, binary logit-diff)

| Task | Metric | Corruption |
|------|--------|------------|
| `ioi` | logit(IO) − logit(S) | ABC name replace (primary); role-flip secondary for IOI only |
| `capitals` | logit(correct capital) − logit(distractor) | Country swap |
| `agreement` | logit(agreeing form) − logit(disagreeing) | Gender (hi) / number (en) subject swap |
| `induction` | logit(copied token) − logit(distractor) | Break A B C A B repeat |

Parallel English/Hindi templates. Multi-token answers use **first-subword** logit-diff; single-token rate is gated.

## Competence gate (run before patching)

On N = 100 prompts, seed = 0:

| Criterion | Threshold |
|-----------|----------:|
| Clean accuracy (pos logit > neg) | ≥ 0.80 |
| Mean clean logit-diff | ≥ 0.50 |
| Mean corrupted logit-diff | ≤ 0.25 |
| Single-token answer rate | ≥ 0.90 |

Script: `scripts/gate.py` → `results/gate_coverage.{json,md}`.  
**Gated cell list:** locked to the `passed_cells` array in `gate_coverage.json` after the Ada gate run. Until then, `expected_pass_cells` from `--offline` is a planning draft only and must not be treated as locked.

## Arms (per gated cell)

- **Arm 0:** Activation patching of every head output (`hook_z`) and MLP output.
- **Arm 1:** Attribution patching on the same components.
- **Arm 2:** Node-level EAP-IG; primary `n_steps = 5`; sweep `{5, 10, 20}` once per cell (seed 0) to close the v1 open limit.

**Directions:** `denoising` (primary), `noising` (secondary, same gated cells).

**Seeds:** 5 prompt-sampling seeds (0–4) per gated cell × primary direction. Noising may use seed 0 only if compute-constrained (state which in the writeup).

## Metrics (per cell)

| Metric | Definition |
|--------|------------|
| Spearman ρ(AP, AttrP) | All heads |
| Restricted Spearman | Heads with \|AP\| > **0.05** (a priori; not post-hoc) |
| Top-k overlap with AP | k ∈ {5, 10, 14, 20}; primary k = 14 |
| MLP Pearson r(AP, AttrP) and r(AP, EAP-IG) | |
| Per-layer Pearson | Flag layers with r < 0.5 |
| Wall-clock speedup | AP / AttrP |

IOI checklist Precision@14 retained only for GPT-2 IOI English (v1 comparability). All other cells use top-k overlap with AP.

## Hypotheses

- **H1:** AttrP head Spearman ρ ≥ 0.90 on every gated cell (all seeds mean).
- **H2:** MLP Pearson r(AP, AttrP) < 0 on a majority of gated cells (AttrP MLP failure recurs).
- **H3:** For each model that has both `en` and `hi` gated for the same task, mean ρ_hi < mean ρ_en (paired over seeds). Two-sided paired bootstrap; report if the inequality reverses.
- **H4:** On cells where MLP r(AP, AttrP) < 0, EAP-IG (5 steps) yields MLP r(AP, EAP-IG) > 0.5.

## Verdict rule (per cell, primary denoising)

- **GO:** ρ(AP, AttrP) ≥ 0.90 **and** top-14 overlap(AP, AttrP) ≥ 0.50  
- **PARTIAL:** exactly one of the two  
- **NO-GO:** neither  

EAP-IG “helps” if ΔSpearman ≥ 0.05 **or** (MLP AttrP r < 0 and MLP EAP-IG r > 0.5).

## Cross-cell analysis

- Grid table: mean ± sd ρ by model × task × language.
- Paired Hindi vs English bootstrap for H3.
- Tokens-per-word (from tokenizer audit) as a reported covariate, not a preregistered moderator test.

## Compute

Ada `u22`, 2080 Ti, pin `--nodelist` for `/ssd_scratch` venv. Array job: `slurm/run_grid.sbatch`.

## What this cannot claim

- Circuits are the same across languages (we measure method agreement, not circuit identity).
- Results above ~2B parameters or outside the gated task set.
- Edge-level EAP-IG as in Hanna et al. (we use node-level).

## Amendments

Append dated notes below; do not edit locked thresholds above after the gate is run.

### Amendment log

- 2026-10-25: Document created. Gate cell list pending Ada `scripts/gate.py` run.
