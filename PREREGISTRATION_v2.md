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
| `bigscience/bloom-560m` | Multilingual (replaces Gemma-3-1B: TransformerLens 2.15 has no Gemma-3 support; deviation logged in Amendment 2) | en, hi |
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

### Amendment 1 (2026-11-18, before any gate or grid result exists)

This amendment aligns the plan with `proposal/proposal.pdf` and fixes the analysis rules in advance.
No cluster result had been produced when it was written (see git history: committed before the first
gate/grid job). Gate thresholds above are unchanged.

**Scope**
- Core tables: GPT-2, Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B, Gemma-3-1B (if it loads) on `ioi`, `capitals`, `agreement`.
- Appendix (English robustness): Pythia-160m/410m (all tasks) and the `induction` task (all models).
- Sarvam-1 is dropped (no TransformerLens support). `agreement_ne` (ne-perfective) is not run.
- Romanised Hindi IOI (`hi_roman`) is a **core** condition for models expected to handle Hindi.
- Both directions (denoising, noising) are run with all 5 seeds (supersedes "noising seed 0 only").
- Models run in float32; Llama-3.2-1B / Gemma-3-1B use the ungated `unsloth/*` mirrors if the official repos are refused (provenance recorded in the gate output).

**Metric clarification.** Top-k overlap ranks heads by |score| (matches the proposal). The signed-score overlap
is also stored (`topk_overlap_signed_*`). The GO/PARTIAL/NO-GO rule uses the |score| overlap.

**Hypotheses (final numbering, matches the proposal)**
- **H1:** AttrP head rho >= 0.90 in gated cells. Reported as the fraction of core cells with seed-mean rho >= 0.90 (denoising); H1 "holds" only if all core cells reach it.
- **H2:** MLP Pearson r(AP, AttrP) < 0 in a majority of gated core cells.
- **H3:** On matched (model, task) pairs, Hindi agreement is lower than English. Tested two ways, both reported:
  (a) paired bootstrap of the difference in seed-mean head rho (hi - en), 95% CI over seeds, per pair and pooled;
  (b) equivalence (TOST, margin +/-0.05 on rho, 90% CI): "equivalent" if the pooled CI lies inside the margin; "Hindi worse" only if the pooled CI lies entirely below -0.05.
- **H4:** Cell-level rho decreases with tokenizer fertility (tokens per word) and increases with clean logit difference. Estimated as (a) Spearman correlation across all core cells with a cell-bootstrap CI and (b) OLS of the paired rho difference (hi - en) on the paired differences in fertility and in clean logit difference, with a pair-bootstrap CI. The question is whether the language effect remains after controlling for both.
- **H5:** EAP-IG (5 steps) yields higher MLP agreement than first-order AttrP (replaces the old H4). Also reported: head rho vs EAP-IG steps {5, 10, 20} and wall-clock cost per method.

**Script vs language.** Devanagari Hindi IOI vs romanised Hindi IOI vs English IOI, same models: reported as paired differences in head rho with bootstrap CIs. No directional claim is preregistered.

**Reporting rules.** The gate coverage table is reported first, before any agreement number. Cells that fail the gate make no contribution to agreement claims and are not rerun with changed prompts. Pythia and induction appear only in the appendix.


## Amendment 2 (during gate phase, before any grid result)

1. **Model substitution.** `google/gemma-3-1b-pt` cannot be loaded by TransformerLens 2.15 (not in its model table; TL>=2.16 needs torch>=2.6, unavailable on the cluster build). It is replaced by `bigscience/bloom-560m` (Hindi in pretraining corpus, TL-supported). Decided before any grid result.
2. **Sign convention fix.** In the noising direction AttrP/EAP-IG scores had the opposite sign to AP (delta.grad estimates -(effect)); fixed to the AP convention. Found by the smoke test (rho = -0.92); no study results existed.
3. **MultiBLiMP agreement cleaning.** (a) clause punctuation stripped from the contrast words, (b) English contractions ("is" vs "'m") dropped, (c) pairs whose first tokens coincide dropped, (d) if >=100 pairs have single-token contrast words on both sides the pool is restricted to them (otherwise the single-token rate fails the gate), (e) prompts are a fixed window: first token + last K-1 prefix tokens, K = clamp(30th percentile of prefix lengths, 8, 24), so no padding is needed. Window size, pool size and pool single-token rate are logged per cell.
4. **Gate rule changes (made after seeing the first gate numbers, before any AP/AttrP comparison).** (a) The corrupted-logit-difference criterion becomes `corr_ld <= 0.25 OR corr_ld <= 0.5 * clean_ld` (agreement/induction corruptions leave a residual grammatical prior, so an absolute 0.25 cap rejected every agreement cell although the corruption removes most of the effect). (b) Cells failing *only* the single-token criterion form a **first-token tier**: their logit difference is a first-token contrast; they are run but analysed only as a sensitivity set, never in H1-H5 main tests. Strict-tier cells (all criteria) form the main tests.
5. **Capitals.** The `Mexico` entry was dropped (answer is copied from the prompt); the capitals pool is restricted to single-token capitals when >= 8 exist, and prompts use the modal token length (instead of the length of the first draw).
