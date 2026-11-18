# Pre-registration: Attribution Patching vs Activation Patching on IOI (GPT-2 Small)

**Project:** A (chosen from mini-project options)  
**Date:** 2026-10-21  
**Status:** Locked before analysis

## One-line question

Does attribution patching reproduce activation-patching head rankings on the Indirect Object Identification (IOI) circuit in GPT-2 small, and where does the linear approximation fail?

## Motivation

Activation patching is the ground-truth causal intervention for circuit discovery but is expensive (one forward pass per component). Attribution patching (Syed et al., 2023) approximates the same effect with a single backward pass. The field uses attribution patching as a cheap proxy; this experiment measures agreement and localises failure modes on a well-studied circuit.

## Model and task

| Item | Choice |
|------|--------|
| Model | `gpt2` (GPT-2 small, 124M) via TransformerLens |
| Task | Indirect Object Identification (IOI) |
| Metric | Logit difference: `logit(IO) − logit(S)` at the final token |
| Dataset | Synthetic IOI prompts (Wang et al. 2022 style), N = 100 clean/corrupted pairs |
| Corruption | Swap the subject name that appears first (ABC → ABA / ABB style name swap used by TransformerLens IOI demos) |
| Device | Ada `u22`, `--constraint=2080ti` |

## Baseline (arm 0)

**Full activation patching** of every attention head output (`blocks.L.attn.hook_z` per head) and every MLP output (`blocks.L.hook_mlp_out`) from the corrupted run into the clean run (or vice versa; we use the standard “patch corrupted → clean and measure recovery of clean logit-diff” convention from the TransformerLens IOI tutorial).

For each component \(c\):

\[
\text{AP}(c) = \frac{\text{metric}(\text{patch } c) - \text{metric}(\text{corrupted})}{\text{metric}(\text{clean}) - \text{metric}(\text{corrupted})}
\]

Save a layer × head heatmap of AP scores and the ranked list of heads.

## Intervention (arm 1)

**Attribution patching** on the same components. For activation \(a_c\):

\[
\text{AttrP}(c) = (a_c^{\text{clean}} - a_c^{\text{corrupted}}) \cdot \nabla_{a_c} \text{metric}(\text{corrupted})
\]

Normalise AttrP into the same [recovery-fraction] scale by dividing by \(\text{metric}(\text{clean}) - \text{metric}(\text{corrupted})\) when reporting rank correlation (raw AttrP used for ranking).

## Refinement arm (arm 2)

**Edge Attribution Patching with Integrated Gradients (EAP-IG)** style scores for head outputs: average gradients along the linear path from corrupted to clean activations (Hanna et al., 2024), with \(n_{\text{steps}} = 5\). Same components as arms 0–1.

## Known IOI heads (for precision@k)

Using the standard GPT-2-small IOI circuit inventory (Wang et al. / Conmy et al.):

- **Name movers:** L9H9, L9H6, L10H0
- **S-inhibition:** L7H3, L7H9, L8H6, L8H10
- **Induction:** L5H5, L5H8, L6H9
- **Duplicate token:** L0H1, L3H0
- **Previous token:** L2H2, L4H11

Union of these sets = **positive head set** \(H^\star\) (\(|H^\star| = 14\)).

## Pre-stated metrics

1. **Spearman \(\rho\)** between AP and AttrP rankings over all 144 heads (12 layers × 12 heads).
2. **Spearman \(\rho\)** between AP and EAP-IG rankings over the same 144 heads.
3. **Precision@k** for recovering \(H^\star\) at \(k \in \{5, 10, 14, 20\}\) for AP, AttrP, and EAP-IG.
4. **Disagreement localisation:** for each layer, and separately for head vs MLP components, report Pearson \(r\) between AP and AttrP; flag layers/types with \(r < 0.5\).
5. **Wall-clock:** seconds for full AP vs AttrP vs EAP-IG (same hardware).

## Success / failure thresholds (pre-registered)

| Claim | Threshold |
|-------|-----------|
| “AttrP is a safe proxy for head ranking” | Spearman \(\rho(\text{AP}, \text{AttrP}) \ge 0.80\) over all heads |
| “AttrP recovers the known circuit” | Precision@14(AttrP) \(\ge 0.50\) (at least 7/14 known heads in top 14) |
| “EAP-IG improves on AttrP” | \(\rho(\text{AP}, \text{EAP-IG}) - \rho(\text{AP}, \text{AttrP}) \ge 0.05\) **or** Precision@14 improves by \(\ge 2\) heads |
| “Failure concentrates in early layers / pattern heads” | Exploratory: if overall \(\rho < 0.80\), report which layers have \(r < 0.5\); no hard threshold (qualitative claim) |

**Primary verdict rules (pick one):**

- **GO (AttrP safe):** \(\rho \ge 0.80\) and Precision@14 ≥ 0.50.
- **PARTIAL:** one of the two holds.
- **NO-GO (AttrP unsafe as sole ranking tool):** both fail.
- **EAP-IG helps / does not help:** evaluated only if AttrP is PARTIAL or NO-GO, or as secondary claim under the improvement threshold above.

## Negative-result reading (allowed)

- High agreement → AttrP is a validated cheap proxy at this scale; report compute ratio.
- Low agreement, especially if concentrated where softmax/attention patterns matter → document a localised failure mode of the linear approximation.
- EAP-IG not helping → integrated gradients do not buy ranking fidelity on this circuit at \(n_{\text{steps}}=5\).

Any of these is a publishable mini-project result if thresholds above are applied honestly.

## What this experiment cannot tell you (limits)

- Results are for GPT-2 small IOI only; no claim about larger models or other tasks.
- We patch head *outputs* / MLP outputs, not Q/K/V or attention patterns separately (optional appendix if time).
- Precision@k uses a literature head list that itself came from activation patching; circularity for “discovering” the circuit is avoided by treating \(H^\star\) as an external checklist, not as proof of circuit completeness.
- Single seed for data generation (template sampling); we fix `seed=0` and do not claim variance over prompt sampling unless we add a 3-seed sensitivity check.

## Compute budget

- Expected: &lt; 5 GPU-hours on one 2080 Ti.
- Code + results: `~/mp/results/` (NFS home).
- Venv + HF cache: `/ssd_scratch/$USER/` on a pinned compute node (`~/mp/.compute_node`).
- Note: `/share1` is login-only on Ada and is **not** used for compute caches.

## Analysis blinding

Metrics and thresholds above are locked in this file before inspecting intervention heatmaps for the writeup verdict. Baseline AP numbers may be computed first to verify the IOI task works (sanity: clean logit-diff &gt; 0, corrupted ≈ 0 or negative).

---

## Amendments (added after the first run; thresholds above are unchanged)

1. **Dataset (2026-10-22).** The first run used a non-canonical role-flip corruption (clean logit-diff 1.01 vs the usual 3 to 4). The primary condition is now the canonical ABBA/BABA templates with ABC corruption (all three names replaced by fresh distinct names). The role-flip is retained as a secondary condition.
2. **EAP-IG bug fix.** The original implementation overwrote all layers' activations simultaneously, severing gradient flow. It now interpolates input embeddings (5 midpoint steps) and reads gradients at each component. Thresholds are unchanged.
3. **Post-hoc analysis.** `scripts/exploratory_rank.py` restricts rank agreement to heads with |AP| above 0.02 or 0.05. This is exploratory, not pre-registered, and thresholds were chosen after seeing the data.
4. The first-run results (`results/superseded_full_seed0_roleflip_v0/`) are superseded and must not be cited.
5. **Seeds (added after the first run).** The pre-registration specified a single seed (seed 0) and treated more seeds as optional. To address that limitation, seeds 0 to 4 were run for both conditions. Thresholds are unchanged. Reported verdicts are counts over the 5 seeds, and `scripts/aggregate_seeds.py` produces the summary. The role-flip verdict turned out to be seed-dependent (3 GO, 2 PARTIAL).
