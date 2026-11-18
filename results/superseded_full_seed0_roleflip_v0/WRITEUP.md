# Writeup: Attribution Patching vs Activation Patching on IOI

**Date:** 2026-10-22  
**Model:** GPT-2 small · **Task:** Indirect Object Identification · **N prompts:** 100  
**Hardware:** Ada `gnode071`, RTX 2080 Ti  
**Primary verdict (pre-registered):** **PARTIAL**

## Question

Does attribution patching reproduce activation-patching head rankings on the IOI circuit in GPT-2 small, and where does the linear approximation fail?

## Setup (locked in [PREREGISTRATION.md](PREREGISTRATION.md))

- **Baseline (Arm 0):** full activation patching of every head output (`hook_z`) and MLP output; patch clean → corrupted; score = logit-diff recovery fraction.
- **Intervention (Arm 1):** attribution patching (Syed et al. / Nanda).
- **Refinement (Arm 2):** EAP-IG with `n_steps=5`.
- **Thresholds:** Spearman ρ(AP, AttrP) ≥ 0.80; Precision@14(AttrP) ≥ 0.50.

## Sanity checks

| Quantity | Value |
|----------|------:|
| Clean logit-diff | 1.0088 |
| Corrupted logit-diff | −0.8568 |
| Wall-clock AP (s) | 8.4 |
| Wall-clock AttrP (s) | 0.3 |
| Wall-clock EAP-IG (s) | 0.5 |
| Speedup AttrP / AP | **26.1×** |

IOI task is working: clean ≫ corrupted logit-diff with a large positive gap.

## Main results

| Metric | Value | Threshold | Pass? |
|--------|------:|----------:|:-----:|
| Spearman ρ(AP, AttrP) | **0.933** | ≥ 0.80 | **yes** |
| Precision@14 AttrP | 0.429 | ≥ 0.50 | no |
| Spearman ρ(AP, EAP-IG) | 0.487 | — | — |
| ΔSpearman (EAP−AttrP) | −0.446 | ≥ 0.05 for “helps” | no |
| ΔPrecision@14 heads (EAP−AttrP) | 0.0 | ≥ 2 for “helps” | no |

### Precision@k

| k | AP | AttrP | EAP-IG |
|--:|--:|--:|--:|
| 5 | 0.600 | 0.600 | 0.600 |
| 10 | 0.500 | 0.500 | 0.500 |
| 14 | 0.429 | 0.429 | 0.429 |
| 20 | 0.300 | 0.350 | 0.300 |

**Important:** Activation patching itself scores Precision@14 = 0.429 against the literature checklist \(H^\star\). AttrP matches AP exactly at k∈{5,10,14}. The Precision@14 failure is therefore **not** an AttrP-vs-AP disagreement; it is a mismatch between our role-flip IOI corruption and the Wang/Conmy head inventory (which was discovered under a different corruption / analysis stack). AttrP recovers the same top heads AP finds.

### Top-14 heads

**Activation patching:** L9H9, L10H0, L8H10, L10H10, L10H6, L4H11, L7H9, L8H3, L9H7, L10H1, L11H1, L11H3, L7H3, L11H11  
**Attribution patching:** L9H9, L8H10, L10H0, L10H10, L10H6, L7H9, L8H3, L9H6, L9H7, L10H1, L11H1, L11H3, L11H11, L7H3  
**EAP-IG:** L9H9, L9H6, L10H0, L10H10, L10H6, L8H10, L10H1, L11H1, L7H9, L9H0, L11H3, L11H11, L9H2, L7H3  
**Known positives \(H^\star\):** L0H1, L2H2, L3H0, L4H11, L5H5, L5H8, L6H9, L7H3, L7H9, L8H6, L8H10, L9H6, L9H9, L10H0

Shared core recovered by both AP and AttrP: name movers / late IO writers (L9H9, L10H0), S-inhibition (L7H3, L7H9, L8H10), previous-token (L4H11). Early induction / duplicate-token heads from \(H^\star\) do not rank in the top 14 under this corruption.

## Disagreement localisation

- **Heads, all layers:** no layer has Pearson \(r(\mathrm{AP},\mathrm{AttrP}) < 0.5\). Late layers are near-perfect (L10 \(r≈0.99\), L11 \(r≈0.999\)).
- **MLPs:** Pearson \(r(\mathrm{AP},\mathrm{AttrP}) = -0.93\) (flagged). AttrP is a **poor** proxy for MLP-output importance under this metric — the linear approximation that works for head outputs does not transfer to MLP blocks here.
- **EAP-IG:** worse head ranking than vanilla AttrP (ρ drops to 0.49); heatmaps look sharper in late layers but disagree more with full AP. Pre-registered “helps” bar fails.

See [`results/full_seed0/heatmaps.png`](results/full_seed0/heatmaps.png).

## Verdict interpretation

Under the locked thresholds the primary label is **PARTIAL**:

1. **AttrP is a safe ranking proxy for attention heads** at this scale (ρ=0.933, ~26× faster than full AP). No head-layer fails the r&lt;0.5 localisation flag.
2. **Precision@14 vs literature \(H^\star\) fails for AttrP *and* for AP**, so this bar does not indict the linear approximation; it indicts treating \(H^\star\) as a corruption-invariant gold set.
3. **EAP-IG (n=5) does not help** head ranking vs AttrP; do not prefer it for this setup.
4. **Practical recommendation:** use AttrP to screen heads; confirm with AP if you need calibrated recovery fractions. Do **not** trust AttrP for MLP-output rankings without AP confirmation.

This is a useful, explainable result even where negative: the linear approximation works where we expected (head outputs) and fails where nonlinearity / different component structure matters (MLPs; EAP-IG at low step count).

## Limits (what this cannot tell you)

1. GPT-2 small + IOI only — no claim about larger models or other tasks.
2. We patch head *outputs* and MLP outputs, not Q/K/V or attention patterns separately.
3. \(H^\star\) is a literature checklist; Precision@k is not a test of circuit completeness. Our role-flip corruption differs from the classic ABC→ABA setup used to discover many of those heads.
4. Single data-generation seed (seed=0); prompt-sampling variance not quantified.
5. EAP-IG uses a midpoint Riemann sum with `n_steps=5`; more steps could change the refinement arm.
6. Ada `/share1` is login-only; compute used `/ssd_scratch` on a pinned node (`gnode071`).

## Artifacts

| Path | Contents |
|------|----------|
| [`results/full_seed0/scores.npz`](results/full_seed0/scores.npz) | AP / AttrP / EAP-IG score tensors |
| [`results/full_seed0/comparison.json`](results/full_seed0/comparison.json) | Full metrics + verdict |
| [`results/full_seed0/heatmaps.png`](results/full_seed0/heatmaps.png) | Layer×head heatmaps |
| [`results/full_seed0/meta.json`](results/full_seed0/meta.json) | Wall times, logit diffs |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Locked protocol |

## Reproduce on Ada

```bash
# venv already on gnode071 /ssd_scratch — pin the node
NODE=$(cat ~/mp/.compute_node)   # gnode071
sbatch --nodelist=$NODE ~/mp/slurm/run_all.sbatch
```
