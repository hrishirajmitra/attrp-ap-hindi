# Does attribution patching reproduce activation patching on IOI? (GPT-2 small)

**Setup:** GPT-2 small, Indirect Object Identification, 100 prompts per run, **5 prompt-sampling seeds** (0 to 4), single RTX 2080 Ti on Ada. Protocol: [PREREGISTRATION.md](PREREGISTRATION.md) plus its amendments. Sources: [literature/PAPERS.md](literature/PAPERS.md), [literature/references.bib](literature/references.bib).

## Question

For the IOI circuit in GPT-2 small, how well does attribution patching (AttrP; Nanda 2023, Syed et al. 2023) rank attention heads relative to full activation patching (AP), and where does it break? Added comparison: a node-level integrated-gradients variant (EAP-IG; after Hanna et al. 2024, Sundararajan et al. 2017).

## Experiment

- **Baseline (arm 0):** AP of every head output (`hook_z`) and MLP output, clean into corrupted (denoising; Heimersheim & Nanda 2024). Metric: logit difference IO − S (Wang et al. 2022), normalised by us into a recovery fraction (our choice, not from Wang). Wang et al. instead patch ABC into IOI (noising direction) and report raw logit-diff drops, so our setup is a different protocol on the same task (their clean mean logit-diff is 3.56; ours 3.52).
- **Comparison (arm 1):** AttrP on the same components.
- **Extra arm (arm 2):** EAP-IG, 5 midpoint steps, interpolating input embeddings from corrupted to clean and reading gradients at each component. This is node-level, not the edge-level method of Hanna et al.
- **Conditions:** ABC corruption (all three names replaced; canonical, primary) and role-flip (IO and S swapped; secondary).
- **Baseline sanity (mean over seeds):** clean logit-diff 3.52, ABC-corrupted -0.01, role-flip-corrupted -3.39. Name movers L9H9, L9H6, L10H0 are at the top of AP under ABC.

## Results (mean ± sd over 5 seeds, [min, max])

| Metric | ABC (primary) | Role-flip |
|---|---|---|
| rho(AP, AttrP), all 144 heads | **0.989 ± 0.007** [0.98, 1.00] | 0.955 ± 0.010 [0.95, 0.97] |
| Precision@14 AP / AttrP / EAP-IG | 0.46 / 0.43 / **0.84** | 0.54 / 0.49 / 0.57 |
| rho(AP, EAP-IG) | 0.40 ± 0.11 | 0.72 ± 0.04 |
| MLP Pearson r with AP: AttrP / EAP-IG | **-0.93** / 0.95 | -0.52 / 1.00 |
| Worst per-layer head r (AttrP) | 0.79 [0.55, 0.98] | 0.57 [0.37, 0.71] |
| AttrP speedup over AP | 27.8x | 27.9x |
| Pre-registered verdict (GO / PARTIAL / NO-GO) | **0 / 5 / 0** | 3 / 2 / 0 |
| EAP-IG "helps" by the pre-registered rule | 5 of 5 | 2 of 5 |

**Post-hoc (not pre-registered), `results/exploratory_rank.json`:** 86 to 88% of heads have |AP| < 0.02, so the all-head Spearman is mostly rank noise among near-zero heads. Restricted to heads with |AP| > 0.05 (about 11 heads), AttrP vs AP gives rho = 0.998 ± 0.004 (ABC) and 0.88 ± 0.03 (role-flip). Top-20 overlap with AP is 100% under ABC (all 5 seeds) and 91% under role-flip. The thresholds 0.02 and 0.05 were chosen after seeing the data.

## Findings

1. **Attribution patching reproduces activation-patching head rankings on IOI.** rho >= 0.95 on every seed in both conditions, about 28x faster. This is robust to prompt sampling. Under the canonical ABC corruption, it is near-identical on the heads that matter (rho about 1.0).
2. **Attribution patching fails on MLP outputs.** It anti-correlates with AP (r = -0.93 ± 0.02 under ABC; -0.52 ± 0.11 under role-flip), while EAP-IG agrees (r >= 0.95). With 12 MLPs per run this is a coarse comparison, but the sign is consistent across all 10 runs.
3. **The pre-registered Precision@14 bar is not a test of AttrP.** AP itself scores only 0.46 under ABC, and AttrP matches it. AP's top 14 includes L10H10, L10H6, L10H1 and L9H0, which Wang et al. identify as backup name movers. They are absent from my literature checklist, so the checklist understates every method.
4. **EAP-IG and AP measure different things.** Under ABC, EAP-IG recovers 0.84 of the checklist (about 12 of 14), including induction, duplicate-token and previous-token heads that AP does not rank highly (specific heads not listed here; see `scores.npz`). Its rank correlation with AP is low (0.40). I interpret this as AP (single-component sufficiency) surfacing late writers while EAP-IG's path integral also credits upstream heads. I did not test that mechanism.
5. **The role-flip verdict is not stable.** It was GO on seed 0 but is 3 GO / 2 PARTIAL over 5 seeds, because Precision@14 sits at the 0.50 threshold (0.43 to 0.57). Do not read the earlier single-seed GO as a result. Per-layer agreement under role-flip also dips below r = 0.5 on some seeds (worst 0.37), so a localisation claim for that condition is not supported.

## Verdict

Under the locked thresholds: **PARTIAL on the canonical ABC setup (5/5 seeds)**, driven entirely by the Precision@14 bar that AP itself fails. The substantive conclusion is findings 1 and 2: AttrP is a faithful, cheap proxy for head-level AP on IOI and an unreliable one for MLP outputs.

## Relation to prior work

Syed et al. report a **weak** attribution-vs-activation correlation (R² = 0.27, slope 0.53) for edges on the Docstring task, and Kramár et al. identify attention-softmax saturation and effect cancellation as failure modes that produce false negatives. Our high rank agreement does not contradict these. We used a different task (IOI), node-level head outputs, a rank statistic and a denoising direction, and we did not test attention-pattern patching (the saturation case). I did not isolate which of these differences explains the gap. The MLP failure is consistent with the general point that the linear approximation is not uniformly reliable.

## Limits

- GPT-2 small and IOI only. 5 prompt seeds give a variance estimate over prompt sampling, not over models or tasks. Runs share one model, so they are not independent evidence about other models.
- Head and MLP outputs only. Query/key/value and attention patterns are not separated, so "where does it break by component type" is answered only as heads vs MLPs.
- The checklist H* is incomplete (finding 3), so Precision@k is a lower bound for every method.
- EAP-IG is a node-level variant with 5 steps. Results do not transfer to Hanna et al.'s edge-level method. Step-count sensitivity is untested.
- Denoising direction only. A noising analysis could rank components differently (Heimersheim & Nanda).
- Finding 4's explanation and the restricted-head analysis are post-hoc interpretations.

## Revision history

A first version of this project reported a PARTIAL verdict from a flawed baseline (non-canonical role-flip data, clean logit-diff 1.01) and a buggy EAP-IG (it overwrote all layers' activations at once, cutting gradient flow). Both were fixed (see PREREGISTRATION amendments). The old run is archived in `results/superseded_full_seed0_roleflip_v0/` and must not be cited. The 5-seed run was added afterwards to address the single-seed limitation.

## Reproduce

```bash
# env lives on gnode071 /ssd_scratch (see slurm/setup_env.sh); pin the node
for c in abc flip; do for s in 0 1 2 3 4; do
  sbatch --nodelist=gnode071 --export=ALL,CORR=$c,SEED=$s ~/mp/slurm/run_all.sbatch
done; done
python scripts/aggregate_seeds.py results        # pre-registered metrics -> results/seed_summary.{json,md}
python scripts/exploratory_rank.py results       # post-hoc restricted-rank analysis (needs numpy/scipy)
```

---

# Extension (v2): Hindi, more tasks, more models (+ stage-2 language profiles)

**Protocol:** [PREREGISTRATION_v2.md](PREREGISTRATION_v2.md). Formal proposal: [proposal/proposal.pdf](proposal/proposal.pdf).

**Two stages, one study.** Stage 1: does AttrP reproduce AP per gated `(model, task, language)` cell? Stage 2: on cells where stage 1 is trustworthy, do English and Hindi **important-head profiles** (AP scores) overlap near a within-language ceiling and above a different-task floor? Without stage 1, an en–hi difference could just be the cheap localiser failing in Hindi.

**Reuse.** Patching engine: ours (primary), optional cross-check with [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG) — see [results/ENGINE_DECISION.md](results/ENGINE_DECISION.md). Agreement data: MultiBLiMP (`data/multiblimp/`). Logit lens / romanization ideas from llm-latent-language and RomanLens.

## Design summary

- **Models (≤2B / 11GB):** GPT-2, Pythia-160m/410m (English controls), Qwen2.5-0.5B/1.5B, Llama-3.2-1B, Gemma-3-1B, optional Sarvam-1.
- **Tasks:** IOI, capitals (factual recall), gender/number agreement, induction copy. Each with parallel English and Hindi templates (`src/data_*.py`).
- **Matching:** competence gate (`scripts/gate.py`) — clean acc ≥ 0.80, clean LD ≥ 0.50, corrupted LD ≤ 0.25, single-token rate ≥ 0.90. Failed cells stay in the coverage table.
- **Metrics:** Spearman ρ; restricted ρ on heads with |AP| > 0.05 (a priori); top-k overlap with AP (replaces checklist Precision@k except for GPT-2 IOI); MLP Pearson; EAP-IG step sweep {5,10,20}; denoising + noising.
- **Hypotheses:** H1 ρ≥0.90 on all gated cells; H2 MLP AttrP failure recurs; H3 Hindi ρ < English ρ (paired); H4 EAP-IG repairs MLP.

## Coverage (offline expectations — lock on Ada)

Until `scripts/gate.py` runs on Ada, `results/gate_coverage.json` holds **expected** pass/fail only (not locked). Offline draft: **52 / 64** cells expected to pass; Hindi fails for GPT-2 and Pythia by design. See `results/gate_coverage.md` and `results/tokenizer_audit.md`.

## Grid results

_Pending Ada runs._ After `slurm/run_grid.sbatch` + `scripts/aggregate_seeds.py`:

| Artifact | Content |
|----------|---------|
| `results/grid_summary.md` | ρ / top-14 overlap / MLP r by cell |
| `results/h3_language_paired.md` | Paired Hindi vs English bootstrap |
| `results/grid_rho.png` | Heatmap of mean ρ |

## Closing v1 open limits

| v1 limit | v2 action |
|----------|-----------|
| One model / one task | Multi-model × multi-task × en/hi grid |
| Post-hoc \|AP\| thresholds | Restricted ρ threshold **0.05 locked a priori** |
| Checklist Precision@14 | Top-k overlap with AP (checklist only for GPT-2 IOI) |
| EAP-IG step count untested | Sweep 5, 10, 20 on seed 0 |
| Denoising only | Noising secondary (seed 0) |
