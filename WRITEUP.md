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

# Extension (v2): Hindi, more tasks, more models

**Protocol:** [PREREGISTRATION_v2.md](PREREGISTRATION_v2.md) (Amendments 1-2 record every deviation, all made before any AP/AttrP/EAP-IG comparison existed). Formal proposal: [proposal/proposal.pdf](proposal/proposal.pdf). Full analysis: [results/analysis/ANALYSIS.md](results/analysis/ANALYSIS.md); raw per-run table: `results/analysis/cells.csv`; every run: `results/grid/<model>__<task>__<lang>__<direction>__seed<k>/`.

## What was run

- 9 models: GPT-2, Pythia-160m/410m (English controls), Qwen2.5-0.5B/1.5B, Llama-3.2-1B, **BLOOM-560m/1b1/1b7** (Gemma-3-1B was dropped: TransformerLens 2.15 cannot load it; BLOOM has Hindi in its pretraining data). Llama weights come from the ungated `unsloth/Llama-3.2-1B` mirror because the official repo refused the token.
- 4 tasks (IOI, capitals, MultiBLiMP subject-verb agreement, induction) x English/Hindi, plus romanised-Hindi IOI: 66 candidate cells, all gated first (`results/gate_coverage.md`).
- Every gated cell: AP, first-order AttrP and EAP-IG, both directions (denoising, noising), 5 seeds x 100 prompts, fp32, EAP-IG step sweep {5,10,20} on seed 0. **440 runs, 0 failures.** Compute: Ada 2080 Ti, about 4 h wall on up to 12 GPUs across the `irel` and `research` accounts.
- Correctness checks that ran on the cluster: the new engine reproduces the v1 GPT-2 IOI scores to 1e-6 (`results/validate_v1.json`); EAP-IG at alpha=0 equals AttrP exactly, and EAP-IG at 5 steps matches 50 steps (`scripts/test_eap_equiv.py`); the fast AP path is self-checked against the full forward pass in every run.
- Bugs found *by* those checks and fixed before the grid: noising-direction AttrP/EAP-IG had the opposite sign to AP; MultiBLiMP contrasts were polluted by contractions and glued punctuation; capitals used a multi-token answer pool.

## Results (seed means over 5 seeds; "core" = strict-gate cells, excluding Pythia and induction)

**Gate coverage (a result in itself).** 43 of 66 cells pass the strict gate, 1 passes only as a first-token contrast, 22 fail. **English passes almost everywhere. Hindi passes in only two settings: agreement on BLOOM (3 sizes) and induction on BLOOM/Llama.** Hindi IOI, Hindi capitals and romanised-Hindi IOI fail the competence gate in every model (accuracy 0.27-0.64, i.e. the models cannot do the task, so there is nothing to localise). Qwen2.5 and Llama cannot form a single-token Hindi agreement contrast at all (byte-level BPE: the first token of a correct and an incorrect verb form coincide).

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1**: head rho(AP, AttrP) >= 0.90 in every core cell | **Almost** (22 / 24 cells) | Misses: Qwen2.5-0.5B agreement/en (0.858) and Qwen2.5-1.5B agreement/en (0.884). Range of the other cells 0.91-0.995. Noising direction is lower (mean 0.93; Qwen/Llama agreement 0.76-0.86). |
| **H2**: MLP outputs anti-correlate (r < 0) in most cells | **Rejected** (2 / 24 cells) | The v1 GPT-2 IOI failure (r = -0.93) reproduces, and also appears in Qwen2.5-0.5B IOI (-0.87) and in Pythia IOI/induction, but in most cells MLP r is strongly positive (BLOOM: 0.89-0.998). The MLP failure is circuit-specific (IOI-type name-mover MLP structure), not general. |
| **H3**: AttrP is worse in Hindi than English | **Not supported; equivalent within +/-0.05** | Hindi-minus-English rho on agreement, BLOOM 560m/1b1/1b7: -0.008, -0.007, +0.041; pooled +0.009, 95% CI [-0.017, +0.040]. **Scope: one task, one model family, 3 pairs.** |
| **H4**: rho falls with tokenizer fertility, rises with clean logit difference | Supported, **but observational** | Across the 24 core cells, Spearman(rho, tokens/word) = -0.51 [-0.73, -0.15]; Spearman(rho, clean LD) = +0.45 [0.07, 0.72]. Fertility and model/task are entangled (no Hindi cells with high fertility passed the gate), so this does not establish a causal effect of Hindi. |
| **H5**: EAP-IG repairs MLP agreement and is worth its cost | **Rejected** | MLP r gain over AttrP -0.05 (95% CI [-0.21, +0.10]); head rho *drops* by 0.36 (CI [-0.39, -0.32]); cost 3.5x AttrP. Convergence is not the issue (5, 10, 20 steps give the same numbers; EAP at alpha=0 reproduces AttrP exactly). Node-level IG along the embedding path is a worse proxy for AP than plain first-order AttrP here. |

Other numbers: AP costs ~49x AttrP in wall time; first-token-tier cell Llama-3.2-1B capitals/hi: rho = 0.895, top-14 overlap 0.86 (sensitivity only). Pythia (English appendix) behaves like GPT-2: rho 0.84-0.98, IOI MLP r about -0.85.

## Answer to the question

**Cheap first-order attribution patching reproduces activation patching for head ranking in all but two (Qwen agreement/English) of the 24 core cells and gives the same answer in Hindi as in English wherever a Hindi cell is testable. It fails for MLP outputs in IOI-type circuits (GPT-2, Qwen-0.5B, Pythia), not in general. EAP-IG with integrated gradients does not fix this and is worse for heads.**

## Limits (read before citing)

1. **Hindi evidence is narrow**: the only strict-gate Hindi cells with a testable AP/AttrP comparison are agreement and induction on the BLOOM family (plus Llama induction). We cannot say anything about Hindi IOI, factual recall or the script-vs-language question: no model passed the competence gate. This is a limit on the *models*, which is itself the headline finding about Hindi.
2. English agreement windows are shorter (about 7 tokens of context) than Hindi (about 23 tokens), because the window is set per language from prefix lengths; sequence length is a confound for H3 and H4 and is logged per cell (`seq_len`).
3. H3 has only 3 pairs, all from one family; the equivalence margin (+/-0.05) was fixed before the results.
4. "Seeds" resample prompts from the same templates / MultiBLiMP pool with the same model weights; CIs reflect prompt sampling, not variation across trained models.
5. Agreement uses the single-token pool of MultiBLiMP pairs (the logit difference is then a clean one-token contrast); this selects frequent verbs.
6. The Hindi templates have **not** been reviewed by a native speaker (`data/NATIVE_REVIEW.md`).
7. Llama-3.2-1B weights are from the `unsloth` mirror. **Gemma-3-1B was not run**: TransformerLens 2.15 cannot load it and a newer TL needs torch >= 2.6, unavailable on Ada's CUDA 12.1 setup; it was replaced by BLOOM (Amendment 2, item 1). Hindi results therefore rest mostly on one family.
8. Competence-gate rule changes (corrupted-LD criterion, first-token tier, extra BLOOM models) were made after seeing gate numbers but before any patching result; they are listed in Amendment 2.

## Reproduce (Ada)

```bash
sbatch -A irel --qos=normal --export=ALL,GATE_MODEL=<model> slurm/gate_model.sbatch   # per model
python scripts/merge_gate.py                                                         # results/gate_coverage.*, gated_cells.json
python scripts/queue_init.py --queue-dir ~/mp/queue
sbatch -A irel --qos=normal --array=0-7 slurm/worker.sbatch                          # also: -A research --qos=medium
sbatch -A irel --qos=normal slurm/analyze.sbatch                                     # results/analysis/
```
