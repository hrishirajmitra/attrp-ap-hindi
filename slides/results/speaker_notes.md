# Speaker notes · results deck (≈10 min)

Rehearse from the notes pane in PowerPoint, or from this page. Aim ~1 minute per main slide.

## 1. Title
Open with the question. One sentence of setup: AP is exact and expensive; AttrP is the cheap shortcut. Preview the answer: heads yes; early MLPs sometimes no; Hindi only where the model can do the task.

## 2. Trade-off
Left = causal gold standard. Right = first-order Taylor. ~49× is measured wall time on our grid (median per-run ratio; mean ~157 s vs ~3.2 s per run). End on: when do they agree?

## 3. What we ran
Four boxes left to right. Emphasise the competence gate: 66 cells gated, 44 runnable (43 strict + 1 first-token). Gemma→BLOOM if asked. 440 runs, 0 failures.

## 4. Gate / Hindi
Point at the red Hindi block: Hindi IOI and romanised IOI fail in all 6 models gated on Hindi; capitals fails in 5 of 6 (one first-token pass, Llama). Failing-cell accuracy 0.27–0.64. Green Hindi = BLOOM agreement (+ induction on BLOOM/Llama). Native review: one place-word fix; re-gate unchanged.

## 5. Heads (H1)
22/24 core cells ρ≥0.90 (teal bars). Two Qwen English-agreement misses, 0.86 and 0.88 (red bars). Threshold was preregistered. Noising mean ρ 0.93.

## 6. Hindi = English (H3)
Three BLOOM agreement pairs. Pooled Hindi−English +0.009, 95% CI [-0.017, +0.040]: equivalent within ±0.05. Say the caveat out loud.

## 7. MLP layer 0 (H2 exploratory)
Chart: only layer 0 disagrees (AP +0.88 vs AttrP −0.46; layers 1–11 within 0.01). MLP r < 0 in only 2/24 core cells (H2 rejected). Same pattern Qwen-0.5B / Pythia IOI. BLOOM IOI MLPs agree (Pearson r ≈ +0.997) — circuit-specific.

## 8. EAP-IG (H5)
GPT-2 IOI: fixes MLP 0 (r −0.93 → +0.95), breaks head ranking (ρ 0.99 → 0.40). Grid: mean head ρ drops 0.36; cost ~3.5× AttrP (5 steps). Not under-integration: mean head ρ 0.59 / 0.58 / 0.59 at 5 / 10 / 20 steps (seed 0).

## 9. Takeaway
Three rules. One breath of limits. Questions.

## Backup
Gate table · noising mean ρ 0.93 · Amendment 2 (gate phase, before any grid result) + H4 observational (Spearman −0.51 with tokens/word, +0.45 with clean LD).
