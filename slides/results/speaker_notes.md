# Speaker notes · results deck (≈10 min)

Rehearse from the notes pane in PowerPoint, or from this page. Aim ~1 minute per main slide.

## 1. Title
Open with the question. One sentence of setup: AP is exact and expensive; AttrP is the cheap shortcut. Preview the answer: heads yes; early MLPs sometimes no; Hindi only where the model can do the task.

## 2. Trade-off
Left = causal gold standard. Right = first-order Taylor. ~49× is measured wall time on our grid. End on: when do they agree?

## 3. What we ran
Four boxes left to right. Emphasise the competence gate. Gemma→BLOOM if asked. 440 runs, 0 failures.

## 4. Gate / Hindi
Point at the red Hindi block. Green Hindi = BLOOM agreement (+ induction). Native review: one place-word fix; re-gate unchanged.

## 5. Heads (H1)
22/24 cells ρ≥0.90. Two Qwen agreement misses. Threshold was preregistered.

## 6. Hindi = English (H3)
Three BLOOM pairs. Equivalent within ±0.05. Say the caveat out loud.

## 7. MLP layer 0 (H2 exploratory)
Chart: only layer 0 disagrees. Same pattern Qwen-0.5B / Pythia IOI. BLOOM IOI MLPs agree — circuit-specific.

## 8. EAP-IG (H5)
Fixes MLP 0, breaks head ranking. ~3.4× AttrP. Not under-integration (5/10/20 agree).

## 9. Takeaway
Three rules. One breath of limits. Questions.

## Backup
Gate table · noising mean ρ · Amendment 2 + H4 observational.
