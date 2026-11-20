# Speaker notes · results deck (≈10 min live)

Spoken text is cut for a ~10 minute live talk. Aim under a minute per main slide; leave "If asked" for questions.
Spoken text is filled from the same SCRIPT that builds the deck; numbers come from deck_data.json.

## 1. Title
Activation patching finds which attention heads and MLP layers matter for a behaviour. Exact, and expensive. Attribution patching is the cheap first-order shortcut people use when models get bigger.

I measured when the cheap method matches the expensive one, across tasks, models, and English and Hindi. Short answer: for heads, almost always. For MLPs, not always, and the failure is usually one early layer. Hindi only where the model can do the task.

**If asked**
- Is this a new method? No. I measure when an existing cheap method matches the expensive one, across languages.

## 2. Trade-off
On the left, activation patching: copy one component from a clean run into a corrupted run. Exact, slow. On the right, attribution patching: a Taylor approximation, one forward and one backward, every component ranked at once.

On this grid the median wall-time ratio is about 49: roughly 157 seconds against 3.2 seconds per run on a 2080 Ti. When do they agree?

**If asked**
- Why not always use activation patching? At one and a half billion parameters and many cells, it is the bottleneck.
- The 49 is median wall time over 120 core denoising runs, not a FLOP count.

## 3. What I ran
Nine models under two billion parameters: GPT-2, Pythia, Qwen, Llama, BLOOM. Four tasks in English and Hindi: indirect object, capitals, agreement, induction. A cell is one model on one task in one language.

Before I patch, I gate on competence: clean accuracy at least point eight, and a clear logit difference. Sixty-six cells gated, forty-four runnable. Then 440 runs, five seeds, zero failures.

**If asked**
- Why five seeds? They resample prompts, not weights.
- Gemma three was in the proposal; tooling could not load it, so BLOOM replaced it before any grid result.

## 4. Hindi gate
Before I could compare anything, most Hindi tasks dropped out. English is green everywhere. Hindi indirect-object fails in all six models I gated, Devanagari and romanised. Capitals fails in five of six; only Llama gets it, and only on the first token. Failing accuracy is 27 to 64 percent, so there is nothing to localise.

What remains is BLOOM agreement, plus induction on BLOOM and Llama. I reviewed the Hindi templates myself; one place-word fix, re-gate unchanged.

**If asked**
- Why so little Hindi? That is the finding. You cannot localise a behaviour the model does not have.

## 5. Heads
Where the gate passes, do the methods agree on heads? Each bar is the Spearman correlation of head rankings. The threshold of 0.90 was preregistered.

22 of 24 cells clear it. The two misses, in red, are Qwen English agreement at 0.86 and 0.88. Still high, just under the bar. Noising averages about 0.93. Is attribution worse in Hindi?

**If asked**
- Is 0.90 magic? No; it was fixed in the preregistration. Do not lean on restricted correlation over large-effect heads: higher than plain correlation in only eleven of twenty-two cells.

## 6. Hindi vs English
The only full-gate English–Hindi pair is BLOOM agreement, three sizes. The bars are almost the same. Equivalence within plus or minus point zero five: pooled difference plus 0.009, ninety-five percent CI from minus 0.017 to 0.040.

Caveat out loud: three pairs, one family, one task. I do not claim this for Hindi IOI. Heads look fine. What about MLPs?

**If asked**
- Is BLOOM special? It is the only family with usable Hindi agreement. That is a limit, not a virtue.

## 7. MLP layer 0
I predicted MLP anti-correlation in most cells. Rejected: only two of twenty-four. The GPT-2 IOI failure is real, and it is almost entirely layer zero: activation 0.88, attribution minus 0.46. From layer one onward the lines sit on top of each other.

Same pattern in Qwen point five B and both Pythias; that part is exploratory. BLOOM MLPs agree. Can integrated gradients fix layer zero without changing the head ranking?

**If asked**
- Why layer zero? Early nonlinearities; the linear approximation is worst there.

## 8. Integrated gradients
On GPT-2 IOI, integrated gradients fixes layer zero: MLP agreement goes from minus 0.93 to 0.95. But head ranking drops from 0.99 to 0.40. That trade is the point.

Across the grid, mean head correlation drops by 0.36, and the MLP gain is indistinguishable from zero. Cost about 3.5 times attribution patching. More steps do not help.

**If asked**
- Alpha zero reduces to attribution patching; checked on GPT-2. The head drop is from integrating along the path.
- Step sweep on seed 0: mean head correlation 0.59, 0.58, 0.59 at five, ten, twenty steps.

## 9. Takeaway
Three rules. Use attribution patching for heads: 22 of 24, about 49 times cheaper. Check early MLPs with real activation patching; layer zero can flip sign. Gate the task before you localise.

Limits: Hindi evidence is narrow, mostly BLOOM agreement. Gemma was dropped for tooling. Seeds resample prompts. I reviewed the templates myself. Thank you.

**If asked**
- What next? A larger Hindi-capable model with single-token agreement, or hand-hooked Gemma once tooling catches up.

## Backup
Gate table · noising mean 0.93 · Amendment 2 (gate phase, before any grid result) · fertility observational (Spearman minus 0.51 with tokens/word, 0.45 with clean LD).
