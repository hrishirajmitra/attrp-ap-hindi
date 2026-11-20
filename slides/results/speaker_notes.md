# Speaker notes · results deck (≈10 min)

Rehearse from the notes pane in PowerPoint, or from this page. Aim about one minute per main slide.
Spoken text is filled from the same SCRIPT that builds the deck; numbers come from deck_data.json.

## 1. Title
Activation patching is the usual way to find which attention heads and which MLP layers matter for a behaviour. It is exact, and it is expensive. Attribution patching is the first-order shortcut people reach for when models get bigger.

I am not proposing a new method. I measured when the cheap method gives the same answer as the expensive one, across tasks, models, and across English and Hindi.

The short version: for attention heads, almost always. For MLP outputs, not always, and the failure is usually one early layer. And for Hindi, I can only test it where the model can actually do the task. That last point turns out to matter as much as the methods themselves.

**If asked**
- Is this a new method? No. I measure when an existing cheap method matches the expensive one, across languages.

## 2. Trade-off
On the left is activation patching. You run the model on a clean prompt and on a corrupted prompt, then copy one component's activation from the clean run into the corrupted run. That tells you how much that component mattered. Exact, and slow.

On the right is attribution patching. It is a Taylor approximation: one forward pass, one backward pass, and every component is ranked at once.

On my grid the median wall-time ratio is about 49, over the 120 core denoising runs. Roughly 157 seconds against 3.2 seconds per run, on a two thousand eighty Ti. That is measured wall time, not a theoretical operation count. The question of the talk is when that cheaper ranking matches the expensive one.

**If asked**
- Why not always use activation patching? At one and a half billion parameters and many cells, it is the bottleneck. The cheap method is what people ship.

## 3. What I ran
I ran nine models under two billion parameters: GPT-2, two Pythias, two Qwens, Llama three point two, and three BLOOM sizes. Four tasks, each in English and Hindi: the indirect-object task, capitals, subject-verb agreement, and induction.

A cell here is one model on one task in one language. Before I patch anything I run a competence gate: clean accuracy at least point eight, and a clear logit difference. If the model cannot do the task, I do not localise it.

Sixty-six cells went through the gate. Forty-four were runnable: forty-three with a full pass, and one that only works on the first token. Then 440 runs: activation patching, attribution patching, and integrated gradients, two directions, five seeds. Zero failures.

One note if it comes up: Gemma three was in the original plan. The tooling could not load it, so BLOOM replaced it, and that was logged before any grid result.

**If asked**
- Why five seeds? They resample prompts, not weights. The confidence intervals reflect prompt sampling.

## 4. Hindi gate
Before I could compare anything, most of the Hindi tasks dropped out. This grid shows why. Each row is a model and each column is a task in one language. Green means the model passed the check.

English is green everywhere. In Hindi, none of the six models I gated can do the indirect-object task, in Devanagari or romanised. Capitals fails in five of six; only Llama gets it, and only on the first token, the orange half circle. In the failing cells accuracy sits between 27 and 64 percent, so there is no behaviour to trace.

What is left in Hindi is verb agreement on the three BLOOM models, plus induction on BLOOM and Llama. That is the entire Hindi evidence for whether the two methods agree.

I checked the Hindi templates myself and fixed one place word. Rerunning the gate changed nothing, so I think this is the models and not my prompts.

**If asked**
- Why so little Hindi? That is the finding. You cannot localise a behaviour the model does not have.

## 5. Heads
Where the gate passes, do the two methods agree on heads? Each bar is one cell. Height is the Spearman correlation between the head rankings from activation patching and from attribution patching. The threshold of 0.90 was fixed in the preregistration before any comparison ran.

22 of 24 cells clear it. The two that miss, in red, are Qwen English agreement, at 0.86 and 0.88. Still high, just under the bar. Everything else sits between point nine one and point nine nine five.

The noising direction averages about 0.93 and tells the same story. So among the cells that pass the Hindi gate, is attribution patching worse in Hindi?

**If asked**
- Is 0.90 magic? No, but it was fixed in the preregistration. The two misses are 0.86 and 0.88; the other 22 cells are point nine one to point nine nine five. Do not lean on the restricted correlation over heads with large activation-patching effect: only a handful of heads pass that cut-off, and it is higher than the plain correlation in just eleven of the twenty-two cells where it is defined.

## 6. Hindi vs English
The only place I can pair English and Hindi on the same model with a full gate pass is BLOOM agreement, three sizes. Induction also pairs on BLOOM and Llama, but that was set aside as appendix by the preregistration.

The bars are almost the same; the axis starts at point eight zero. An equivalence test with a preregistered margin of plus or minus point zero five puts the pooled Hindi minus English difference at plus 0.009. The ninety-five percent interval runs from minus 0.017 to 0.040, inside the margin, so the verdict is equivalent.

I want the caveat clear: three pairs, one model family, one task. I do not claim this for Hindi indirect object identification. Those cells never passed the gate. Heads look fine. What about MLPs?

**If asked**
- Is BLOOM special? It is the only family with usable Hindi agreement. That is a limit, not a virtue.

## 7. MLP layer 0
I predicted that attribution patching would anti-correlate on MLP outputs in most cells. That failed: only two of twenty-four. The GPT-2 failure on the indirect-object task is real, but it is almost entirely layer zero.

From layer one onward the two lines sit on top of each other. Only layer zero flips sign: 0.88 for activation patching, minus 0.46 for attribution patching. Layers one through eleven match within point zero one.

The same pattern shows up in Qwen point five B and in both Pythias on this task. That part is exploratory; I looked after the prediction failed. On BLOOM the MLPs agree almost perfectly, Pearson r about 0.997 or higher, so this is specific to some circuits, not universal.

Can integrated gradients fix layer zero without changing the head ranking?

**If asked**
- Why layer zero? Early nonlinearities; the linear approximation is worst there. I did not claim a mechanism beyond that.

## 8. Integrated gradients
I asked whether integrated gradients repairs MLP agreement. Across the grid, the preregistered answer is no: the mean MLP gain is not distinguishable from zero.

On GPT-2 indirect object identification it does fix layer zero: MLP agreement goes from minus 0.93 to 0.95. But the head ranking no longer matches activation patching. It drops from 0.99 to 0.40. That trade is the point of this slide.

Across the whole grid the mean head correlation drops by 0.36. Integrated gradients also costs about 3.5 times attribution patching, at five steps. More steps do not help: mean head correlation is 0.59, 0.58, 0.59 at five, ten and twenty steps. So this is not under-integration. Use attribution patching for heads; pay for activation patching on early MLPs when it matters.

**If asked**
- Does integrated gradients at one endpoint equal attribution patching? Yes. With a single gradient at alpha equals zero it reduces to attribution patching; I checked that on GPT-2 to one e minus five. The head drop comes from integrating along the path.
- More steps? Single cells shift by up to point zero nine in head correlation between step counts, far less than the 0.36 drop.

## 9. Takeaway
Three rules from this.

Use attribution patching for attention heads. It agreed in 22 of 24 cells, and it is about 49 times cheaper.

Check early MLPs with real activation patching. Layer zero can flip sign.

Gate the task before you localise. Hindi indirect object identification fails in all six models I gated, and capitals in five of six. If the model cannot do the task, there is no circuit worth tracing.

The limits in one breath: the Hindi evidence is narrow, mostly BLOOM agreement. Gemma was dropped for tooling. Seeds resample prompts, not weights. And I reviewed the templates myself.

Thank you. I am happy to take questions.

**If asked**
- What next? A larger Hindi-capable model with single-token agreement, or a hand-hooked Gemma once the tooling catches up.

## Backup
Gate table · noising mean 0.93 · Amendment 2 (gate phase, before any grid result) · fertility observational (Spearman minus 0.51 with tokens/word, 0.45 with clean LD).
