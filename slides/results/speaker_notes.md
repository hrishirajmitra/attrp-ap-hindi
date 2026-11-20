# Speaker notes · results deck (13 minute talk)

The spoken text is sized for a 13 minute talk, about a minute and a half per slide. "If asked" items are for questions, not for the talk.
Spoken text is filled from the same SCRIPT that builds the deck; numbers come from deck_data.json.

## 1. Title
If you want to know which parts of a language model are responsible for a behaviour, the standard tool is activation patching. It is a causal test and it is exact, but it is expensive. So in practice people reach for attribution patching, a gradient-based shortcut that is far cheaper.

My question is simple: when can the shortcut stand in for the real thing? I am not proposing a new method. I measured how often the cheap method gives the same answer as the expensive one, across four tasks, nine models, and two languages, English and Hindi.

Here is where this ends up. For attention heads, the two methods agree almost everywhere. For MLP layers they sometimes disagree, and when they do, it comes from a single early layer. And for Hindi, I could only ask the question where the model can actually do the task, which turned out to be a result in its own right.

**If asked**
- Is this a new method? No. I measure when an existing cheap method matches the expensive one, across languages.

## 2. Trade-off
Let me make the two methods concrete. Take a prompt the model gets right, the clean prompt, and a minimally changed version where that answer no longer follows, the corrupted prompt.

In activation patching, on the left, I run the corrupted prompt, and I copy in the activation of one component, say one attention head, from the clean run. If the right answer comes back, that head matters. Then I repeat that for every head and every MLP layer, which is why it is slow.

Attribution patching, on the right, replaces all of those runs with a first-order approximation. One forward pass and one backward pass, and I get an estimated effect for every component at once.

On my grid the difference is about 49 times. A typical run takes about 157 seconds with activation patching and about 3.2 seconds with attribution patching, on the same GPU. A gap that size is why people use the shortcut, so what I want to know is whether its answers can be trusted.

**If asked**
- Why not always use activation patching? At one and a half billion parameters and many cells, it is the bottleneck.
- The 49 is the median per-run wall-time ratio over 120 core denoising runs on a 2080 Ti, not a FLOP count.

## 3. What I ran
Nine models, all under two billion parameters: GPT-2 and two Pythia models, which are English only, then two Qwen models, Llama 3.2, and three sizes of BLOOM, which I also tested in Hindi.

Four tasks, each in English and in Hindi. The first is indirect object identification. The prompt is something like: After Mary and John went to the store, John gave a book to. And the answer is Mary. The others are country capitals, subject-verb agreement, and induction, where the model has to continue a repeated pattern.

I will call one model on one task in one language a cell. Before patching any cell, I check that the model can actually do the task. This is the competence gate: at least 80 percent accuracy on clean prompts, and a clear gap between the score of the right answer and the wrong one. If a model cannot do the task, there is no behaviour to explain, so I do not patch it.

66 cells went through the gate and 44 were usable. On those I ran activation patching, attribution patching, and integrated gradients, a third method I will come back to. Each in two patching directions, with five seeds. That is 440 runs, and all of them completed. The hypotheses and the thresholds were fixed before the grid was run.

**If asked**
- Why five seeds? They resample prompts, not weights. The intervals reflect prompt sampling.
- Of the 44 usable cells, 43 pass fully and 1 passes only on the first token of the answer.
- Gemma 3 was in the proposal; the library could not load it, so BLOOM replaced it before any grid result.

## 4. Hindi gate
The gate produced the first result, and it is about Hindi. Each row is a model, and each column is a task in one language. Green means the model passed, and grey means I did not test it.

The left half, English, is green everywhere. The right half is mostly red. None of the six models can do the Hindi indirect-object task, in Devanagari or in romanised Hindi. Hindi capitals fails in five of the six. Llama is the exception, and it passes only on the first token of the answer, which is the orange half circle. In the red cells of those columns, the models pick the right answer between 27 and 64 percent of the time, so there is no reliable behaviour to trace.

What survives in Hindi is subject-verb agreement on the three BLOOM models, and induction on BLOOM and Llama. That is all the Hindi evidence I have for the main question.

Could this be my prompts and not the models? I went through the Hindi templates myself and found one error, a place word in the wrong form. I fixed it and ran the gate again, and every Hindi indirect-object cell still fails. So I read this as a limit of these small models, not of the prompts.

**If asked**
- Why so little Hindi? That is the finding. You cannot localise a behaviour the model does not have.
- Why do Qwen and Llama fail Hindi agreement? The right and wrong verb forms start with the same token, so there is no clean single-token contrast.

## 5. Heads
Now the main question, on the cells that passed. For each cell, both methods give every attention head a score, and I ask whether they put the heads in the same order. The measure is the Spearman rank correlation, where 1 means an identical ranking. Each bar is one cell, averaged over five seeds. Note that the axis starts at 0.80.

The main analysis covers 24 cells; induction and the Pythia models are kept as side checks. Before running anything, I set the bar at 0.90. 22 of the 24 cells clear it, and 18 of them are above 0.95. The two that miss, in red, are the two Qwen models on English agreement, at 0.86 and 0.88, so they are close.

The other patching direction averages about 0.93, with the same picture.

So for attention heads, the shortcut gives you essentially the ranking the expensive method would. Three of these bars are Hindi, and they look like the rest. The next slide tests that properly.

**If asked**
- Is 0.90 magic? No, but it was fixed in the preregistration. Do not lean on the restricted correlation over large-effect heads: only a handful of heads pass that cut-off, and it is higher than the plain correlation in just eleven of the twenty-two cells where it is defined.
- The other direction is noising: patching the corrupted activation into the clean run.

## 6. Hindi vs English
Is the shortcut any worse in Hindi than in English? To compare fairly I need the same model, on the same task, passing the gate in both languages. That leaves subject-verb agreement on the three BLOOM models.

The teal bars are English and the orange bars are Hindi. For the two smaller models English is very slightly higher, and for the largest one Hindi is higher. I ran an equivalence test with a margin of 0.05 that I fixed in advance. The pooled Hindi minus English difference is plus 0.009, and the 95 percent interval runs from minus 0.017 to 0.040, which is inside the margin. So on this evidence, the two languages are equivalent.

I want to be clear about how narrow this is. It is three pairs, from one model family, on one task. It says nothing about Hindi indirect object identification, because no model could do that task.

So far this has all been about attention heads. MLP layers are a different story.

**If asked**
- Is BLOOM special? It is the only family with usable Hindi agreement. That is a limit, not a virtue.
- Induction also pairs English and Hindi, on BLOOM and Llama. It is a side check by the preregistration; Hindi is within 0.02 of English there too.

## 7. MLP layer 0
On GPT-2's indirect-object task, attribution patching gives MLP scores that are anti-correlated with the true ones. I predicted that this would happen in most cells. It does not. It happens in only two of the 24: GPT-2 and the small Qwen model, both on the indirect-object task.

So I looked at where the disagreement comes from. This chart is GPT-2. The horizontal axis is the MLP layer, and the vertical axis is the effect of patching that layer. From layer 1 to layer 11 the two lines sit on top of each other, within 0.01. The whole disagreement is layer 0. Activation patching says layer 0 has a large positive effect, 0.88. Attribution patching says minus 0.46. It gets the sign wrong.

I see the same thing in the small Qwen model and in both Pythia models on this task. I only looked at this after my prediction failed, so treat it as exploratory. And it is not universal: on the BLOOM models, the MLP scores agree almost perfectly on the same task.

The obvious next question is whether a better approximation fixes it.

**If asked**
- Why layer zero? Probably early nonlinearities, where the linear approximation is worst. I did not test a mechanism.
- BLOOM indirect-object MLPs: Pearson r between the two methods is about 0.997 or higher in all three sizes.

## 8. Integrated gradients
That better approximation is integrated gradients. Instead of taking a single gradient, it averages the gradient at several points on the path between the corrupted input and the clean one. The version I used is called EAP-IG.

On GPT-2 it does repair layer 0. On the left, agreement on MLP layers goes from minus 0.93 to plus 0.95. But look at the right. On the same model and task, agreement on attention heads falls from 0.99 to 0.40. It fixes one thing and loses the other.

And that is not just GPT-2. Across all 24 cells, head agreement drops by 0.36 on average, and the average gain on MLPs is not distinguishable from zero. It also costs about 3.5 times as much as plain attribution patching. I checked whether it simply needed more integration steps. At 5, 10 and 20 steps the average head agreement is 0.59, 0.58, 0.59, so no.

One caution on reading this. Integrated gradients measures something slightly different from patching one component at a time, so lower agreement does not prove it is wrong. But if what you want is the activation patching answer, it is not a substitute.

**If asked**
- Is the head drop a bug? With a single gradient at alpha zero, EAP-IG reduces exactly to attribution patching; I checked that on GPT-2. The drop comes from integrating along the path.
- More steps? The sweep is seed 0 over the 24 core cells. Single cells move by up to 0.09 in head correlation between step counts, far less than the 0.36 drop.
- The preregistered question was whether EAP-IG repairs MLP agreement across the grid. The answer is no: the interval on the mean gain crosses zero.

## 9. Takeaway
So what would I do in practice? Three things.

First, for attention heads, use attribution patching. It agreed with activation patching in 22 of 24 cells, and it is about 49 times cheaper.

Second, do not trust it on early MLP layers. Run real activation patching on those, because layer 0 can come out with the wrong sign.

Third, check that the model can do the task before you localise anything. In Hindi, that check removed most of what I had planned to study.

Now the limits. The Hindi evidence is thin: one task on one model family, plus induction. Gemma was in my original plan and had to be replaced by BLOOM, because the library could not load it. The five seeds resample prompts, not model weights. And the Hindi templates were checked only by me.

Thank you. I am happy to take questions.

**If asked**
- What next? A larger Hindi-capable model with a single-token agreement contrast, or a hand-hooked Gemma once the tooling catches up.

## Backup
Gate table · noising mean 0.93 · Amendment 2 (gate phase, before any grid result) · fertility observational (Spearman minus 0.51 with tokens/word, 0.45 with clean LD).

# Appendix (for questions; not part of the 13 minutes)

## 13. activation patching
Every task is built as pairs. The clean prompt is one the model gets right; the corrupted prompt has the same template and the same token length, but the cue that decides the answer is changed, so the answer no longer follows.

The metric is the logit difference at the last token: the logit of the correct answer minus the logit of the wrong one.

In the denoising direction I run the corrupted prompt, overwrite one component's output with its value from the clean run, and measure again. I divide the change by the full clean-minus-corrupted gap, so 1 means that component alone restores the behaviour and 0 means it does nothing. Noising is the mirror image: run the clean prompt and put in the corrupted value.

One forward pass per component per batch is what makes it slow. To save time, each patched pass starts at the patched layer from a cached residual stream, and every run checks that shortcut against the full forward pass.

**If asked**
- Why the ABC corruption for IOI? It is the standard one from Wang et al.: replacing all three names removes the information about who the indirect object is, without changing the template.
- Why equal token length? So clean and corrupted activations line up position by position and no padding is needed.

## 14. attribution patching
Attribution patching replaces the per-component re-run with a first-order Taylor expansion of the metric around the corrupted run. If one component's activation moves from its corrupted value to its clean value, the change in the metric is approximately that difference dotted with the gradient of the metric with respect to the activation.

So I need three things: the clean activations, from one cached forward pass; the corrupted activations and the gradient at every head and MLP output, from one forward and one backward pass; and a dot product per component. The result is divided by the same clean-minus-corrupted gap as activation patching, so the two scores are on the same scale.

The approximation is good when the metric is close to linear along the change. It is known to be weak when a patch pushes through a strong nonlinearity, such as a saturated attention softmax, or when the activation change is large, as in early layers. The layer-zero result in this study fits that.

**If asked**
- Is any training involved? No. The weights are frozen; the gradient is taken with respect to activations only.
- Does micro-batching change the numbers? No. The metric is a mean over prompts, so summing the per-chunk contributions gives exactly the single-batch value.

## 15. integrated gradients
Integrated gradients replaces the single gradient at the corrupted point with an average of gradients along a straight path of inputs from corrupted to clean. Here the path is in the token embeddings: at step alpha the input is the corrupted embedding plus alpha times the difference. I take five midpoints along the path, average the gradient at each component, and dot it with the same clean-minus-corrupted activation difference.

With a single step at alpha equals zero, this is exactly attribution patching. I checked that on GPT-2 and the difference was zero.

Why can it disagree with activation patching even when it is a better approximation of something? Along the path every component moves toward clean at once. Activation patching moves one component and holds the rest at their corrupted values. So integrated gradients answers a slightly different question.

This is node-level, on head and MLP outputs, not the edge-level method of Hanna and colleagues.

**If asked**
- Why five steps? The pilot used five; the five, ten, twenty sweep on seed 0 shows the grid means do not move.

## 16. measuring agreement
For heads I use Spearman rank correlation over all heads, because what you use these scores for is choosing which components to look at, and a rank statistic does not care about scale.

For MLPs I use Pearson correlation over layers, as in the GPT-2 pilot. There are only twelve to twenty-eight layers per model, and the question is whether the sign flips.

Two secondary checks: the rank correlation restricted to heads with an activation-patching effect above 0.05, and the overlap of the top fourteen heads. Fourteen is the size of the known GPT-2 indirect-object circuit.

Each cell has five seeds, and a seed resamples the hundred prompts. Intervals are bootstrap intervals over seeds. For Hindi against English I used an equivalence test, two one-sided tests, because the question is whether the two are the same, and a non-significant difference alone does not show that.

**If asked**
- Why not top-k precision against a known circuit? Known circuits exist only for GPT-2 IOI, so for the other cells I compare with activation patching itself.

## 17. gate thresholds
The gate exists because agreement between two attribution methods means nothing if the model is not doing the task. Each threshold guards against one specific way a cell could be meaningless.

Accuracy of at least 0.80: chance is 0.50 for a two-way choice, so this asks for a reliable behaviour, not an occasional one.

Clean logit difference of at least 0.5: every score is divided by the clean-minus-corrupted gap, and a tiny gap turns the scores into noise.

Corrupted logit difference of at most 0.25, or at most half the clean value: the corruption has to actually remove the behaviour. The half rule was added in Amendment 2, because agreement corruptions leave a residual grammatical preference, and the absolute cap alone rejected every agreement cell.

Single-token answers in at least 90 percent of prompts: the metric reads one token per answer. Cells that fail only this are kept as a first-token tier and used only as a sensitivity check.

**If asked**
- Why 100 prompts? Same as the GPT-2 pilot. At 80 percent accuracy, 100 prompts give a 95 percent interval of about plus or minus 0.08.
- Were thresholds changed after seeing results? The corrupted-gap rule and the first-token tier were changed after the first gate numbers, before any comparison of methods. They are in Amendment 2.

## 18. other design choices
The head threshold of 0.90 is a judgement call, not something derived from data. The GPT-2 pilot used 0.80 and reached about 0.99, so for the grid I set a stricter bar, and fixed it before the grid ran.

The equivalence margin of 0.05 is the same size as the pilot's bar for a meaningful improvement in rank correlation, and it was also fixed in advance.

The restricted cut-off of 0.05 was set in advance to stop many near-zero heads from inflating the correlation; in the pilot, 86 to 88 percent of heads had effects below 0.02.

Denoising is the primary direction, as in the pilot; noising is the secondary check. Everything runs in 32-bit floats on models up to two billion parameters, which is what fits on one eleven-gigabyte 2080 Ti, and avoids half-precision rounding in the gradients.

BLOOM replaced Gemma 3 because the library could not load Gemma, and two more BLOOM sizes were added because the first gate left only one usable Hindi cell. Both changes came before any comparison of methods.

**If asked**
- Is 0.90 cherry-picked? It was fixed before the grid. With 0.80, as in the pilot, every core cell would pass; the two Qwen misses are 0.86 and 0.88.
