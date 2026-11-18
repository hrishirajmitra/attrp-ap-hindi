# Papers for: attribution patching vs activation patching on IOI (GPT-2 small)

Every entry below was opened and its title and authors checked against the arXiv page (or the original blog post) on 2026-10-21. The "why it matters" lines are tied to specific claims I read in the source. Anything marked **(my inference)** is my connection to our results, not a claim made by the paper.

## Tier 1: cite these (they define your question, baseline, and comparison)

| # | Paper | Link | Why it matters |
|---|---|---|---|
| 1 | Wang, Variengien, Conmy, Shlegeris, Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small* (2022) | [arXiv 2211.00593](https://arxiv.org/abs/2211.00593) | Defines the IOI task, the logit-difference metric, and the head inventory (name movers, S-inhibition, induction, duplicate-token, previous-token). It also names backup name movers (9.0, 10.1, 10.10, 10.6) and negative name movers. **(my inference)** Our activation-patching top 14 includes L9H0, L10H1, L10H10, L10H6. These match the backup name movers, so our literature checklist H* was incomplete. |
| 2 | Nanda. *Attribution Patching: Activation Patching At Industrial Scale* (blog, Feb 2023) | [neelnanda.io](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) | Origin of attribution patching: a gradient-based linear approximation to activation patching. The accompanying notebook applies it to IOI in GPT-2 small, which is our exact setup. This is the method we test. |
| 3 | Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery* (2023) | [arXiv 2310.10348](https://arxiv.org/abs/2310.10348) | Formalises edge attribution patching (EAP). Section 5.1 reports a **weak** attribution vs activation correlation (R² = 0.27, slope 0.53) on the Docstring task, and explains it by a concave metric curve. This is a tension with our ρ ≈ 0.98 that you need to discuss. **(my inference)** Likely explanations are differences in task, component granularity (edges vs nodes) and statistic (rank correlation vs R²). I did not test this. |
| 4 | Hanna, Pezzelle, Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms* (2024) | [arXiv 2403.17806](https://arxiv.org/abs/2403.17806) | Introduces EAP-IG: edge attribution patching with integrated gradients, motivated by the flat-derivative problem at the clean activation. This is the origin of our refinement arm. Ours is a node-level, input-embedding variant, **not** their edge-level method. |
| 5 | Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks* (2017) | [arXiv 1703.01365](https://arxiv.org/abs/1703.01365) | Original integrated gradients. Needed to justify the EAP-IG arm. |
| 6 | Kramár, Lieberum, Shah, Nanda. *AtP\*: An efficient and scalable method for localizing LLM behaviour to components* (2024) | [arXiv 2403.00745](https://arxiv.org/abs/2403.00745) | The most direct prior work on where attribution patching fails. It finds two failure classes that produce false negatives: saturated attention softmax (gradients are a poor approximation) and brittle cancellation of positive and negative effects. **(my inference)** Our flat/near-zero heads could hide such cases. We patch head outputs, not query/key, so we did not test the attention-saturation failure. |

## Tier 2: methodology (cite for choices you made, or should defend)

| # | Paper | Link | Why it matters |
|---|---|---|---|
| 7 | Heimersheim, Nanda. *How to use and interpret activation patching* (2024) | [arXiv 2404.15255](https://arxiv.org/abs/2404.15255) | Defines denoising (clean into corrupted) vs noising (corrupted into clean). We patch clean into corrupted (denoising). This tests sufficiency, so it mostly surfaces late heads. **(my inference)** This may be part of why early heads were not recovered. Untested. |
| 8 | Zhang, Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (2023) | [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) | Compares corruption methods (Gaussian noising vs symmetric token replacement) and metrics (probability vs logit difference). Justifies the logit-diff metric and motivates reporting more than one corruption, as we did. |
| 9 | Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC, 2023) | [arXiv 2304.14997](https://arxiv.org/abs/2304.14997) | Source of the IOI circuit evaluation framing and of the head list many later papers use as ground truth. |
| 10 | Miller, Chughtai, Saunders. *Transformer Circuit Faithfulness Metrics Are Not Robust* (2024) | [arXiv 2407.08734](https://arxiv.org/abs/2407.08734) | Cautions that faithfulness metrics depend on methodological choices. Supports your limits section: our verdict depends on corruption, metric and checklist. |

## Tier 3: optional background

| # | Paper | Link | Why it matters |
|---|---|---|---|
| 11 | Goldowsky-Dill, MacLeod, Sato, Arora. *Localizing Model Behavior With Path Patching* (2023) | [arXiv 2304.05969](https://arxiv.org/abs/2304.05969) | The path-patching tool used to find the IOI heads. Relevant to the point that single-component patching and path-level analysis find different things. |
| 12 | Meng, Bau, Andonian, Belinkov. *Locating and Editing Factual Associations in GPT* (2022) | [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) | Causal tracing, an early large-scale use of activation patching. |
| 13 | Hanna, Liu, Variengien. *How does GPT-2 compute greater-than?* (2023) | [arXiv 2305.00586](https://arxiv.org/abs/2305.00586) | Second GPT-2 small circuit task. Natural extension if you want to test whether results generalise beyond IOI. |

## Reading order

1. Wang et al. (1): skim sections 1 to 3 and the backup name mover discussion.
2. Nanda blog (2) and Syed et al. (3): the method and its known weakness.
3. Kramár et al. (6): the failure modes.
4. Hanna et al. (4) and Sundararajan et al. (5): the EAP-IG arm.
5. Heimersheim and Nanda (7), then Zhang and Nanda (8): defend your design choices.

## Not covered

- I did not open the TransformerLens documentation or the Nanda attribution-patching notebook itself, only the blog post. Check the notebook before claiming to reproduce its numbers.
- I did not search for 2025 to 2026 follow-up work on attribution patching. That is a gap if your course expects recent literature.
