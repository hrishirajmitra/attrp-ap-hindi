# Engine decision (env-check)

**Date:** 2026-10-27  
**Question:** Use [hannamw/EAP-IG](https://github.com/hannamw/EAP-IG) as the patching engine, or keep ours and cross-check?

## Findings

| Item | Result |
|------|--------|
| Package | `eap` 2.0.0; `pip install .` from the repo |
| Deps | `transformer-lens>=2.15.0` (compatible with our pin `>=2,<4`) |
| Supported models (README) | GPT-2, Pythia, Qwen, Llama, Gemma (Gemma needs a residual-stream workaround); GQA needs `ungroup_grouped_query_attention=True` |
| API | `Graph.from_model` → `attribute(..., method='EAP'\|'EAP-IG-inputs'\|...)` → node/edge scores; also exact eval via `evaluate_graph` |
| Overlap with ours | Node-level EAP ≈ our AttrP; EAP-IG-inputs ≈ our `eap_ig.py`; exact circuit eval is extra |

## Decision

**Keep our AP / AttrP / EAP-IG as the primary engine.** Reasons:

1. We already have locked GPT-2 IOI results and a TaskBatch/gate pipeline wired to our score layout (`[L,H]` heads + `[L]` MLPs).
2. EAP-IG is edge-oriented by default; adapting every cell to Graph/edge scores would rewrite stage-1 metrics and break v1 comparability.
3. Local machine has no torch; Ada still needs a one-line install check. Until that passes on our pinned venv, falling back to ours is required anyway.

**Use hannamw/EAP-IG as an optional cross-check** on a small subset (GPT-2 IOI seed 0; one multilingual cell) via `scripts/validate_engine.py` and `src/eap_lib_adapter.py`. Install on Ada with:

```bash
pip install --cache-dir "$MP_PIPCACHE" "git+https://github.com/hannamw/EAP-IG.git"
# or: cd /tmp && git clone ... && pip install ./EAP-IG
```

If the adapter's AttrP head ranking Spearman vs our AttrP is ≥ 0.95 on GPT-2 IOI, treat engines as interchangeable for ranking; report any gap.

## Ada checklist (run once)

```bash
source ~/mp/env.sh
"$VENV/bin/python" -c "import eap; import transformer_lens; print('eap ok', transformer_lens.__version__)"
"$VENV/bin/python" scripts/validate_engine.py --device cuda --n-prompts 20 --out results/engine_check
```
