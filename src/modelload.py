"""Model loading with a mirror fallback for licence-gated repos.

`meta-llama/Llama-3.2-1B` and `google/gemma-3-1b-pt` need an accepted licence. When the token
is refused we load byte-identical weights from the ungated `unsloth/*` mirrors and hand the
loaded HF model + tokenizer to TransformerLens under the official architecture name.
Everything runs in float32: gradients are needed and fp16 can overflow.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

MIRRORS = {
    "meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B",
}


def _freeze(model: HookedTransformer) -> HookedTransformer:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_model(name: str, device: str, dtype: torch.dtype = torch.float32) -> HookedTransformer:
    print(f"Loading {name} on {device} ({dtype})...", flush=True)
    try:
        model = HookedTransformer.from_pretrained(name, device=device, dtype=dtype)
    except Exception as e:  # gated repo / 401 / 403
        mirror = MIRRORS.get(name)
        if mirror is None:
            raise
        print(f"  official repo failed ({type(e).__name__}); using mirror {mirror}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = AutoModelForCausalLM.from_pretrained(mirror, torch_dtype=dtype)
        tok = AutoTokenizer.from_pretrained(mirror)
        model = HookedTransformer.from_pretrained(
            name, hf_model=hf, tokenizer=tok, device=device, dtype=dtype
        )
        del hf
        model.cfg.mirror_of = mirror  # recorded for provenance
    return _freeze(model)


def model_source(name: str, model: HookedTransformer) -> str:
    return getattr(model.cfg, "mirror_of", None) or name
