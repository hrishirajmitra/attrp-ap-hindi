"""Model / task / language registry for the multi-cell study."""

from __future__ import annotations

from typing import Any

# Models within the ~2B / 11GB budget. Keys are HookedTransformer.from_pretrained names
# where available; HF hub ids noted in comments for fallbacks.
MODELS: dict[str, dict[str, Any]] = {
    "gpt2": {
        "hf_id": "gpt2",
        "family": "gpt2",
        "n_params_m": 124,
        "expected_langs": ["en"],
        "role": "anchor",
    },
    "EleutherAI/pythia-160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "family": "pythia",
        "n_params_m": 160,
        "expected_langs": ["en"],
        "role": "en_control",
    },
    "EleutherAI/pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "family": "pythia",
        "n_params_m": 410,
        "expected_langs": ["en"],
        "role": "en_control",
    },
    "Qwen/Qwen2.5-0.5B": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "family": "qwen2.5",
        "n_params_m": 500,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "Qwen/Qwen2.5-1.5B": {
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "family": "qwen2.5",
        "n_params_m": 1500,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "meta-llama/Llama-3.2-1B": {
        "hf_id": "meta-llama/Llama-3.2-1B",
        "family": "llama3.2",
        "n_params_m": 1000,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "google/gemma-3-1b-pt": {
        "hf_id": "google/gemma-3-1b-pt",
        "family": "gemma3",
        "n_params_m": 1000,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
        "tl_note": "May need nnsight fallback if TransformerLens lacks support",
    },
    "sarvamai/sarvam-1": {
        "hf_id": "sarvamai/sarvam-1",
        "family": "sarvam",
        "n_params_m": 2000,
        "expected_langs": ["en", "hi"],
        "role": "indic",
        "optional": True,
        "tl_note": "Drop first if VRAM tight; TL support uncertain",
    },
}

TASKS = ("ioi", "capitals", "agreement", "induction")
# agreement_ne and hi_roman are stretch (H7); not in the default gate grid
STRETCH_TASKS = ("agreement_ne",)
LANGS = ("en", "hi")
STRETCH_LANGS = ("hi_roman",)
DIRECTIONS = ("denoising", "noising")

# Competence gate thresholds (preregistered in PREREGISTRATION_v2.md)
GATE = {
    "n_prompts": 100,
    "min_clean_accuracy": 0.80,
    "min_clean_logit_diff": 0.5,
    "max_corrupted_logit_diff": 0.25,  # near zero or negative
    "min_single_token_rate": 0.90,
}

# Restricted-head Spearman: a priori (not post-hoc) |AP| threshold
RESTRICTED_AP_ABS_THRESHOLD = 0.05

# Top-k overlap with AP itself (replaces Precision@14 checklist for non-IOI)
TOPK_OVERLAP = (5, 10, 14, 20)


def cell_name(model: str, task: str, lang: str, seed: int, direction: str = "denoising") -> str:
    """Filesystem-safe result directory name."""
    safe = model.replace("/", "__")
    return f"{safe}_{task}_{lang}_{direction}_seed{seed}"


def short_model(model: str) -> str:
    return model.split("/")[-1]
