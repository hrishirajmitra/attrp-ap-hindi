"""Model / task / language registry for the multi-cell study."""

from __future__ import annotations

from typing import Any

# Models within the ~2B / 11GB budget. Keys are HookedTransformer.from_pretrained names
# where available; HF hub ids noted in comments for fallbacks.
MODELS: dict[str, dict[str, Any]] = {
    "gpt2": {
        "n_layers": 12,
        "n_heads": 12,
        "micro_batch": 50,
        "hf_id": "gpt2",
        "family": "gpt2",
        "n_params_m": 124,
        "expected_langs": ["en"],
        "role": "anchor",
    },
    "EleutherAI/pythia-160m": {
        "n_layers": 12,
        "n_heads": 12,
        "micro_batch": 50,
        "hf_id": "EleutherAI/pythia-160m",
        "family": "pythia",
        "n_params_m": 160,
        "expected_langs": ["en"],
        "role": "en_control",
    },
    "EleutherAI/pythia-410m": {
        "n_layers": 24,
        "n_heads": 16,
        "micro_batch": 25,
        "hf_id": "EleutherAI/pythia-410m",
        "family": "pythia",
        "n_params_m": 410,
        "expected_langs": ["en"],
        "role": "en_control",
    },
    "Qwen/Qwen2.5-0.5B": {
        "n_layers": 24,
        "n_heads": 14,
        "micro_batch": 25,
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "family": "qwen2.5",
        "n_params_m": 500,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "Qwen/Qwen2.5-1.5B": {
        "n_layers": 28,
        "n_heads": 12,
        "micro_batch": 10,
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "family": "qwen2.5",
        "n_params_m": 1500,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "meta-llama/Llama-3.2-1B": {
        "n_layers": 16,
        "n_heads": 32,
        "micro_batch": 10,
        "hf_id": "meta-llama/Llama-3.2-1B",
        "family": "llama3.2",
        "n_params_m": 1000,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
    },
    "google/gemma-3-1b-pt": {
        "n_layers": 26,
        "n_heads": 4,
        "micro_batch": 10,
        "hf_id": "google/gemma-3-1b-pt",
        "family": "gemma3",
        "n_params_m": 1000,
        "expected_langs": ["en", "hi"],
        "role": "multilingual",
        "tl_note": "May need nnsight fallback if TransformerLens lacks support",
    },
    "sarvamai/sarvam-1": {
        "n_layers": 28,
        "n_heads": 16,
        "micro_batch": 5,
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
# agreement_ne is a stretch task, not in the default grid
STRETCH_TASKS = ("agreement_ne",)
LANGS = ("en", "hi")
# Romanised Hindi IOI is a CORE condition (script vs language test); other tasks have no
# romanised variant. It is gated/run for models that are expected to handle Hindi.
CORE_ROMAN_TASKS = ("ioi",)
STRETCH_LANGS = ()
DIRECTIONS = ("denoising", "noising")

# Main tables use core models and core tasks; the rest is the English robustness appendix.
APPENDIX_TASKS = ("induction",)


def is_appendix(model: str, task: str) -> bool:
    return MODELS[model].get("role") == "en_control" or task in APPENDIX_TASKS


def grid_models() -> list[str]:
    """All non-optional models (Sarvam-1 is dropped: no TransformerLens support)."""
    return [m for m, meta in MODELS.items() if not meta.get("optional")]


def gate_cells(model: str) -> list[tuple[str, str]]:
    """(task, lang) cells to gate / run for a model."""
    langs = [l for l in LANGS if l in MODELS[model]["expected_langs"]]
    cells = [(t, l) for t in TASKS for l in langs]
    if "hi" in langs:
        cells += [(t, "hi_roman") for t in CORE_ROMAN_TASKS]
    return cells

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
