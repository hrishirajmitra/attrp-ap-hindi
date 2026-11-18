"""Run one (model, task, lang, direction, seed) cell end to end and write results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from scipy.stats import pearsonr, spearmanr
from transformer_lens import HookedTransformer

from .activation_patch import activation_patch_heads_and_mlps
from .attribution_patch import attribution_patch_heads_and_mlps
from .compare import compare_all, save_results
from .data import generate_batch
from .eap_ig import eap_ig_heads_and_mlps
from .modelload import load_model  # noqa: F401  (re-exported)


def prompt_descriptors(model: HookedTransformer, batch) -> dict[str, Any]:
    """Cell descriptors used in the H4 analysis: tokenizer fertility and prompt length."""
    tok_counts, word_counts = [], []
    for p in batch.prompts_clean:
        ids = model.to_tokens(p, prepend_bos=False)[0]
        tok_counts.append(int(ids.shape[0]))
        word_counts.append(max(1, len(p.split())))
    n = max(1, len(tok_counts))
    return {
        "tokens_per_word": float(sum(tok_counts) / max(1, sum(word_counts))),
        "mean_prompt_tokens": float(sum(tok_counts) / n),
        "mean_prompt_words": float(sum(word_counts) / n),
        "seq_len": int(batch.clean_tokens.shape[1]),
    }


def run_cell(
    model: HookedTransformer,
    model_name: str,
    task: str,
    lang: str,
    direction: str,
    seed: int,
    out: Path,
    device: str,
    n_prompts: int = 100,
    micro_batch: int | None = None,
    n_ig_steps: int = 5,
    ig_sweep: list[int] | None = None,
    skip_eap: bool = False,
    agreement_source: str = "multiblimp",
    corruption: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    batch = generate_batch(
        model,
        task=task,
        lang=lang,
        n=n_prompts,
        seed=seed,
        device=device,
        corruption=corruption,
        agreement_source=agreement_source,
    )
    desc = prompt_descriptors(model, batch)
    print(
        f"task={task} lang={lang} dir={direction} n={batch.clean_tokens.shape[0]} "
        f"seq_len={desc['seq_len']} tpw={desc['tokens_per_word']:.2f} "
        f"single_tok_rate={batch.meta.get('single_token_answer_rate')} mb={micro_batch}",
        flush=True,
    )

    ap = activation_patch_heads_and_mlps(
        model, batch, direction=direction, micro_batch=micro_batch, show_progress=show_progress
    )
    print(
        f"  AP  clean_ld={ap['clean_logit_diff']:.3f} corr_ld={ap['corrupted_logit_diff']:.3f} "
        f"wall={ap['wall_seconds']:.1f}s fast={ap['fast_mode']}",
        flush=True,
    )
    attr = attribution_patch_heads_and_mlps(
        model, batch, direction=direction, micro_batch=micro_batch
    )
    print(f"  AttrP wall={attr['wall_seconds']:.1f}s", flush=True)

    eap = None
    eap_sweep: dict[int, dict] = {}
    steps_list = sorted(set(ig_sweep or [])) or ([] if skip_eap else [n_ig_steps])
    for steps in steps_list:
        res = eap_ig_heads_and_mlps(
            model, batch, n_steps=steps, direction=direction, micro_batch=micro_batch
        )
        print(f"  EAP-IG steps={steps} wall={res['wall_seconds']:.1f}s", flush=True)
        eap_sweep[steps] = res
        if steps == n_ig_steps or eap is None:
            eap = res
    if skip_eap:
        eap, eap_sweep = None, {}

    comparison = compare_all(ap, attr, eap, task=task)
    comparison["cell"] = {
        "model": model_name,
        "task": task,
        "lang": lang,
        "seed": seed,
        "direction": direction,
        "corruption": batch.corruption,
        "batch_meta": batch.meta,
        "descriptors": desc,
        "micro_batch": micro_batch,
        "ap_fast_mode": ap.get("fast_mode"),
    }
    if len(eap_sweep) > 1:
        comparison["eap_sweep"] = {}
        for s, res in eap_sweep.items():
            rho_s, _ = spearmanr(ap["head_scores"].ravel(), res["head_scores"].ravel())
            r_m, _ = pearsonr(ap["mlp_scores"], res["mlp_scores"])
            comparison["eap_sweep"][str(s)] = {
                "spearman_ap_eap_ig": float(rho_s),
                "mlp_pearson_r": float(r_m),
                "wall_seconds": res["wall_seconds"],
            }

    save_results(out, ap, attr, comparison, eap, eap_sweep=eap_sweep or None)
    print(
        f"  rho(AP,AttrP)={comparison['spearman_ap_attr']:.3f} "
        f"top14={comparison['topk_overlap_ap_attr']['14']:.2f} "
        f"mlp_r={comparison['mlp_pearson_ap_attr']['pearson_r']:.2f} "
        f"verdict={comparison['verdict']['primary']}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return comparison


def validate_result_dir(out: Path) -> bool:
    """A cell is complete only if comparison.json parses and contains the key fields."""
    p = out / "comparison.json"
    if not p.exists() or not (out / "scores.npz").exists():
        return False
    try:
        d = json.loads(p.read_text())
        return "spearman_ap_attr" in d and "verdict" in d and "cell" in d
    except Exception:
        return False
