#!/usr/bin/env python3
"""Competence gate over (model, task, lang) cells.

A cell PASSES if, on N prompts (default 100, seed=0):
  - clean accuracy (logit pos > logit neg) >= 0.80
  - mean clean logit-diff >= 0.5
  - mean corrupted logit-diff <= 0.25
  - single-token answer rate >= 0.90

One model is loaded once and all of its cells are evaluated. Per-model results go to
<out-dir>/gate_<model>.json; `scripts/merge_gate.py` merges them into gate_coverage.{json,md}.
Failed cells are reported, never silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import GATE, MODELS, gate_cells, grid_models

try:
    import torch
except ImportError:  # offline / no venv
    torch = None  # type: ignore


def safe(model: str) -> str:
    return model.replace("/", "__")


def evaluate_model(
    model_name: str,
    cells: list[tuple[str, str]],
    device: str,
    n: int,
    seed: int,
    micro_batch: int,
) -> list[dict]:
    from src.data import generate_batch
    from src.metrics import accuracy_from_logits, logit_diff_from_logits
    from src.modelload import load_model, model_source
    from src.runner import prompt_descriptors

    rows: list[dict] = []
    try:
        model = load_model(model_name, device)
        source = model_source(model_name, model)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"  MODEL LOAD FAILED {model_name}: {err}", flush=True)
        for task, lang in cells:
            rows.append(
                {
                    "model": model_name, "task": task, "lang": lang, "n": n, "seed": seed,
                    "pass": False, "error": err, "reasons": [f"model_load: {err}"],
                    "traceback": traceback.format_exc()[-1500:],
                }
            )
        return rows

    for task, lang in cells:
        row = {
            "model": model_name, "task": task, "lang": lang, "n": n, "seed": seed,
            "source_repo": source, "pass": False, "error": None, "reasons": [],
        }
        try:
            batch = generate_batch(model, task=task, lang=lang, n=n, seed=seed, device=device)
            cl, co, pos_l, neg_l = [], [], [], []
            with torch.no_grad():
                nn_ = int(batch.clean_tokens.shape[0])
                clean_logits_acc, ld_c, ld_o = [], [], []
                for s in range(0, nn_, micro_batch):
                    e = min(nn_, s + micro_batch)
                    lc = model(batch.clean_tokens[s:e])
                    lo = model(batch.corrupted_tokens[s:e])
                    p, q = batch.pos_token_ids[s:e], batch.neg_token_ids[s:e]
                    ld_c.append(logit_diff_from_logits(lc, p, q))
                    ld_o.append(logit_diff_from_logits(lo, p, q))
                    del lc, lo
                ld_c = torch.cat(ld_c)
                ld_o = torch.cat(ld_o)
            clean_ld = float(ld_c.mean().item())
            corr_ld = float(ld_o.mean().item())
            clean_acc = float((ld_c > 0).float().mean().item())
            single_rate = float(batch.meta.get("single_token_answer_rate", 0.0))
            desc = prompt_descriptors(model, batch)
            row.update(
                {
                    "clean_logit_diff": clean_ld,
                    "corrupted_logit_diff": corr_ld,
                    "clean_accuracy": clean_acc,
                    "single_token_answer_rate": single_rate,
                    "batch_meta": batch.meta,
                    "descriptors": desc,
                    "seq_len": int(batch.clean_tokens.shape[1]),
                    "n_prompts_used": nn_,
                }
            )
            reasons = []
            if clean_acc < GATE["min_clean_accuracy"]:
                reasons.append(f"clean_acc {clean_acc:.3f} < {GATE['min_clean_accuracy']}")
            if clean_ld < GATE["min_clean_logit_diff"]:
                reasons.append(f"clean_ld {clean_ld:.3f} < {GATE['min_clean_logit_diff']}")
            if corr_ld > GATE["max_corrupted_logit_diff"]:
                reasons.append(f"corr_ld {corr_ld:.3f} > {GATE['max_corrupted_logit_diff']}")
            if single_rate < GATE["min_single_token_rate"]:
                reasons.append(
                    f"single_tok_rate {single_rate:.3f} < {GATE['min_single_token_rate']}"
                )
            row["reasons"] = reasons
            row["pass"] = len(reasons) == 0
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-2000:]
            row["reasons"] = [f"exception: {row['error']}"]
            row["pass"] = False
        print(
            f"  gate {model_name} {task}/{lang}: pass={row['pass']} "
            f"acc={row.get('clean_accuracy')} cld={row.get('clean_logit_diff')} "
            f"old={row.get('corrupted_logit_diff')} st={row.get('single_token_answer_rate')} "
            f"{row['reasons']}",
            flush=True,
        )
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--n-prompts", type=int, default=GATE["n_prompts"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--micro-batch", type=int, default=25)
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "gate")
    args = p.parse_args()

    if torch is None:
        raise SystemExit("torch is required to run the gate (run on a node with the venv)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = args.models or grid_models()
    for model in models:
        if model not in MODELS:
            raise SystemExit(f"unknown model {model}")
        cells = gate_cells(model)
        if args.tasks:
            cells = [c for c in cells if c[0] in args.tasks]
        if args.langs:
            cells = [c for c in cells if c[1] in args.langs]
        print(f"Gating {model}: {len(cells)} cells", flush=True)
        rows = evaluate_model(model, cells, args.device, args.n_prompts, args.seed, args.micro_batch)
        payload = {"model": model, "thresholds": {**GATE, "n_prompts": args.n_prompts}, "rows": rows}
        (args.out_dir / f"gate_{safe(model)}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.out_dir / f'gate_{safe(model)}.json'}", flush=True)


if __name__ == "__main__":
    main()
