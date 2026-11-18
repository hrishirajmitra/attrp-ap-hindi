#!/usr/bin/env python3
"""Competence gate over (model, task, lang) cells.

A cell PASSES if, on N prompts (default 100, seed=0):
  - clean accuracy (logit pos > logit neg) >= 0.80
  - mean clean logit-diff >= 0.5
  - mean corrupted logit-diff <= 0.25
  - single-token answer rate >= 0.90

Writes results/gate_coverage.json and results/gate_coverage.md.
Failed cells are reported, not silently dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import GATE, LANGS, MODELS, TASKS

try:
    import torch
except ImportError:  # offline / no venv
    torch = None  # type: ignore


def evaluate_cell(
    model_name: str,
    task: str,
    lang: str,
    device: str,
    n: int,
    seed: int,
) -> dict:
    row = {
        "model": model_name,
        "task": task,
        "lang": lang,
        "n": n,
        "seed": seed,
        "pass": False,
        "error": None,
        "reasons": [],
    }
    try:
        if torch is None:
            raise RuntimeError("torch not installed; use --offline")
        from transformer_lens import HookedTransformer

        from src.data import generate_batch
        from src.metrics import accuracy_from_logits, logit_diff_from_logits

        try:
            model = HookedTransformer.from_pretrained(model_name, device=device)
        except Exception:
            model = HookedTransformer.from_pretrained(
                model_name, device=device, dtype=torch.float16
            )
        model.eval()
        batch = generate_batch(
            model, task=task, lang=lang, n=n, seed=seed, device=device
        )
        with torch.no_grad():
            clean_logits = model(batch.clean_tokens)
            corr_logits = model(batch.corrupted_tokens)
            clean_ld = float(
                logit_diff_from_logits(
                    clean_logits, batch.pos_token_ids, batch.neg_token_ids
                )
                .mean()
                .item()
            )
            corr_ld = float(
                logit_diff_from_logits(
                    corr_logits, batch.pos_token_ids, batch.neg_token_ids
                )
                .mean()
                .item()
            )
            clean_acc = accuracy_from_logits(
                clean_logits, batch.pos_token_ids, batch.neg_token_ids
            )
        single_rate = float(batch.meta.get("single_token_answer_rate", 0.0))
        row.update(
            {
                "clean_logit_diff": clean_ld,
                "corrupted_logit_diff": corr_ld,
                "clean_accuracy": clean_acc,
                "single_token_answer_rate": single_rate,
                "batch_meta": batch.meta,
                "seq_len": int(batch.clean_tokens.shape[1]),
            }
        )
        reasons = []
        if clean_acc < GATE["min_clean_accuracy"]:
            reasons.append(
                f"clean_acc {clean_acc:.3f} < {GATE['min_clean_accuracy']}"
            )
        if clean_ld < GATE["min_clean_logit_diff"]:
            reasons.append(
                f"clean_ld {clean_ld:.3f} < {GATE['min_clean_logit_diff']}"
            )
        if corr_ld > GATE["max_corrupted_logit_diff"]:
            reasons.append(
                f"corr_ld {corr_ld:.3f} > {GATE['max_corrupted_logit_diff']}"
            )
        if single_rate < GATE["min_single_token_rate"]:
            reasons.append(
                f"single_tok_rate {single_rate:.3f} < {GATE['min_single_token_rate']}"
            )
        row["reasons"] = reasons
        row["pass"] = len(reasons) == 0
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-2000:]
        row["reasons"] = [f"exception: {row['error']}"]
        row["pass"] = False
    return row


def expected_offline_coverage() -> list[dict]:
    """Prior-expectation coverage when models cannot be loaded (local laptop).

    These are NOT gate results — marked expected_pass. Real gate must be run on Ada.
    """
    rows = []
    # Heuristic expectations for planning / PREREG draft before Ada gate run
    expect = {
        # (family_role pattern via model meta)
        ("gpt2", "ioi", "en"): True,
        ("gpt2", "ioi", "hi"): False,
        ("gpt2", "capitals", "en"): True,
        ("gpt2", "capitals", "hi"): False,
        ("gpt2", "agreement", "en"): True,
        ("gpt2", "agreement", "hi"): False,
        ("gpt2", "induction", "en"): True,
        ("gpt2", "induction", "hi"): False,
        ("pythia", "ioi", "en"): True,
        ("pythia", "ioi", "hi"): False,
        ("pythia", "capitals", "en"): True,
        ("pythia", "capitals", "hi"): False,
        ("pythia", "agreement", "en"): True,
        ("pythia", "agreement", "hi"): False,
        ("pythia", "induction", "en"): True,
        ("pythia", "induction", "hi"): False,
        ("qwen2.5", "ioi", "en"): True,
        ("qwen2.5", "ioi", "hi"): True,
        ("qwen2.5", "capitals", "en"): True,
        ("qwen2.5", "capitals", "hi"): True,
        ("qwen2.5", "agreement", "en"): True,
        ("qwen2.5", "agreement", "hi"): True,
        ("qwen2.5", "induction", "en"): True,
        ("qwen2.5", "induction", "hi"): True,
        ("llama3.2", "ioi", "en"): True,
        ("llama3.2", "ioi", "hi"): True,
        ("llama3.2", "capitals", "en"): True,
        ("llama3.2", "capitals", "hi"): True,
        ("llama3.2", "agreement", "en"): True,
        ("llama3.2", "agreement", "hi"): True,
        ("llama3.2", "induction", "en"): True,
        ("llama3.2", "induction", "hi"): True,
        ("gemma3", "ioi", "en"): True,
        ("gemma3", "ioi", "hi"): True,
        ("gemma3", "capitals", "en"): True,
        ("gemma3", "capitals", "hi"): True,
        ("gemma3", "agreement", "en"): True,
        ("gemma3", "agreement", "hi"): True,
        ("gemma3", "induction", "en"): True,
        ("gemma3", "induction", "hi"): True,
        ("sarvam", "ioi", "en"): True,
        ("sarvam", "ioi", "hi"): True,
        ("sarvam", "capitals", "en"): True,
        ("sarvam", "capitals", "hi"): True,
        ("sarvam", "agreement", "en"): True,
        ("sarvam", "agreement", "hi"): True,
        ("sarvam", "induction", "en"): True,
        ("sarvam", "induction", "hi"): True,
    }
    for model, meta in MODELS.items():
        fam = meta["family"]
        for task in TASKS:
            for lang in LANGS:
                # English-only families: Hindi expected fail
                key = (fam, task, lang)
                exp = expect.get(key, False)
                if lang == "hi" and "hi" not in meta["expected_langs"]:
                    exp = False
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "lang": lang,
                        "pass": None,
                        "expected_pass": exp,
                        "offline_note": True,
                        "reasons": [
                            "offline expectation only — run scripts/gate.py on Ada to lock"
                        ],
                    }
                )
    return rows


def to_markdown(rows: list[dict], thresholds: dict) -> str:
    lines = [
        "# Competence gate coverage",
        "",
        f"Thresholds: clean_acc≥{thresholds['min_clean_accuracy']}, "
        f"clean_ld≥{thresholds['min_clean_logit_diff']}, "
        f"corr_ld≤{thresholds['max_corrupted_logit_diff']}, "
        f"single_tok≥{thresholds['min_single_token_rate']}, N={thresholds['n_prompts']}.",
        "",
        "| Model | Task | Lang | Pass | Clean acc | Clean LD | Corr LD | Single-tok | Reasons |",
        "|---|---|:---:|:---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("offline_note"):
            lines.append(
                f"| `{r['model']}` | {r['task']} | {r['lang']} | expected={r.get('expected_pass')} | "
                f"— | — | — | — | {'; '.join(r.get('reasons', []))} |"
            )
            continue
        lines.append(
            f"| `{r['model']}` | {r['task']} | {r['lang']} | {r.get('pass')} | "
            f"{r.get('clean_accuracy', '') if r.get('clean_accuracy') is not None else ''} | "
            f"{_fmt(r.get('clean_logit_diff'))} | {_fmt(r.get('corrupted_logit_diff'))} | "
            f"{_fmt(r.get('single_token_answer_rate'))} | {'; '.join(r.get('reasons') or []) or 'ok'} |"
        )
    passed = [r for r in rows if r.get("pass") is True]
    expected = [r for r in rows if r.get("expected_pass") is True]
    lines += [
        "",
        f"Passed cells: {len(passed)} / {len(rows)}",
        f"Expected (offline) pass cells: {len(expected)} / {len(rows)}",
        "",
    ]
    return "\n".join(lines)


def _fmt(x) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--n-prompts", type=int, default=GATE["n_prompts"])
    p.add_argument("--seed", type=int, default=0)
    default_device = "cpu"
    if torch is not None and torch.cuda.is_available():
        default_device = "cuda"
    p.add_argument("--device", default=default_device)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = args.models or list(MODELS.keys())
    tasks = args.tasks or list(TASKS)
    langs = args.langs or list(LANGS)

    if args.offline or torch is None:
        if not args.offline and torch is None:
            print("torch missing — writing offline expected coverage")
        args.offline = True

    if args.offline:
        rows = expected_offline_coverage()
        # Filter to requested models/tasks/langs
        rows = [
            r
            for r in rows
            if r["model"] in models and r["task"] in tasks and r["lang"] in langs
        ]
    else:
        rows = []
        for model in models:
            for task in tasks:
                for lang in langs:
                    print(f"Gating {model} / {task} / {lang} ...")
                    rows.append(
                        evaluate_cell(
                            model, task, lang, args.device, args.n_prompts, args.seed
                        )
                    )
                    print(
                        f"  pass={rows[-1]['pass']} reasons={rows[-1].get('reasons')}"
                    )

    payload = {
        "thresholds": {**GATE, "n_prompts": args.n_prompts},
        "rows": rows,
        "passed_cells": [
            {"model": r["model"], "task": r["task"], "lang": r["lang"]}
            for r in rows
            if r.get("pass") is True
        ],
        "expected_pass_cells": [
            {"model": r["model"], "task": r["task"], "lang": r["lang"]}
            for r in rows
            if r.get("expected_pass") is True
        ],
        "offline": bool(args.offline),
    }
    (args.out_dir / "gate_coverage.json").write_text(json.dumps(payload, indent=2) + "\n")
    md = to_markdown(rows, payload["thresholds"])
    (args.out_dir / "gate_coverage.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
