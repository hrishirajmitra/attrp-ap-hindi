#!/usr/bin/env python3
"""Smoke test on a GPU node: every (task, lang) cell of two models, both directions, tiny n.

Prints a human-checkable line per cell (prompt tail, answer tokens) and fails loudly.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import generate_batch
from src.modelload import load_model
from src.registry import gate_cells
from src.runner import run_cell, validate_result_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=["gpt2", "Qwen/Qwen2.5-0.5B"])
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "smoke")
    args = p.parse_args()

    failures = []
    for name in args.models:
        model = load_model(name, "cuda")
        print(f"\n##### {name}  L={model.cfg.n_layers} H={model.cfg.n_heads}", flush=True)
        for task, lang in gate_cells(name):
            for direction in ("denoising", "noising"):
                tag = f"{name.replace('/', '__')}__{task}__{lang}__{direction}"
                try:
                    if direction == "denoising":
                        b = generate_batch(model, task, lang, n=args.n, seed=0, device="cuda")
                        tk = model.tokenizer
                        print(
                            f"--- {tag}\n    prompt_tail={b.prompts_clean[0][-70:]!r}\n"
                            f"    corr_tail={b.prompts_corrupted[0][-70:]!r}\n"
                            f"    pos={tk.decode([int(b.pos_token_ids[0])])!r} "
                            f"neg={tk.decode([int(b.neg_token_ids[0])])!r} "
                            f"single_tok={b.meta.get('single_token_answer_rate')} "
                            f"seq={b.clean_tokens.shape[1]}",
                            flush=True,
                        )
                    out = args.out / tag
                    run_cell(
                        model, name, task, lang, direction, 0, out, "cuda",
                        n_prompts=args.n, micro_batch=4, n_ig_steps=3, ig_sweep=[3, 5],
                    )
                    assert validate_result_dir(out), "result dir incomplete"
                except Exception as e:  # noqa: BLE001
                    failures.append((tag, f"{type(e).__name__}: {e}"))
                    print(f"!!! FAIL {tag}: {e}\n{traceback.format_exc()[-1200:]}", flush=True)
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    print("\nSMOKE_SUMMARY", json.dumps({"n_failures": len(failures), "failures": failures}, indent=1))
    print("SMOKE_ALL_OK" if not failures else "SMOKE_FAILED")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
