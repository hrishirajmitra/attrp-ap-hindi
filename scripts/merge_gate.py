#!/usr/bin/env python3
"""Merge per-model gate files into results/gate_coverage.{json,md} and lock the cell list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import GATE, MODELS, gate_cells, grid_models, is_appendix


def fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate-dir", type=Path, default=ROOT / "results" / "gate")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--directions", nargs="*", default=["denoising", "noising"])
    p.add_argument("--sweep", default="5,10,20")
    args = p.parse_args()

    rows = []
    for f in sorted(args.gate_dir.glob("gate_*.json")):
        rows += json.loads(f.read_text())["rows"]
    have = {(r["model"], r["task"], r["lang"]) for r in rows}
    missing = [
        (m, t, l) for m in grid_models() for (t, l) in gate_cells(m) if (m, t, l) not in have
    ]
    if missing:
        print(f"WARNING: {len(missing)} expected cells have no gate result: {missing[:6]}...")

    passed = [r for r in rows if r.get("tier") in ("strict", "first_token")]
    payload = {
        "thresholds": GATE,
        "rows": rows,
        "passed_cells": [
            {"model": r["model"], "task": r["task"], "lang": r["lang"], "tier": r["tier"]}
            for r in passed
        ],
        "missing_cells": [{"model": m, "task": t, "lang": l} for m, t, l in missing],
        "offline": False,
    }
    (args.out_dir / "gate_coverage.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Competence gate coverage",
        "",
        f"Thresholds: clean_acc>={GATE['min_clean_accuracy']}, clean_ld>={GATE['min_clean_logit_diff']}, "
        f"corr_ld<=max({GATE['max_corrupted_logit_diff']}, {GATE['max_corrupted_fraction']}*clean_ld), single_tok>={GATE['min_single_token_rate']}, "
        f"N={GATE['n_prompts']}.",
        "",
        "| Model | Task | Lang | Tier | Clean acc | Clean LD | Corr LD | Single-tok | Tok/word | Reasons |",
        "|---|---|:---:|:---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        tpw = (r.get("descriptors") or {}).get("tokens_per_word")
        lines.append(
            f"| `{r['model']}` | {r['task']} | {r['lang']} | {r.get('tier')} | {fmt(r.get('clean_accuracy'))} | "
            f"{fmt(r.get('clean_logit_diff'))} | {fmt(r.get('corrupted_logit_diff'))} | "
            f"{fmt(r.get('single_token_answer_rate'))} | {fmt(tpw)} | "
            f"{'; '.join(r.get('reasons') or []) or 'ok'} |"
        )
    n_strict = sum(1 for r in rows if r.get("tier") == "strict")
    lines += ["", f"Strict: {n_strict}; first_token tier: {len(passed) - n_strict}; failed: {len(rows) - len(passed)}; total {len(rows)}", ""]
    (args.out_dir / "gate_coverage.md").write_text("\n".join(lines))

    # Task list for the queue: one task per (cell, direction, seed); sweep on seed 0 denoising.
    tasks = []
    for c in payload["passed_cells"]:
        for d in args.directions:
            for s in args.seeds:
                tasks.append(
                    {
                        **c,
                        "direction": d,
                        "seed": s,
                        "ig_sweep": args.sweep if (s == 0 and d == "denoising") else "",
                        "n_ig_steps": 5,
                        "appendix": is_appendix(c["model"], c["task"]),
                    }
                )
    (args.out_dir / "gated_cells.json").write_text(
        json.dumps(
            {"n_jobs": len(tasks), "source_gate": "gate_coverage.json", "used_expected": False, "jobs": tasks},
            indent=2,
        )
        + "\n"
    )
    print(f"{len(passed)}/{len(rows)} cells passed -> {len(tasks)} queue tasks")


if __name__ == "__main__":
    main()
