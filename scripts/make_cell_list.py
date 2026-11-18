#!/usr/bin/env python3
"""Expand gated cells into a flat job list (model, task, lang, direction, seed, ig_sweep).

Reads results/gate_coverage.json (passed_cells, or expected_pass_cells with --allow-expected).
Writes results/gated_cells.json used by slurm/run_grid.sbatch array index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path, default=ROOT / "results" / "gate_coverage.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "gated_cells.json")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--directions", nargs="*", default=["denoising"])
    p.add_argument(
        "--noising-seed0-only",
        action="store_true",
        default=True,
        help="If noising in directions, only seed 0 (compute-saving default)",
    )
    p.add_argument("--ig-sweep-seed0", action="store_true", default=True)
    p.add_argument("--allow-expected", action="store_true",
                   help="Use expected_pass_cells when passed_cells empty (planning only)")
    args = p.parse_args()

    gate = json.loads(args.gate.read_text())
    cells = gate.get("passed_cells") or []
    if not cells and args.allow_expected:
        cells = gate.get("expected_pass_cells") or []
        print(f"WARNING: using {len(cells)} expected_pass cells (not locked)")
    if not cells:
        raise SystemExit("No passed_cells in gate file. Run gate on Ada or pass --allow-expected.")

    jobs = []
    for cell in cells:
        for direction in args.directions:
            seeds = args.seeds
            if direction == "noising" and args.noising_seed0_only:
                seeds = [0]
            for seed in seeds:
                ig_sweep = ""
                if args.ig_sweep_seed0 and seed == 0 and direction == "denoising":
                    ig_sweep = "5,10,20"
                jobs.append(
                    {
                        "model": cell["model"],
                        "task": cell["task"],
                        "lang": cell["lang"],
                        "direction": direction,
                        "seed": seed,
                        "ig_sweep": ig_sweep,
                        "n_ig_steps": 5,
                    }
                )

    payload = {
        "n_jobs": len(jobs),
        "source_gate": str(args.gate),
        "used_expected": bool(not gate.get("passed_cells") and args.allow_expected),
        "jobs": jobs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(jobs)} jobs → {args.out}")


if __name__ == "__main__":
    main()
