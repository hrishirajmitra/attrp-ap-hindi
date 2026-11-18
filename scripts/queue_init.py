#!/usr/bin/env python3
"""Build the shared task queue from results/gated_cells.json (longest tasks first)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import MODELS


def task_id(j: dict) -> str:
    safe = j["model"].replace("/", "__")
    return f"{safe}__{j['task']}__{j['lang']}__{j['direction']}__seed{j['seed']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=Path, default=ROOT / "results" / "gated_cells.json")
    p.add_argument("--gate", type=Path, default=ROOT / "results" / "gate_coverage.json")
    p.add_argument("--queue-dir", type=Path, required=True)
    p.add_argument("--reset", action="store_true", help="clear claims/done markers")
    args = p.parse_args()

    jobs = json.loads(args.cells.read_text())["jobs"]
    seq = {}
    for r in json.loads(args.gate.read_text())["rows"]:
        seq[(r["model"], r["task"], r["lang"])] = (r.get("descriptors") or {}).get("seq_len", 40)

    tasks = []
    for j in jobs:
        m = MODELS[j["model"]]
        passes = m["n_layers"] * (m["n_heads"] + 1)
        sl = seq.get((j["model"], j["task"], j["lang"]), 40)
        # relative cost: patched passes * params * tokens, + sweep extra for EAP-IG
        cost = passes * m["n_params_m"] * sl * (1.15 if j.get("ig_sweep") else 1.0)
        tasks.append({**j, "id": task_id(j), "cost": cost})
    tasks.sort(key=lambda t: -t["cost"])

    q = args.queue_dir
    for sub in ("claim", "done", "failed", "stale"):
        (q / sub).mkdir(parents=True, exist_ok=True)
    if args.reset:
        import shutil

        for sub in ("claim", "done", "failed", "stale"):
            shutil.rmtree(q / sub, ignore_errors=True)
            (q / sub).mkdir(parents=True, exist_ok=True)
    (q / "tasks.json").write_text(json.dumps({"n": len(tasks), "tasks": tasks}, indent=1) + "\n")
    print(f"queue {q}: {len(tasks)} tasks; total relative cost {sum(t['cost'] for t in tasks):.3e}")


if __name__ == "__main__":
    main()
