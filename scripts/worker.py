#!/usr/bin/env python3
"""Queue worker: claims tasks atomically from a shared NFS queue and runs them.

Safe against dead workers (heartbeat + stale reclaim), CUDA OOM (halves the micro-batch and
retries) and duplicate execution (a task is done only when comparison.json validates).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modelload import load_model
from src.registry import MODELS
from src.runner import run_cell, validate_result_dir

STALE_SECONDS = 20 * 60
MAX_ATTEMPTS = 2


class Heartbeat(threading.Thread):
    def __init__(self, path: Path):
        super().__init__(daemon=True)
        self.path = path
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.wait(45):
            try:
                self.path.touch()
            except OSError:
                pass


def try_claim(q: Path, tid: str, wid: str) -> bool:
    cdir = q / "claim" / tid
    if (q / "done" / tid).exists():
        return False
    if cdir.exists():
        hb = cdir / "hb"
        try:
            age = time.time() - hb.stat().st_mtime
        except FileNotFoundError:
            try:
                age = time.time() - cdir.stat().st_mtime
            except FileNotFoundError:
                age = 0
        if age < STALE_SECONDS:
            return False
        try:  # atomic: only one worker wins the rename
            os.rename(cdir, q / "stale" / f"{tid}.{int(time.time())}.{wid}")
        except OSError:
            return False
    try:
        cdir.mkdir()
    except FileExistsError:
        return False
    (cdir / "owner").write_text(f"{wid}\n")
    (cdir / "hb").touch()
    return True


def attempts(q: Path, tid: str) -> int:
    p = q / "failed" / f"{tid}.json"
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text()).get("attempts", 1))
    except Exception:
        return 1


def pick_task(q: Path, tasks: list[dict], wid: str, current_model: str | None):
    done = {p.name for p in (q / "done").iterdir()}
    pending = [t for t in tasks if t["id"] not in done and attempts(q, t["id"]) < MAX_ATTEMPTS]
    if not pending:
        return None, 0
    ordered = [t for t in pending if t["model"] == current_model] + [
        t for t in pending if t["model"] != current_model
    ]
    for t in ordered:
        if try_claim(q, t["id"], wid):
            return t, len(pending)
    return None, len(pending)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "grid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--time-limit-s", type=float, default=7.0 * 3600)
    p.add_argument("--stop-margin-s", type=float, default=35 * 60)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--only-models", nargs="*", default=None)
    args = p.parse_args()

    q = args.queue_dir
    wid = f"{socket.gethostname()}.{os.getpid()}"
    t_start = time.time()
    tasks = json.loads((q / "tasks.json").read_text())["tasks"]
    if args.only_models:
        tasks = [t for t in tasks if t["model"] in args.only_models]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    model = None
    model_name = None
    mb_override: dict[str, int] = {}
    n_done = 0
    print(f"[{wid}] worker start; {len(tasks)} tasks in view", flush=True)

    while True:
        if time.time() - t_start > args.time_limit_s - args.stop_margin_s:
            print(f"[{wid}] time budget reached; exiting", flush=True)
            break
        tasks = json.loads((q / "tasks.json").read_text())["tasks"]  # re-read: queue can grow
        if args.only_models:
            tasks = [t for t in tasks if t["model"] in args.only_models]
        t, n_pending = pick_task(q, tasks, wid, model_name)
        if t is None:
            if n_pending == 0:
                print(f"[{wid}] queue empty; exiting", flush=True)
                break
            time.sleep(30)  # others hold the remaining tasks; wait for stale/finish
            if time.time() - t_start > args.time_limit_s:
                break
            continue

        tid = t["id"]
        out = args.results_dir / tid
        hb = Heartbeat(q / "claim" / tid / "hb")
        hb.start()
        print(f"[{wid}] === {tid} (pending {n_pending})", flush=True)
        ok = False
        err = None
        try:
            if validate_result_dir(out):
                ok = True  # finished earlier (e.g. worker died after writing)
            else:
                if model_name != t["model"]:
                    model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model = load_model(t["model"], args.device)
                    model_name = t["model"]
                mb = mb_override.get(t["model"], MODELS[t["model"]]["micro_batch"])
                sweep = [int(x) for x in t["ig_sweep"].split(",") if x.strip()]
                while True:
                    try:
                        run_cell(
                            model, t["model"], t["task"], t["lang"], t["direction"], t["seed"],
                            out, args.device, n_prompts=args.n_prompts, micro_batch=mb,
                            n_ig_steps=t.get("n_ig_steps", 5), ig_sweep=sweep,
                        )
                        ok = validate_result_dir(out)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if mb <= 1:
                            raise
                        mb = max(1, mb // 2)
                        mb_override[t["model"]] = mb
                        print(f"[{wid}] OOM -> micro_batch {mb}", flush=True)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            print(f"[{wid}] FAILED {tid}: {err}\n{traceback.format_exc()[-1500:]}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, model_name = None, None  # force clean reload after an error
        finally:
            hb.stop_flag.set()

        if ok:
            (q / "done" / tid).write_text(f"{wid} {time.time()}\n")
            n_done += 1
            print(f"[{wid}] DONE {tid} ({n_done} by this worker)", flush=True)
        else:
            n = attempts(q, tid) + 1
            (q / "failed" / f"{tid}.json").write_text(
                json.dumps({"attempts": n, "error": err, "worker": wid, "time": time.time()}) + "\n"
            )
        # release the claim so a retry can pick it up (done marker prevents re-run on success)
        try:
            (q / "claim" / tid / "hb").unlink(missing_ok=True)
            (q / "claim" / tid / "owner").unlink(missing_ok=True)
            (q / "claim" / tid).rmdir()
        except OSError:
            pass

    print(f"[{wid}] finished; {n_done} tasks done", flush=True)


if __name__ == "__main__":
    main()
