#!/usr/bin/env bash
# prefetch_models.sh -- download model repos into $HF_HOME (node-local) with retries.
# Run AFTER sourcing node_bootstrap.sh (needs $VENV, $HF_HOME, $HF_TOKEN).
# Safe to run concurrently on one node (flock) and idempotent (skips cached files).
#
# Prints machine-readable lines:  MODEL_SOURCE <official_id> <actual_repo_id>
set -uo pipefail

: "${VENV:?source slurm/node_bootstrap.sh first}"
: "${HF_HOME:?source slurm/node_bootstrap.sh first}"
SCRATCH="${MP_SCRATCH:-/ssd_scratch/hrishiraj.mitra}"
mkdir -p "$HF_HOME"

exec 8>"$SCRATCH/.prefetch.lock"
flock -w 3600 8 || { echo "ERROR: prefetch lock timeout" >&2; exit 1; }

t0=$SECONDS
"$VENV/bin/python" - <<'PY'
import os, sys, time, json
import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

TOKEN = os.environ.get("HF_TOKEN") or None
CACHE = os.environ.get("HF_HUB_CACHE")
api = HfApi(token=TOKEN)
PLAIN = ["gpt2", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
         "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "bigscience/bloom-560m"]
GATED = {"meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B"}
TOK_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt")
SMALL = ["*.json", "*.txt", "*.model", "*.tiktoken", "merges.txt", "tokenizer*", "vocab*"]


def http_status(repo, fname="config.json"):
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    h = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    for i in range(3):
        try:
            return requests.get(url, headers=h, timeout=30, stream=True).status_code
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return -1


def weight_patterns(repo):
    files = api.list_repo_files(repo)
    st = [f for f in files if f.endswith(".safetensors") and "/" not in f]
    if st:
        w = st
    else:
        w = [f for f in files if f.startswith("pytorch_model") and "/" not in f]
    idx = [f for f in files if f.endswith(".index.json") and "/" not in f]
    has_tok = any(f in files for f in TOK_FILES)
    if not w:
        raise RuntimeError(f"{repo}: no weight files")
    if not has_tok:
        raise RuntimeError(f"{repo}: no tokenizer files")
    return w + idx + SMALL, w


def download(repo, tries=5):
    pats, w = weight_patterns(repo)
    last = None
    for i in range(tries):
        try:
            p = snapshot_download(repo, allow_patterns=pats, token=TOKEN, cache_dir=CACHE,
                                  max_workers=4)
            return p, w
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  retry {i+1}/{tries} for {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            time.sleep(min(60, 5 * 2 ** i))
    raise last


failures = []
def fetch(official, order):
    """order: list of repos to try in order; first usable wins."""
    for repo in order:
        st = http_status(repo)
        print(f"  config.json {repo}: HTTP {st}", flush=True)
        if st in (401, 403, 404):
            continue
        if st != 200:
            # transient: let download() retry
            pass
        try:
            t = time.time()
            path, w = download(repo)
            print(f"  ok {repo} -> {path} weights={w} ({time.time()-t:.0f}s)", flush=True)
            print(f"MODEL_SOURCE {official} {repo}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {repo}: {type(e).__name__}: {str(e)[:300]}", flush=True)
            if isinstance(e, HfHubHTTPError) and getattr(e.response, "status_code", 0) in (401, 403):
                continue
    failures.append(official)
    print(f"MODEL_SOURCE {official} FAILED", flush=True)
    return False


for r in PLAIN:
    print(f"== {r}", flush=True)
    fetch(r, [r])
for official, mirror in GATED.items():
    print(f"== {official} (gated; mirror {mirror})", flush=True)
    fetch(official, [official, mirror])

if failures:
    print("PREFETCH_MODELS_FAILED", failures, flush=True)
    sys.exit(1)
print("PREFETCH_MODELS_DONE", flush=True)
PY
rc=$?
echo "PREFETCH_MODELS_SECONDS $((SECONDS - t0)) rc=$rc host=$(hostname)"
exit $rc
