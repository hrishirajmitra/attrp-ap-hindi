#!/bin/bash
# Usage (account/qos come from the command line):
#   sbatch -A irel     --qos=normal ~/mp/slurm/prefetch.sbatch
#   sbatch -A research --qos=medium ~/mp/slurm/prefetch.sbatch
# Optionally pin: add --nodelist=gnodeXXX
#SBATCH -p u22
#SBATCH -J mp-prefetch
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err

set -uo pipefail
mkdir -p "$HOME/mp-attrpatch-logs"
echo "JOB $SLURM_JOB_ID host=$(hostname) account=${SLURM_JOB_ACCOUNT:-?} qos=${SLURM_JOB_QOS:-?} start=$(date -Iseconds)"
nvidia-smi -L

# shellcheck disable=SC1091
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo "BOOTSTRAP_FAILED $(hostname)"; exit 1; }

bash "$HOME/mp/slurm/prefetch_models.sh" || { echo "PREFETCH_FAILED $(hostname)"; exit 1; }

cat > "${TMPDIR:-/tmp}/smoke_$SLURM_JOB_ID.py" <<'PY'
import time, torch
from transformer_lens import HookedTransformer
assert torch.cuda.is_available()
for name in ["gpt2", "Qwen/Qwen2.5-0.5B"]:
    t = time.time()
    m = HookedTransformer.from_pretrained(name, device="cuda")
    toks = m.to_tokens("The quick brown fox jumps over the lazy dog")
    with torch.no_grad():
        logits = m(toks)
    assert torch.isfinite(logits).all()
    print(f"SMOKE_OK {name} logits={tuple(logits.shape)} dtype={logits.dtype} {time.time()-t:.1f}s", flush=True)
    del m, logits
    torch.cuda.empty_cache()
PY

# Prove the cache is coherent (offline first); fall back to online only with a warning.
if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" "${TMPDIR:-/tmp}/smoke_$SLURM_JOB_ID.py"; then
  echo "SMOKE_MODE offline-cache"
elif "$VENV/bin/python" "${TMPDIR:-/tmp}/smoke_$SLURM_JOB_ID.py"; then
  echo "WARNING: smoke test needed network (cache incomplete for TL)"
  echo "SMOKE_MODE online"
else
  echo "SMOKE_FAILED $(hostname)"; exit 1
fi
rm -f "${TMPDIR:-/tmp}/smoke_$SLURM_JOB_ID.py"
echo "PREFETCH_OK $(hostname)"
