#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-validate
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"
module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
module load u22/python/3.12.15 2>/dev/null || true
unset PYTHONHOME
# shellcheck disable=SC1091
source "$HOME/mp/env.sh"
cd "$HOME/mp"

# Optional: install EAP-IG for cross-check (non-fatal)
"$VENV/bin/pip" install --cache-dir "${MP_PIPCACHE:-$HOME/pipcache/mp-attrpatch}" \
  "git+https://github.com/hannamw/EAP-IG.git" || echo "EAP-IG install failed; continuing with ours"

"$VENV/bin/python" scripts/validate_engine.py --device cuda --n-prompts 20 \
  --try-eap-lib --out results/engine_check
cat results/engine_check/validate_engine.json
