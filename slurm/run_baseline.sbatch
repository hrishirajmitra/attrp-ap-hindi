#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-ap-base
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /home2/%u/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/%u/mp-attrpatch-logs/%x-%j.err

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"

module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
# shellcheck disable=SC1091
source "$HOME/mp/env.sh"

if [[ -n "${MP_COMPUTE_NODE:-}" && "$(hostname)" != "$MP_COMPUTE_NODE" ]]; then
  echo "ERROR: need --nodelist=$MP_COMPUTE_NODE" >&2
  exit 2
fi

cd "$HOME/mp"
OUT="$HOME/mp/results/baseline"
"$VENV/bin/python" scripts/run_baseline.py \
  --model gpt2 \
  --n-prompts 100 \
  --seed 0 \
  --device cuda \
  --out "$OUT"

echo "Baseline done → $OUT"
