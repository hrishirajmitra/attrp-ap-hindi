#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-attrpatch
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
#
# After setup_env.sbatch, re-submit with:
#   NODE=$(cat ~/mp/.compute_node)
#   sbatch --nodelist=$NODE ~/mp/slurm/run_all.sbatch
# so the /ssd_scratch venv is present.

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"

module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
# Load python module for libpython, then unset PYTHONHOME (env.sh also sets LD_LIBRARY_PATH)
module load u22/python/3.12.15 2>/dev/null || true
unset PYTHONHOME

# shellcheck disable=SC1091
source "$HOME/mp/env.sh"

echo "Host=$(hostname) expected_node=${MP_COMPUTE_NODE:-?} VENV=$VENV"
if [[ -n "${MP_COMPUTE_NODE:-}" && "$(hostname)" != "$MP_COMPUTE_NODE" ]]; then
  echo "ERROR: scheduled on $(hostname) but venv lives on $MP_COMPUTE_NODE. Resubmit with --nodelist=$MP_COMPUTE_NODE" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: missing $VENV/bin/python — run setup_env.sbatch first" >&2
  exit 2
fi

nvidia-smi -L
cd "$HOME/mp"
CORR="${CORR:-abc}"
SEED="${SEED:-0}"
OUT="$HOME/mp/results/full_${CORR}_seed${SEED}"
# Legacy GPT-2 IOI job. Prefer slurm/run_grid.sbatch for the multi-cell study.
"$VENV/bin/python" scripts/run_all.py \
  --model gpt2 \
  --task ioi \
  --lang en \
  --n-prompts 100 \
  --seed "$SEED" \
  --corruption "$CORR" \
  --direction denoising \
  --device cuda \
  --n-ig-steps 5 \
  --out "$OUT"

"$VENV/bin/python" scripts/write_report.py --results "$OUT" --out "$OUT/WRITEUP.md"
# Also copy writeup to repo root for convenience
rm -f "$OUT/WRITEUP.md"

# Mirror small results to login /share1 if mounted
if [[ -d /share1/hrishiraj.mitra ]]; then
  mkdir -p /share1/hrishiraj.mitra/mp-attrpatch/results
  cp -a "$OUT" /share1/hrishiraj.mitra/mp-attrpatch/results/ || true
fi

echo "Full run done → $OUT"
cat "$OUT/comparison.json"
