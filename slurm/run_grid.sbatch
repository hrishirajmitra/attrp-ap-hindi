#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-grid
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%A_%a.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%A_%a.err
#SBATCH --array=0-0
#
# After gate:
#   N=$(python3 -c "import json; print(json.load(open('$HOME/mp/results/gated_cells.json'))['n_jobs'])")
#   NODE=$(cat ~/mp/.compute_node)
#   sbatch --nodelist=$NODE --array=0-$((N-1)) ~/mp/slurm/run_grid.sbatch

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"

module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
module load u22/python/3.12.15 2>/dev/null || true
unset PYTHONHOME
# shellcheck disable=SC1091
source "$HOME/mp/env.sh"

if [[ -n "${MP_COMPUTE_NODE:-}" && "$(hostname)" != "$MP_COMPUTE_NODE" ]]; then
  echo "ERROR: wrong node $(hostname); need $MP_COMPUTE_NODE" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: missing $VENV/bin/python" >&2
  exit 2
fi

CELLS="$HOME/mp/results/gated_cells.json"
if [[ ! -f "$CELLS" ]]; then
  echo "ERROR: missing $CELLS — run run_gate.sbatch then make_cell_list.py" >&2
  exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
read -r MODEL TASK LANG DIRECTION SEED IG_SWEEP N_IG < <(
  "$VENV/bin/python" - <<PY
import json
jobs = json.load(open("$CELLS"))["jobs"]
j = jobs[$IDX]
print(j["model"], j["task"], j["lang"], j["direction"], j["seed"], j.get("ig_sweep") or "", j.get("n_ig_steps", 5))
PY
)

echo "Job $IDX: model=$MODEL task=$TASK lang=$LANG dir=$DIRECTION seed=$SEED ig_sweep=$IG_SWEEP"
cd "$HOME/mp"

SAFE_MODEL="${MODEL//\//__}"
OUT="$HOME/mp/results/${SAFE_MODEL}_${TASK}_${LANG}_${DIRECTION}_seed${SEED}"

EXTRA=()
if [[ -n "$IG_SWEEP" ]]; then
  EXTRA+=(--ig-sweep "$IG_SWEEP")
fi

"$VENV/bin/python" scripts/run_all.py \
  --model "$MODEL" \
  --task "$TASK" \
  --lang "$LANG" \
  --direction "$DIRECTION" \
  --seed "$SEED" \
  --n-prompts 100 \
  --device cuda \
  --n-ig-steps "$N_IG" \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "Done → $OUT"
cat "$OUT/comparison.json" | head -c 2000
echo
