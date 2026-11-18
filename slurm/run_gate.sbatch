#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-gate
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
#
# NODE=$(cat ~/mp/.compute_node)
# sbatch --nodelist=$NODE ~/mp/slurm/run_gate.sbatch

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

cd "$HOME/mp"
"$VENV/bin/python" scripts/tokenizer_audit.py --device cuda --out-dir "$HOME/mp/results"
"$VENV/bin/python" scripts/gate.py --device cuda --out-dir "$HOME/mp/results"
"$VENV/bin/python" scripts/make_cell_list.py --gate "$HOME/mp/results/gate_coverage.json" \
  --out "$HOME/mp/results/gated_cells.json"

echo "Gate done. Passed cells:"
python3 -c "import json; d=json.load(open('$HOME/mp/results/gate_coverage.json')); print(len(d['passed_cells'])); print(d['passed_cells'])"
