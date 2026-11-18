#!/usr/bin/env bash
# Sequence Ada jobs for stage 1 + stage 2. Run from login after setup_env.
# Usage: bash scripts/ada_workflow.sh
set -euo pipefail
MP="${HOME_MP:-$HOME/mp}"
cd "$MP"
NODE=$(cat "$MP/.compute_node" 2>/dev/null || true)
if [[ -z "${NODE:-}" ]]; then
  echo "No .compute_node — run slurm/setup_env.sbatch first" >&2
  exit 2
fi

echo "=== 1. Validate engine (optional EAP-IG install) ==="
sbatch --nodelist="$NODE" "$MP/slurm/run_validate.sbatch"

echo "=== 2. Gate + tokenizer audit ==="
sbatch --nodelist="$NODE" "$MP/slurm/run_gate.sbatch"

echo "After gate finishes, lock cells (no --allow-expected):"
echo "  source env.sh && python scripts/make_cell_list.py --directions denoising noising"
echo "=== 3. Stage 1 grid ==="
echo "  N=\$(python3 -c \"import json; print(json.load(open('results/gated_cells.json'))['n_jobs'])\")"
echo "  sbatch --nodelist=\$NODE --array=0-\$((N-1)) slurm/run_grid.sbatch"
echo "=== 4. Stage 2 language compare + logit lens ==="
echo "  sbatch --nodelist=\$NODE slurm/run_stage2.sbatch"
