#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-stage2
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
#
# After stage-1 grid finishes:
#   NODE=$(cat ~/mp/.compute_node)
#   sbatch --nodelist=$NODE ~/mp/slurm/run_stage2.sbatch

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"
module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
module load u22/python/3.12.15 2>/dev/null || true
unset PYTHONHOME
# shellcheck disable=SC1091
source "$HOME/mp/env.sh"
cd "$HOME/mp"

"$VENV/bin/python" scripts/aggregate_seeds.py results
"$VENV/bin/python" scripts/compare_languages.py --results results --out results

# Optional logit-lens on a few cells (short)
for LANG in en hi; do
  "$VENV/bin/python" scripts/logit_lens.py \
    --model Qwen/Qwen2.5-0.5B --task agreement --lang "$LANG" \
    --n-prompts 16 --device cuda \
    --out "results/logit_lens_qwen05_agreement_${LANG}" || true
done

echo "Stage 2 done"
cat results/language_compare.md
