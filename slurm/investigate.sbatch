#!/bin/bash
# Read-only investigation job (TL loading of gated models via mirrors + peak GPU memory).
#   sbatch -A irel --qos=normal [--nodelist=<node with prefetched models>] ~/mp/slurm/investigate.sbatch [alt]
# "alt" additionally builds a throw-away venv (torch 2.6 cu124 + TL 2.18.0, transformers>=4.57) in
# node-local scratch and re-runs the investigation there (NOT touching the main venv).
#SBATCH -p u22
#SBATCH -J mp-investigate
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
echo "JOB $SLURM_JOB_ID host=$(hostname) account=${SLURM_JOB_ACCOUNT:-?}"; nvidia-smi -L
# shellcheck disable=SC1091
source "$HOME/mp/slurm/node_bootstrap.sh" || exit 1
bash "$HOME/mp/slurm/prefetch_models.sh" || exit 1

echo "######## MAIN ENV (torch 2.4.1 + TL 2.15.x) ########"
"$VENV/bin/python" "$HOME/mp/slurm/investigate_models.py"

if [[ "${1:-}" == alt ]]; then
  echo "######## ALT ENV (torch 2.6 cu124 + TL 2.18.0) ########"
  ALT="$MP_SCRATCH/venvs/probe-tl218"
  if [[ ! -x "$ALT/bin/python" ]]; then
    t=$SECONDS
    "$MP_PYTHON" -m venv "$ALT"
    "$ALT/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0' || exit 1
    "$ALT/bin/pip" install -q 'transformer-lens==2.18.0' 'transformers>=4.57,<5' 'numpy<2' accelerate sentencepiece protobuf || exit 1
    echo "ALT_INSTALL_SECONDS $((SECONDS - t))"
  fi
  "$ALT/bin/pip" list 2>/dev/null | grep -Ei '^(torch|transformer-lens|transformers|numpy|huggingface|tokenizers) '
  "$ALT/bin/python" -c 'import torch;print("cuda",torch.cuda.is_available(), torch.version.cuda)'
  "$ALT/bin/python" "$HOME/mp/slurm/investigate_models.py"
fi
