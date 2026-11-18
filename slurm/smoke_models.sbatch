#!/bin/bash
# sbatch -A research --qos=medium --export=ALL,SMOKE_MODELS="meta-llama/Llama-3.2-1B google/gemma-3-1b-pt" slurm/smoke_models.sbatch
#SBATCH -p u22
#SBATCH -J mp-smokem
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
echo "JOB $SLURM_JOB_ID host=$(hostname) acct=${SLURM_JOB_ACCOUNT:-?} models=$SMOKE_MODELS"
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$VENV/bin/python" scripts/smoke_ada.py --models $SMOKE_MODELS --n 8 ${SMOKE_EXTRA:-} --out results/smoke_${SLURM_JOB_ID}
echo "EXIT $?"
python - <<'PY'
import torch; print("peak_mem_GB", torch.cuda.max_memory_allocated()/1e9)
PY
