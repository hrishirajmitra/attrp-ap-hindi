#!/bin/bash
#SBATCH -A irel
#SBATCH -p u22
#SBATCH --qos=normal
#SBATCH -J mp-setup
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err

set -euo pipefail
mkdir -p "$HOME/mp-attrpatch-logs" "$HOME/mp/results"

# Load python for venv creation only; setup_env.sh unsets PYTHONHOME afterward
module load u22/python/3.12.15
module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true

export PYTHON=/software/apps/u22/python/3.12.15/bin/python3
export HOME_MP="$HOME/mp"
export SCRATCH_BASE="/ssd_scratch/$USER"

echo "Host=$(hostname) GPU=$(nvidia-smi -L | head -1)"
"$PYTHON" --version
nvidia-smi -L

bash "$HOME_MP/slurm/setup_env.sh"

echo "SETUP_OK node=$(hostname) venv=$SCRATCH_BASE/venvs/mp-attrpatch"
