#!/bin/bash
# One job per model:
#   sbatch -A irel --qos=normal --export=ALL,GATE_MODEL=gpt2 slurm/gate_model.sbatch
#SBATCH -p u22
#SBATCH -J mp-gate
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
: "${GATE_MODEL:?set GATE_MODEL}"
echo "JOB $SLURM_JOB_ID host=$(hostname) acct=${SLURM_JOB_ACCOUNT:-?} model=$GATE_MODEL"
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
"$VENV/bin/python" scripts/gate.py --models "$GATE_MODEL" --out-dir "$HOME/mp/results/gate"
rc=$?
echo "GATE_EXIT $rc"
exit $rc
