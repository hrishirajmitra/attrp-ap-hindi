#!/bin/bash
# Queue worker. Launch on both accounts, e.g.
#   sbatch -A irel     --qos=normal --array=0-11 slurm/worker.sbatch
#   sbatch -A research --qos=medium --array=0-3  slurm/worker.sbatch
#SBATCH -p u22
#SBATCH -J mp-worker
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH -t 08:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%A_%a.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%A_%a.err
set -uo pipefail
echo "JOB $SLURM_JOB_ID/${SLURM_ARRAY_TASK_ID:-0} host=$(hostname) acct=${SLURM_JOB_ACCOUNT:-?}"
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
"$VENV/bin/python" scripts/worker.py --queue-dir "$HOME/mp/queue" --results-dir "$HOME/mp/results/grid" \
  --time-limit-s 28500 ${WORKER_ARGS:-}
