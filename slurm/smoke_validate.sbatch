#!/bin/bash
# sbatch -A irel --qos=normal slurm/smoke_validate.sbatch   (or -A research --qos=medium)
#SBATCH -p u22
#SBATCH -J mp-smoke
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /home2/%u/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/%u/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
echo "JOB $SLURM_JOB_ID host=$(hostname) acct=${SLURM_JOB_ACCOUNT:-?}"
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
"$VENV/bin/python" scripts/validate_v1.py; V=$?
"$VENV/bin/python" scripts/smoke_ada.py; S=$?
echo "EXIT validate=$V smoke=$S"
exit $(( V + S ))
