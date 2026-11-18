#!/bin/bash
#SBATCH -p u22
#SBATCH -J mp-eaptest
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
"$VENV/bin/python" scripts/test_eap_equiv.py
