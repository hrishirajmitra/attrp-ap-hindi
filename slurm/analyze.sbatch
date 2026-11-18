#!/bin/bash
# sbatch -A irel --qos=normal slurm/analyze.sbatch   (CPU only)
#SBATCH -p u22
#SBATCH -J mp-analyze
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -uo pipefail
source "$HOME/mp/slurm/node_bootstrap.sh" || { echo BOOTSTRAP_FAILED; exit 1; }
cd "$HOME/mp"
"$VENV/bin/python" scripts/analyze_story.py
echo "ANALYZE_EXIT $?"
