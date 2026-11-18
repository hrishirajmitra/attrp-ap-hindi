#!/bin/bash
# Builds ~/wheelhouse (cp312 manylinux wheels) for node_bootstrap.sh.  CPU-only job.
#   sbatch -A irel --qos=normal ~/mp/slurm/build_wheelhouse.sbatch
#SBATCH -p u22
#SBATCH -J mp-wheelhouse
#SBATCH --exclude=gnode069,gnode077,gnode046
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.out
#SBATCH -e /home2/hrishiraj.mitra/mp-attrpatch-logs/%x-%j.err
set -euo pipefail
module load u22/python/3.12.15
unset PYTHONHOME
export LD_LIBRARY_PATH=/software/apps/u22/python/3.12.15/lib:${LD_LIBRARY_PATH:-}
WH="${MP_WHEELHOUSE:-$HOME/wheelhouse}"
REQ="${MP_REQ:-$HOME/mp/requirements.txt}"
mkdir -p "$WH"
T=$(mktemp -d /tmp/whb.XXXXXX); trap 'rm -rf "$T"' EXIT
/software/apps/u22/python/3.12.15/bin/python3 -m venv "$T/v"
# shellcheck disable=SC1091
source "$T/v/bin/activate"
export PIP_CACHE_DIR="$T/cache"
t0=$SECONDS
# 1) torch (CUDA 12.1 build) + its nvidia-* deps from the PyTorch index
pip download -d "$WH" --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 'torch>=2.1,<2.5'
# 2) everything else from PyPI (torch already in wheelhouse -> satisfied via find-links).
#    `pip wheel` (not `download`) so sdist-only packages (e.g. transformers-stream-generator)
#    are built into wheels once, instead of on every node.
pip install -q wheel setuptools
pip wheel -w "$WH" --find-links "$WH" -r "$REQ"
# 3) tooling so a bare venv can always be upgraded offline
pip download -d "$WH" --only-binary=:all: pip setuptools wheel
echo "WHEELHOUSE_SECONDS $((SECONDS - t0))"
du -sh "$WH"; ls "$WH" | wc -l
# 4) offline install dry-run proves completeness
pip install --no-index --find-links "$WH" --dry-run 'torch>=2.1,<2.5' -r "$REQ" >/dev/null && echo WHEELHOUSE_COMPLETE_OK
