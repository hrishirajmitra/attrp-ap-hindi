# shellcheck shell=bash
# node_bootstrap.sh -- SOURCE this from worker jobs (do not execute).
#
#   source "$HOME/mp/slurm/node_bootstrap.sh" || exit 1
#
# Idempotent, race-free (flock on node-local scratch), node-agnostic.
# After sourcing: $VENV/bin/python is a healthy env (torch 2.1-2.4 cu121 +
# transformer-lens 2.x + requirements.txt), HF caches/token/LD_LIBRARY_PATH are
# exported. Returns non-zero (does NOT exit the caller) on failure.
#
# Env overrides: MP_SCRATCH, MP_REQ, MP_WHEELHOUSE, MP_PYTHON, MP_BOOTSTRAP_FORCE=1
# Reports:  MP_BOOTSTRAP_SECONDS, MP_BOOTSTRAP_BUILT (1 if this job built the venv)

_mpb_t0=$SECONDS
_mpb_log() { echo "[node_bootstrap $(hostname) +$((SECONDS - _mpb_t0))s] $*"; }

# Pick the local disk with the most free space: /ssd_scratch is shared and sometimes full
# (observed 0-22 GB free on gnode046/082) while /tmp is a separate 1.8 TB disk on every node.
if [[ -z "${MP_SCRATCH:-}" ]]; then
  _best=0; _bestdir=""
  for _d in /ssd_scratch /tmp; do
    [[ -d "$_d" && -w "$_d" ]] || continue
    _avail=$(df -BG --output=avail "$_d" 2>/dev/null | tail -1 | tr -dc '0-9')
    _avail=${_avail:-0}
    if (( _avail > _best )); then _best=$_avail; _bestdir=$_d; fi
  done
  if (( _best < 20 )); then
    echo "[node_bootstrap $(hostname)] ERROR: no local disk with >=20GB free (best=${_best}G)"; return 1
  fi
  MP_SCRATCH="$_bestdir/hrishiraj.mitra"
fi
MP_REQ="${MP_REQ:-$HOME/mp/requirements.txt}"
MP_WHEELHOUSE="${MP_WHEELHOUSE:-$HOME/wheelhouse}"
MP_PYTHON="${MP_PYTHON:-/software/apps/u22/python/3.12.15/bin/python3}"
MP_PYLIB="$(dirname "$(dirname "$MP_PYTHON")")/lib"
export MP_BOOTSTRAP_BUILT=0

# ---- modules (as in the old sbatch scripts) -------------------------------
if type module >/dev/null 2>&1; then
  module load u22/cuda/12.1 2>/dev/null || module load u22/cuda/12.4 2>/dev/null || true
  module load u22/python/3.12.15 2>/dev/null || true
fi
unset PYTHONHOME

# ---- environment (identical on every node; NOT pinned to a node) -----------
mkdir -p "$MP_SCRATCH/venvs" "$MP_SCRATCH/hf" "$MP_SCRATCH/pipcache" "$HOME/mp-attrpatch-logs" || {
  _mpb_log "ERROR: cannot create $MP_SCRATCH"; return 1; }
export LD_LIBRARY_PATH="$MP_PYLIB:${LD_LIBRARY_PATH:-}"
export HF_HOME="$MP_SCRATCH/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# keep huggingface_hub and transformers on the SAME cache dir (TRANSFORMERS_CACHE
# otherwise diverges from $HF_HOME/hub and models get downloaded twice)
export HF_HUB_CACHE="$TRANSFORMERS_CACHE"
export HUGGINGFACE_HUB_CACHE="$TRANSFORMERS_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$HF_HOME/torch"
export VENV="$MP_SCRATCH/venvs/mp-attrpatch"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PIP_CACHE_DIR="$MP_SCRATCH/pipcache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
if [[ -r "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
else
  _mpb_log "WARNING: no HF token at ~/.cache/huggingface/token"
fi

_MPB_LOCK="$MP_SCRATCH/.bootstrap.lock"
_MPB_MARK="$VENV/.mp_bootstrap_ok"

_mpb_gpu_visible() {
  [[ -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}${SLURM_STEP_GPUS:-}" ]] && return 0
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q 'GPU'
}

# 0=import+version ok (+cuda ok if GPU visible); 1=import/version bad; 2=imports ok but cuda unavailable
_mpb_health() {
  [[ -x "$VENV/bin/python" ]] || return 1
  local want_gpu=0
  _mpb_gpu_visible && want_gpu=1
  "$VENV/bin/python" - "$want_gpu" 2>/dev/null <<'PY'
import sys
want_gpu = sys.argv[1] == "1"
try:
    import torch, transformer_lens, transformers  # noqa: F401
    tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    assert (2, 1) <= tv < (2, 5), torch.__version__
    assert (torch.version.cuda or "").startswith("12.1"), torch.version.cuda
except Exception as e:  # noqa: BLE001
    print("import/version check failed:", repr(e), file=sys.stderr); sys.exit(1)
if want_gpu and not torch.cuda.is_available():
    sys.exit(2)
sys.exit(0)
PY
}

_mpb_pip_install() {
  # $1 = "offline" | "online"
  local mode="$1" whl="$MP_WHEELHOUSE"
  if [[ "$mode" == offline ]]; then
    ls "$whl"/torch-*.whl >/dev/null 2>&1 || return 1
    pip install --no-index --find-links "$whl" 'torch>=2.1,<2.5' && \
    pip install --no-index --find-links "$whl" -r "$MP_REQ"
  else
    pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.1,<2.5' && \
    pip install --find-links "$whl" -r "$MP_REQ"
  fi
}

_mpb_build() {
  _mpb_log "building venv at $VENV"
  rm -f "$_MPB_MARK"
  rm -rf "$VENV"
  "$MP_PYTHON" -m venv "$VENV" || return 1
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  unset PYTHONHOME
  local t=$SECONDS
  if _mpb_pip_install offline; then
    _mpb_log "offline (wheelhouse) install ok in $((SECONDS - t))s"
  else
    _mpb_log "wheelhouse install unavailable/failed -> falling back to PyPI + download.pytorch.org"
    t=$SECONDS
    _mpb_pip_install online || { _mpb_log "ERROR: pip install failed"; return 1; }
    _mpb_log "online install ok in $((SECONDS - t))s"
  fi
  export MP_BOOTSTRAP_BUILT=1
}

_mpb_main() {
  local rc=0
  if [[ "${MP_BOOTSTRAP_FORCE:-0}" != 1 && -f "$_MPB_MARK" ]] && _mpb_health; then
    _mpb_log "venv healthy (fast path, no lock)"
    return 0
  fi

  _mpb_log "acquiring lock $_MPB_LOCK"
  local fd
  exec {fd}>"$_MPB_LOCK" || return 1
  flock -w 5400 "$fd" || { _mpb_log "ERROR: lock timeout"; exec {fd}>&-; return 1; }
  _mpb_log "lock acquired"

  _mpb_health; rc=$?
  if [[ "${MP_BOOTSTRAP_FORCE:-0}" != 1 && $rc -eq 0 ]]; then
    touch "$_MPB_MARK"
    _mpb_log "venv healthy (after lock)"
  elif [[ "${MP_BOOTSTRAP_FORCE:-0}" != 1 && $rc -eq 2 && -f "$_MPB_MARK" ]]; then
    _mpb_log "ERROR: venv imports fine but torch.cuda.is_available() is False although a GPU is visible. Node/GPU problem; NOT rebuilding."
    nvidia-smi -L 2>&1 | head -3
    exec {fd}>&-
    return 3
  else
    _mpb_build || { exec {fd}>&-; return 1; }
    _mpb_health; rc=$?
    if [[ $rc -eq 0 ]]; then
      touch "$_MPB_MARK"
      _mpb_log "build finished and healthy"
    else
      _mpb_log "ERROR: freshly built venv failed health check (rc=$rc)"
      exec {fd}>&-
      return 1
    fi
  fi
  exec {fd}>&-
  return 0
}

_mpb_main
_mpb_rc=$?
export MP_BOOTSTRAP_SECONDS=$((SECONDS - _mpb_t0))
if [[ $_mpb_rc -eq 0 ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" 2>/dev/null || true
  unset PYTHONHOME
  export LD_LIBRARY_PATH="$MP_PYLIB:${LD_LIBRARY_PATH#"$MP_PYLIB:"}"
  _mpb_log "READY venv=$VENV built_here=$MP_BOOTSTRAP_BUILT total=${MP_BOOTSTRAP_SECONDS}s"
  echo "BOOTSTRAP_SECONDS $MP_BOOTSTRAP_SECONDS built=$MP_BOOTSTRAP_BUILT host=$(hostname)"
fi
(exit $_mpb_rc)
