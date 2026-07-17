#!/usr/bin/env bash
# Reproducible environment for volumetric_primitives via uv:
#   1. create a uv-managed .venv (Python 3.13),
#   2. install the gfields package (editable),
#   3. compile the vendored Mitsuba 3 fork against that venv and wire it in,
#   4. validate.
#
#   ./install.sh                 # default: scalar_rgb + cuda_ad_rgb variants
#   ./install.sh --extras        # + torch / lpips / wandb / scikit-image / flip
#   ./install.sh --variants scalar_rgb,cuda_ad_rgb,llvm_ad_rgb   # add CPU fallback
# Extra flags after these are forwarded to gfields/build_mitsuba.py (--jobs, ...).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

PYVER="${VOLPRIM_PYTHON_VERSION:-3.13}"
VENV="$here/.venv"

# Ensure uv (the Python package/venv manager this script drives) is available.
# The official installer drops the binary in ~/.local/bin (or ~/.cargo/bin),
# which may not be on PATH in a fresh shell, so first try to pick up an existing
# install from there before downloading — avoids clobbering a working uv.
if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "'uv' not found; installing via https://astral.sh/uv/install.sh ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' still not found. Install it manually: https://docs.astral.sh/uv/getting-started/installation/  then reopen your shell." >&2; exit 1; }
    # Persist ~/.local/bin on PATH for future shells so uv is found without
    # re-running this. The installer normally does this, but do it ourselves too
    # (covers the already-installed-but-not-on-PATH case). Guarded by a marker so
    # it is appended at most once per profile file.
    uv_dir="$(dirname "$(command -v uv)")"
    marker="# added by gabor_fields install.sh (uv)"
    for profile in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
        [[ -f "$profile" ]] || continue
        grep -qF "$marker" "$profile" 2>/dev/null && continue
        printf '\n%s\nexport PATH="%s:$PATH"\n' "$marker" "$uv_dir" >> "$profile"
        echo "Added $uv_dir to PATH in $profile (permanent)."
    done
fi
command -v cmake >/dev/null 2>&1 || { echo "ERROR: 'cmake' not found — required to compile the Mitsuba fork." >&2; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "ERROR: 'git' not found." >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || echo "WARNING: 'nvidia-smi' not found; the cuda_ad_rgb variant builds but needs an NVIDIA driver at runtime."

# Ensure the pinned submodules are present (never resets a dirty working tree —
# preserves any in-progress fork edits). Mitsuba (+ its ext/drjit at the pinned
# commit, hence --recursive) supplies the renderer and Dr.Jit; volprim supplies
# the `volprim.*` runtime package that gfields imports.
if [[ ! -f submodules/mitsuba3/CMakeLists.txt ]]; then
    echo "Fetching Mitsuba submodule (+ ext/drjit) ..."
    git submodule update --init --recursive submodules/mitsuba3
fi
if [[ ! -f submodules/volprim/setup.py ]]; then
    echo "Fetching volprim submodule ..."
    git submodule update --init submodules/volprim
fi

extras=0; passthru=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extras|--with-extras) extras=1; shift ;;
        *) passthru+=("$1"); shift ;;
    esac
done

echo "== 1/4  create uv venv (python $PYVER) -> .venv =="
uv venv --python "$PYVER" "$VENV"
if [[ -x "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"; else PY="$VENV/bin/python"; fi

echo "== 2/4  install python packages (editable) =="
uv pip install --python "$PY" -e .
# volprim provides the `volprim.*` package gfields imports. Install it editable
# with --no-deps so its 'mitsuba>=3.9' / 'drjit>=1.4' pins do NOT pull the stock
# PyPI wheels over the locally compiled fork wired in step 3 (its other deps —
# numpy/matplotlib/plyfile/ipywidgets/rich — are already covered by gfields).
uv pip install --python "$PY" --no-deps -e submodules/volprim
if [[ $extras -eq 1 ]]; then
    echo "   (+ extras: torch / lpips / wandb / metrics)"
    uv pip install --python "$PY" -e ".[metrics,training,test]"
fi

echo "== 3/4  compile the Mitsuba fork and wire it into the venv =="
"$PY" gfields/build_mitsuba.py --python "$PY" "${passthru[@]}"

echo "== 4/4  validate =="
"$PY" -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import gfields, gfields.cli; print('OK: mitsuba', mi.__version__, '| gfields import OK')"

cat <<EOF

Environment ready in .venv .
  Activate:  source .venv/bin/activate            (Windows: .venv\\Scripts\\activate)
  Train:     examples/train.sh wdas8    (launchers auto-detect .venv)
  Override interpreter anywhere with VOLPRIM_PYTHON=/path/to/python.
EOF
