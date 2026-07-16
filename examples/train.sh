#!/usr/bin/env bash
# Train volumetric primitives with the paper's winning recipe.
#   ./train.sh <preset|volume>            # Gaussian base + Gabor residual (paper recipe)
#   ./train.sh <preset|volume> --gaussian # Gaussian-only baseline
# <preset> is a paper asset name (wdas8, embergen, ...); <volume> is a .npy/.vol/.vdb
# file (give an unknown volume a density scale, e.g. --opacity_scale 12). Every
# extra flag overrides the recipe; see `python -m gfields.train --help`.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"

find_python() {
    if [[ -n "${VOLPRIM_PYTHON:-}" && -x "$VOLPRIM_PYTHON" ]]; then
        printf '%s\n' "$VOLPRIM_PYTHON"; return 0
    fi
    for c in "$repo/.venv/Scripts/python.exe" "$repo/.venv/bin/python" \
             "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" \
             "${CONDA_PREFIX:+$CONDA_PREFIX/python.exe}" \
             "$HOME/anaconda3/envs/volprim/bin/python" \
             "$HOME/miniconda3/envs/volprim/bin/python" \
             "$(command -v python3 2>/dev/null || true)" \
             "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "$c" && -x "$c" ]]; then printf '%s\n' "$c"; return 0; fi
    done
    echo "Python not found. Run ./install.sh or set VOLPRIM_PYTHON." >&2
    return 1
}
py="$(find_python)"

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # opt in with WANDB_MODE=online
cd "$repo"
exec "$py" -m gfields.train "$@"
