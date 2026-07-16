#!/usr/bin/env bash
# Launch the interactive polyscope viewer for a trained asset.
#   ./view.sh <asset> [flags...]
# Forwards every flag to `python -m gfields viewer`; see `gfields viewer --help`.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"

# Locate a python interpreter (prefer the reproducible uv env from ./install.sh).
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
exec "$py" -m gfields viewer "$@"
