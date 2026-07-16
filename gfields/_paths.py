# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Repository-root resolution, shared by the CLI modules.

The package ships inside ``gfields`` (installed via ``pip install -e .``);
assets, run configs and results all live relative to the repository root.
Resolve it once here instead of per-script ``Path(__file__).parents[N]``.
"""
from __future__ import annotations

from pathlib import Path

# gfields/_paths.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
