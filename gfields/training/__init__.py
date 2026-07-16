# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimization loops: regression of a primitive volume against reference images.

Submodules:
    regression: render -> loss -> step -> checkpoint loop (the optimization pipeline)
    callbacks:  composable callback container (gradient tracker, metrics, plots)
    configs:    per-dataset volume scale / opacity / to_world transforms

The training-time camera samplers (``prepare_cameras`` / ``create_cam_array`` /
``sample_uniform_hemisphere``) live in ``regression`` and are re-exported from
the top-level :mod:`gfields.cameras`.
"""
from __future__ import annotations

from . import callbacks, configs, regression

__all__ = ["callbacks", "configs", "regression"]
