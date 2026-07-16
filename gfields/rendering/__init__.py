# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rendering pipelines for the gfields plugin.

Submodules:
    pipeline:   asset loading, scene-dict construction, integrator/environment configs
    video:      single-frame rendering, video frame loops, ffmpeg MP4 assembly

Camera factories live in the top-level :mod:`gfields.cameras`; timing helpers in
:mod:`gfields.benchmark`.
"""
from __future__ import annotations

from . import pipeline, video

__all__ = ["pipeline", "video"]
