# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""gfields.lod — level of detail: band-weight math, stochastic selection, decomposition.

The runtime LOD math lives in :mod:`gfields.lod.weights` and is re-exported here,
so ``from gfields.lod import LODSelector`` / ``gabor_band_weights`` keep working:

* numpy band math -- ``gabor_frequency_band``, ``gabor_band_weights``,
  ``build_freq_threshold_lut``, ``frequency_threshold_fraction``,
* Dr.Jit runtime -- ``continuous_lod_weight`` and :class:`LODSelector`
  (consumed by ``gfields_prb`` / ``gfields_tomography``).

Decomposition submodules (accessed by full path, e.g. ``gfields.lod.decompose``):
    utils:      canonical orientation-vector tables, steerable sub-band bit
                layout, frequency-band partitioning math, and pyramid-level
                asset plumbing (numpy side) -- single source of truth shared
                by ``LODSelector``, the offline decomposition CLIs, and the
                viewer's LOD slider
    decompose:  the two offline decomposition CLIs -- asset -> steerable
                orientation sub-bands (``decompose_asset`` / ``steerable_main``)
                and trained checkpoint -> frequency-band LOD levels
                (``lod_partition_main``)

plus the CLI entry points dispatched by ``gfields decompose``: ``steerable``
and ``lod-partition`` (both in ``gfields.lod.decompose``), and ``clamp``
(``gfields.training.adaptive``).
"""
from __future__ import annotations

from .weights import (
    DETERMINISTIC_CRITERIA,
    ORIENTATION_CRITERIA,
    STOCHASTIC_CRITERIA,
    LODSelector,
    build_freq_threshold_lut,
    continuous_lod_weight,
    frequency_threshold_fraction,
    gabor_band_weights,
    gabor_frequency_band,
)

__all__ = [
    "DETERMINISTIC_CRITERIA",
    "ORIENTATION_CRITERIA",
    "STOCHASTIC_CRITERIA",
    "LODSelector",
    "build_freq_threshold_lut",
    "continuous_lod_weight",
    "frequency_threshold_fraction",
    "gabor_band_weights",
    "gabor_frequency_band",
]
