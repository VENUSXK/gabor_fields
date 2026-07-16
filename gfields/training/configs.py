# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Keeping here the specific configs used for the paper assets we trained.
Just for reference.
"""
from __future__ import annotations

import abc

import numpy as np
from mitsuba import ScalarTransform4f as Transform


class VolumeConfig(abc.ABC):
    """Abstract base class for volume configurations."""

    name: str
    to_world: Transform
    volume_scale: np.ndarray
    opacity_scale: float


def get_volume_config(path: str, opacity_scale: float | None = None) -> VolumeConfig:
    """Pick the matching :class:`VolumeConfig` subclass from a volume-grid path.

    ``opacity_scale`` overrides the density scale of the matched config. For an
    *unknown* path it also unlocks a :class:`GenericConfig` fallback (isotropic
    2-unit cube), so a user can train their own volume straight from the CLI::

        gfields optimize gabor --volume_grid my.npy --opacity_scale 12 ...

    Without an explicit ``opacity_scale`` an unknown path still raises, to catch
    typos rather than silently mis-scaling a known asset.
    """
    p = path.lower()
    if "bunny" in p:
        cfg: VolumeConfig = BunnyConfig()
    elif "wdas" in p:
        cfg = DisneyConfig()
    elif "cloudy" in p:
        cfg = CloudyConfig()
    elif "cloud" in p:
        cfg = CloudConfig()
    elif "tornado" in p:
        cfg = TornadoConfig()
    elif "explosion" in p:
        cfg = ExplosionConfig()
    elif "fire" in p:
        cfg = FireConfig()
    elif "embergen" in p:
        cfg = EmbergenConfig()
    elif "smoke2" in p:
        cfg = Smokev2Config()
    elif "smoke" in p:
        cfg = SmokeConfig()
    elif "dust" in p:
        cfg = DustShockwaveConfig()
    elif opacity_scale is not None:
        import warnings
        warnings.warn(
            f"No named VolumeConfig for {path!r}; using GenericConfig "
            "(isotropic 2-unit cube centred at the origin). Density is set by "
            "--opacity_scale; add a VolumeConfig subclass if you need a "
            "non-cubic to_world transform."
        )
        cfg = GenericConfig()
    else:
        raise ValueError(
            f"Unknown volume configuration for path: {path}. Either add a "
            "matching VolumeConfig subclass, or pass --opacity_scale <float> to "
            "train it with the generic isotropic config."
        )
    if opacity_scale is not None:
        cfg.opacity_scale = float(opacity_scale)
    return cfg


class CloudyConfig(VolumeConfig):
    name = "cloudy"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 20.0


class DisneyConfig(VolumeConfig):
    name = "disney"
    shape = 1316 + 25, 895 + 37, 1622 + 21
    _ms = max(shape)
    to_world = (
        Transform()
        .scale([shape[2] / _ms, shape[1] / _ms, shape[0] / _ms])
        .scale([2.0, 2.0, 2.0])
        .translate(-0.5)
    )
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 10.0


class BunnyConfig(VolumeConfig):
    name = "bunny"
    to_world = Transform().scale([2, 2, 2]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 10.0


class CloudConfig(VolumeConfig):
    name = "cloud"
    to_world = Transform().scale([1.650, 1.1232, 2.0]).translate(-0.5)
    volume_scale = np.array([[2.0, 1.1232, 1.650]])
    opacity_scale = 7.5


class TornadoConfig(VolumeConfig):
    name = "tornado"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 60.0


class EmbergenConfig(VolumeConfig):
    name = "embergen"
    to_world = Transform().scale([2, 2, 2]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 20.0


class Smokev2Config(VolumeConfig):
    name = "smokev2"
    to_world = Transform().scale([2, 2, 2]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 300.0


class SmokeConfig(VolumeConfig):
    name = "smoke"
    to_world = Transform().scale([2, 2, 2]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 7.5


class DustShockwaveConfig(VolumeConfig):
    name = "dust_shockwave"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 70.0


class ExplosionConfig(VolumeConfig):
    name = "explosion"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 15.0


class FireConfig(VolumeConfig):
    name = "fire"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 15.0


class GenericConfig(VolumeConfig):
    """Fallback for volumes without a named config: isotropic 2-unit cube.

    Density is expected to be supplied via ``--opacity_scale`` (see
    :func:`get_volume_config`). Only the ``opacity_scale`` here is a placeholder.
    """
    name = "generic"
    to_world = Transform().scale([2.0, 2.0, 2.0]).translate(-0.5)
    volume_scale = np.array([2.0, 2.0, 2.0])
    opacity_scale = 10.0


__all__ = [
    "VolumeConfig",
    "get_volume_config",
    "GenericConfig",
    "BunnyConfig",
    "CloudConfig",
    "CloudyConfig",
    "DisneyConfig",
    "DustShockwaveConfig",
    "EmbergenConfig",
    "ExplosionConfig",
    "FireConfig",
    "SmokeConfig",
    "Smokev2Config",
    "TornadoConfig",
]
