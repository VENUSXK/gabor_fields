# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cameras: the single home for every camera helper in gfields.

Three groups:

* **Upstream camera-spec IO** -- ``from volprim.cameras import *`` re-exports the
  unmodified upstream loaders (``CameraSpecs``, ``JSON``/``KRT``/``Colmap``
  spec IO, ``fov2focal`` / ``focal2fov``).
* **Scene-camera factories** (rendering CLIs) -- :func:`create_cameras`
  (turntable / arc), :func:`create_side_top_camera` (single orbiting sensor),
  :func:`compute_asset_aabb_from_scene_dict` (loose AABB over ellipsoidsmesh).
* **Training-time stratified samplers** -- ``prepare_cameras``,
  ``create_cam_array``, ``sample_uniform_hemisphere`` are resolved lazily from
  :mod:`gfields.training.regression` (see :func:`__getattr__`) so importing this
  module never pulls in the heavy training stack.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from volprim.cameras import *  # noqa: F401,F403  (upstream camera-spec IO)


# ---------------------------------------------------------------------------
# Scene-camera factories
# ---------------------------------------------------------------------------

def create_cameras(
    cam_count: int,
    cam_res_x: int,
    cam_res_y: int,
    *,
    cam_scale: float = 2.35,
    use_perspective: bool = False,
    fov: float = 40.0,
    distance: float = 24.0,
) -> List:
    """Turntable arc of cameras around the origin.

    Defaults to ``orthographic`` sensors (no focal length; framing controlled by
    ``cam_scale``). With ``use_perspective=True``, uses ``perspective`` with the
    given horizontal ``fov`` -- Mitsuba rejects uniform scale in perspective
    ``to_world`` so ``cam_scale`` is ignored in that mode.
    """
    import mitsuba as mi

    T = mi.ScalarTransform4f
    cameras: List = []

    for i in range(cam_count):
        angle = 180.0 / cam_count * i - 90.0
        elevation = 90.0 * np.random.rand() - 45.0
        sensor_rotation_y = T().rotate([0, 1, 0], angle)
        sensor_rotation_x = T().rotate([1, 0, 0], elevation)
        base_to_world = T().look_at(target=[0, 0, 0], origin=[0, 0, distance], up=[0, 1, 0])
        base_to_world = sensor_rotation_y @ sensor_rotation_x @ base_to_world
        if use_perspective:
            sensor_to_world = base_to_world
            cam_dict: Dict[str, Any] = {
                "type": "perspective",
                "fov": fov,
                "to_world": sensor_to_world,
            }
        else:
            sensor_to_world = base_to_world @ T().scale([cam_scale, cam_scale, cam_scale])
            cam_dict = {"type": "orthographic", "to_world": sensor_to_world}
        cam_dict["film"] = {
            "type": "hdrfilm",
            "width": cam_res_x,
            "height": cam_res_y,
            "filter": {"type": "gaussian"},
        }
        cameras.append(mi.load_dict(cam_dict))
    return cameras


def create_side_top_camera(
    cam_res_x: int,
    cam_res_y: int,
    *,
    azimuth_deg: float = 30.0,
    elevation_deg: float = 25.0,
    distance: float = 24.0,
    fov: float = 35.0,
    target: Optional[List[float]] = None,
    use_perspective: bool = True,
    cam_scale: float = 2.35,
):
    """Single perspective (default) sensor orbiting ``target`` on a sphere."""
    import mitsuba as mi

    T = mi.ScalarTransform4f
    if target is None:
        target = [0.0, 0.0, 0.0]
    target = [float(target[0]), float(target[1]), float(target[2])]

    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    ox = np.sin(az) * np.cos(el) * float(distance)
    oy = np.sin(el) * float(distance)
    oz = np.cos(az) * np.cos(el) * float(distance)
    origin = [target[0] + ox, target[1] + oy, target[2] + oz]

    sensor_to_world = T.look_at(target=target, origin=origin, up=[0, 1, 0])
    cam_dict: Dict[str, Any]
    if use_perspective:
        cam_dict = {"type": "perspective", "fov": fov, "to_world": sensor_to_world}
    else:
        cam_dict = {
            "type": "orthographic",
            "to_world": sensor_to_world @ T.scale([cam_scale, cam_scale, cam_scale]),
        }
    cam_dict["film"] = {
        "type": "hdrfilm",
        "width": cam_res_x,
        "height": cam_res_y,
        "filter": {"type": "gaussian"},
    }
    return mi.load_dict(cam_dict)


def compute_asset_aabb_from_scene_dict(scene_dict: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """Loose axis-aligned bounding box over every ``ellipsoidsmesh`` in *scene_dict*."""
    all_min: Optional[np.ndarray] = None
    all_max: Optional[np.ndarray] = None
    for v in scene_dict.values():
        if not isinstance(v, dict) or v.get("type") != "ellipsoidsmesh":
            continue
        centers = v.get("centers")
        scales = v.get("scales")
        if centers is None or scales is None:
            continue
        c_arr = np.asarray(centers).reshape(-1, 3)
        s_arr = np.asarray(scales).reshape(-1, 3)
        ext_v = v.get("extent", 1.0)
        if hasattr(ext_v, "shape") and len(getattr(ext_v, "shape", ()) or ()) > 0:
            e_arr = np.asarray(ext_v).reshape(-1, 1)
            if e_arr.shape[0] != c_arr.shape[0]:
                e_arr = np.full((c_arr.shape[0], 1), float(np.asarray(ext_v).reshape(-1)[0]))
        else:
            e_arr = np.full((c_arr.shape[0], 1), float(ext_v))
        radius = e_arr * np.linalg.norm(s_arr, axis=1, keepdims=True)
        lo = (c_arr - radius).min(axis=0)
        hi = (c_arr + radius).max(axis=0)
        all_min = lo if all_min is None else np.minimum(all_min, lo)
        all_max = hi if all_max is None else np.maximum(all_max, hi)
    if all_min is None or all_max is None:
        return None
    return {
        "min": all_min,
        "max": all_max,
        "center": 0.5 * (all_min + all_max),
        "size": all_max - all_min,
    }


# ---------------------------------------------------------------------------
# Training-time stratified samplers (lazy -> gfields.training.regression)
# ---------------------------------------------------------------------------

_TRAINING_SAMPLERS = ("prepare_cameras", "create_cam_array", "sample_uniform_hemisphere")


def __getattr__(name: str) -> Any:
    """Resolve the training-time camera samplers on demand, keeping ``import
    gfields.cameras`` free of the heavy training stack."""
    if name in _TRAINING_SAMPLERS:
        from gfields.training import regression
        return getattr(regression, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
