# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Small, dependency-light helpers shared across ``gfields.lod``.

Three families:

* **Orientation tables & bins** -- :func:`orientation_vectors`,
  :func:`steerable_subband_bit`, :func:`generate_orientation_bins` and
  :func:`compute_visibility_mask` are the single source of truth for the
  steerable encoding's bin directions and bit layout. Both the runtime selector
  (:class:`gfields.lod.weights.LODSelector`, which picks per-ray orientation
  sub-bands) and the offline steerable decomposition
  (:func:`gfields.lod.decompose.decompose_asset`, which assigns each Gabor to a
  bin via the quaternion helpers below) must agree on these tables, so both
  route through here instead of keeping hand-kept copies.

* **Band partitioning** -- :func:`double_spaced` and :func:`equal_partitioning`
  compute the LOD-level frequency bounds used by the offline LOD-pyramid
  partitioning (:mod:`gfields.lod.decompose`).

* **Pyramid-level asset plumbing** (numpy side) -- loads one on-disk pyramid
  level into numpy arrays, classifies Gabor vs Gaussian bands, and builds the
  ``ellipsoidsmesh`` scene-dict entry / PLY export used by the rendering CLIs
  and the viewer's LOD slider.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Canonical local-space frequency direction of a Gabor primitive (body diagonal).
LOCAL_FREQ_DIR = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Orientation bins (steerable encoding)
# ---------------------------------------------------------------------------


def orientation_vectors(num_orientations: int) -> np.ndarray:
    """Unit orientation-bin directions for ``num_orientations`` bins, shape (n, 3).

    * 2 -- X, Y (limited hemisphere coverage; prefer 3+),
    * 3 -- axis-aligned X, Y, Z,
    * 4 -- tetrahedral (four body diagonals of a cube),
    * 6 -- +/- X, Y, Z,
    * 7 -- the three axes plus the four tetrahedral directions,
    * otherwise -- axis-aligned, cycling X/Y/Z.
    """
    s = 1.0 / np.sqrt(3.0)
    axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    tetra = [(s, s, s), (s, s, -s), (s, -s, s), (-s, s, s)]

    if num_orientations == 2:
        vecs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    elif num_orientations == 3:
        vecs = axes
    elif num_orientations == 4:
        vecs = tetra
    elif num_orientations == 6:
        vecs = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    elif num_orientations == 7:
        vecs = axes + tetra
    else:
        vecs = [axes[i % 3] for i in range(num_orientations)]
    return np.asarray(vecs, dtype=np.float64)


def steerable_subband_bit(level: int, ori: int, num_orientations: int) -> int:
    """Visibility-mask bit index of a ``(level, orientation)`` sub-band in the
    steerable encoding -- the single source of truth for the bit layout.

    Bit 7 is the Gaussian base (``level == 0``); each Gabor level ``l >= 1``
    lays its orientation bins out descending from bit 6 as
    ``7 - (1 + (l - 1) * num_orientations + ori)``. Both the runtime selector
    (:class:`gfields.lod.weights.LODSelector`) and the offline decomposition
    (via :func:`compute_visibility_mask`) must agree on this, so they both route
    through here.
    """
    if level == 0:
        return 7
    return 7 - (1 + (level - 1) * num_orientations + ori)


@dataclass
class OrientationBin:
    """An orientation bin: an index + a unit direction vector."""

    index: int
    direction: np.ndarray

    def alignment(self, direction: np.ndarray) -> float:
        return float(np.abs(np.dot(self.direction, direction)))


def generate_orientation_bins(num_orientations: int) -> list[OrientationBin]:
    """Orientation bins from the canonical :func:`orientation_vectors` table
    (same directions the runtime ``LODSelector`` selects)."""
    return [OrientationBin(i, v) for i, v in enumerate(orientation_vectors(num_orientations))]


def compute_visibility_mask(freq_level: int, ori_idx: int, num_orientations: int) -> int:
    """Visibility-mask bit for one sub-band (bit 7 = Gaussian base, counting down)."""
    return 1 << steerable_subband_bit(freq_level, ori_idx, num_orientations)


# ---------------------------------------------------------------------------
# Quaternion / world-frequency helpers (steerable decomposition)
# ---------------------------------------------------------------------------


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix (matches DrJit convention)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def compute_world_frequency_direction(quaternion: np.ndarray) -> np.ndarray:
    """World-space frequency direction of a Gabor primitive."""
    d = quaternion_to_rotation_matrix(quaternion) @ LOCAL_FREQ_DIR
    return d / (np.linalg.norm(d) + 1e-10)


# ---------------------------------------------------------------------------
# Frequency-band LoD partitioning
# ---------------------------------------------------------------------------


def double_spaced(lower: float, upper: float, n_samples: int, /) -> np.ndarray:
    """``n_samples`` between *lower* and *upper* with geometric (doubling) spacing."""
    i = np.arange(n_samples)
    return lower + (upper - lower) * (2 ** i - 1) / (2 ** (n_samples - 1) - 1)


def equal_partitioning(criteria: np.ndarray, n_samples: int) -> np.ndarray:
    """Partition *criteria* into ``n_samples`` bins with an equal number of items."""
    sorted_criteria = np.sort(criteria)
    bounds = [sorted_criteria[0]]
    N = len(criteria)
    for i in range(1, n_samples - 1):
        idx = int(i * N / (n_samples - 1))
        bounds.append(sorted_criteria[idx])
    bounds.append(sorted_criteria[-1] + 1e-5)
    return np.array(bounds)


# ---------------------------------------------------------------------------
# Pyramid-level asset plumbing (numpy side) for LOD rendering
# ---------------------------------------------------------------------------


def _to_numpy_f32(tensor: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Detach drjit / Mitsuba tensors to contiguous float32 numpy."""
    if hasattr(tensor, "numpy"):
        arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(shape)


def _mesh_tensor(flat: np.ndarray, shape: tuple[int, ...]):
    import mitsuba as mi

    return mi.TensorXf(
        np.ascontiguousarray(flat, dtype=np.float32).reshape(-1),
        shape=shape,
    )


def load_pyramid_metadata(asset_root: Path) -> dict[str, Any]:
    """Training metadata used to classify on-disk pyramid bands."""
    gabor_level, gauss_level = 1, 0
    skip_gabor = False
    skip_gaussian = False
    cam_scale = 0.0
    for meta_name in ("pyramid_cache_metadata.json", "args.json"):
        meta_path = asset_root / meta_name
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if "gabor_target_level" in meta:
            gabor_level = int(meta["gabor_target_level"])
        if "gauss_target_level" in meta:
            gauss_level = int(meta["gauss_target_level"])
        skip_gabor = bool(meta.get("skip_gabor_optim", skip_gabor))
        skip_gaussian = bool(meta.get("skip_gaussian_optim", skip_gaussian))
        if "cam_scale" in meta:
            cam_scale = float(meta["cam_scale"])
        break
    return {
        "gabor_target_level": gabor_level,
        "gauss_target_level": gauss_level,
        "skip_gabor_optim": skip_gabor,
        "skip_gaussian_optim": skip_gaussian,
        "cam_scale": cam_scale,
    }


def load_pyramid_target_levels(asset_root: Path) -> tuple[int, int]:
    meta = load_pyramid_metadata(asset_root)
    return meta["gabor_target_level"], meta["gauss_target_level"]


def is_gabor_from_omega(omega: Any) -> bool:
    """Single source of truth for "does this band oscillate?".

    The shape plugin no longer carries an ``is_gabor``/``kernel_type`` flag: the
    Gabor-Gaussian kernel is a plain Gaussian exactly when ``omega == 0``
    (``cos(0) == 1``) and oscillates for any nonzero ``omega``. Gabor-ness is
    therefore derived from the ``omega`` column wherever a boolean is needed.
    ``omega is None`` (a level stored without the column) is treated as Gaussian.
    """
    if omega is None:
        return False
    return bool(np.any(np.asarray(omega, dtype=np.float32) != 0.0))


def is_gabor_for_level(
    level: int,
    gabor_target_level: int,
    gauss_target_level: int,
    *,
    num_levels: int = 1,
    skip_gabor_optim: bool = False,
    skip_gaussian_optim: bool = False,
) -> bool:
    """
    Classify a saved pyramid band (on-disk as ``data/root.primitives_pyr{level}.ply``)
    as Gabor vs Gaussian for rendering.

    ``gabor_target_level`` / ``gauss_target_level`` in args.json describe the
    optimization hierarchy and may not equal on-disk band indices.

    """
    if num_levels == 1:
        if skip_gabor_optim and not skip_gaussian_optim:
            return False
        if skip_gaussian_optim and not skip_gabor_optim:
            return True
        if level == gabor_target_level:
            return True
        if level == gauss_target_level:
            return False
        return level > 0
    if level == gauss_target_level and gauss_target_level < num_levels:
        return False
    if level == 0:
        return False
    if level == gabor_target_level:
        return True
    return level > 0


def compute_visual_aabb_from_levels(levels: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    all_min: np.ndarray | None = None
    all_max: np.ndarray | None = None
    for data in levels:
        c_arr = data["centers"]
        s_arr = data["scales"]
        margin = np.max(s_arr, axis=1, keepdims=True) * 2.0
        lo = (c_arr - margin).min(axis=0)
        hi = (c_arr + margin).max(axis=0)
        all_min = lo if all_min is None else np.minimum(all_min, lo)
        all_max = hi if all_max is None else np.maximum(all_max, hi)
    return {
        "min": all_min,
        "max": all_max,
        "center": 0.5 * (all_min + all_max),
        "size": all_max - all_min,
    }


def subset_numpy_level(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(mask)
    for key, val in data.items():
        if isinstance(val, np.ndarray) and val.shape[0] == n:
            out[key] = np.ascontiguousarray(val[mask])
        else:
            out[key] = val
    return out


def load_level_numpy(
    asset_info,
    level: int,
    *,
    albedo_scale: float,
    opacity_scale: float,
    extent: Any,
    world_scale: float,
    use_asset_albedo: bool,
) -> dict[str, Any]:
    from gfields.rendering.pipeline import load_pyramid_level
    prim = load_pyramid_level(
        asset_info,
        level,
        albedo_scale=albedo_scale,
        opacity_scale=opacity_scale,
        extent=extent,
        world_scale=world_scale,
        use_asset_albedo=use_asset_albedo,
    )
    extent = prim["extent"]
    if hasattr(extent, "numpy"):
        extent_np = _to_numpy_f32(extent, (-1, 1))
    else:
        extent_np = np.full((prim["centers"].shape[0], 1), float(extent), dtype=np.float32)
    n = int(prim["centers"].shape[0])
    return {
        "centers": _to_numpy_f32(prim["centers"], (n, 3)),
        "scales": _to_numpy_f32(prim["scales"], (n, 3)),
        "quaternions": _to_numpy_f32(prim["quaternions"], (n, 4)),
        "opacities": _to_numpy_f32(prim["opacities"], (n, 1)),
        "omega": _to_numpy_f32(prim["omega"], (n, 1)),
        "albedo": _to_numpy_f32(prim["albedo"], (n, 3)),
        "extent": np.ascontiguousarray(extent_np.reshape(-1, 1), dtype=np.float32),
    }


def _extent_mesh_value(extent: np.ndarray, n: int) -> Any:
    extent = np.asarray(extent, dtype=np.float32).reshape(-1)
    if extent.size == 1:
        return float(extent[0])
    if extent.size != n:
        extent = np.full(n, float(extent[0]), dtype=np.float32)
    return _mesh_tensor(extent, (n, 1))


def write_numpy_level_ply(data: dict[str, Any], ply_path: Path) -> None:
    """Write primitive arrays to PLY (avoids AD tensor property assignment in load_dict)."""
    import gfields.io

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    gfields.io.ellipsoid_dict_to_ply(
        {
            "centers": data["centers"],
            "scales": data["scales"],
            "quaternions": data["quaternions"],
            "opacities": data["opacities"],
            "albedo": data["albedo"],
            "omega": data["omega"],
        },
        ["opacities", "albedo", "omega"],
        str(ply_path),
    )


def numpy_level_to_mesh_entry(
    data: dict[str, Any],
    *,
    visibility_mask: int,
    ply_path: Path | None = None,
) -> dict[str, Any]:
    n = data["centers"].shape[0]
    # Gabor-ness is read straight off the ``omega`` column by the kernel: a
    # primitive is a plain Gaussian exactly when ``omega == 0`` (cos(0)=1). To
    # render a level as Gaussian, zero its omega before calling this (the shape
    # no longer takes an is_gabor/kernel_type flag). See is_gabor_from_omega.
    extent = data["extent"]
    if np.asarray(extent).shape[0] != n:
        extent = np.full((n, 1), float(np.asarray(extent).reshape(-1)[0]), dtype=np.float32)
    entry: dict[str, Any] = {
        "type": "ellipsoidsmesh",
        "shell": "uv_sphere",
        "extent": _extent_mesh_value(extent, n),
        "visibility_mask": visibility_mask,
    }
    if ply_path is not None:
        write_numpy_level_ply(data, ply_path)
        entry.pop("extent", None)
        entry["filename"] = str(ply_path)
        return entry
    entry.update({
        "centers": _mesh_tensor(data["centers"], (n, 3)),
        "scales": _mesh_tensor(data["scales"], (n, 3)),
        "quaternions": _mesh_tensor(data["quaternions"], (n, 4)),
        "opacities": _mesh_tensor(data["opacities"], (n, 1)),
        "albedo": _mesh_tensor(data["albedo"], (n, 3)),
        "omega": _mesh_tensor(data["omega"], (n, 1)),
    })
    return entry


__all__ = [
    "LOCAL_FREQ_DIR",
    "orientation_vectors",
    "steerable_subband_bit",
    "OrientationBin",
    "generate_orientation_bins",
    "compute_visibility_mask",
    "quaternion_to_rotation_matrix",
    "compute_world_frequency_direction",
    "double_spaced",
    "equal_partitioning",
    "load_pyramid_metadata",
    "load_pyramid_target_levels",
    "is_gabor_from_omega",
    "is_gabor_for_level",
    "compute_visual_aabb_from_levels",
    "subset_numpy_level",
    "load_level_numpy",
    "write_numpy_level_ply",
    "numpy_level_to_mesh_entry",
]
