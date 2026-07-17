# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Asset loading + scene-dict construction for the gfields plugin.

This module owns the public functions used by every rendering CLI:

* ``AssetInfo`` / ``SteerableAssetInfo`` dataclasses
* ``detect_asset_info`` / ``load_steerable_asset`` (steerable detection lives in
  ``gfields.io.is_steerable_asset``)
* ``load_pyramid_level`` -- build an ``ellipsoidsmesh`` scene-dict entry
* ``create_integrator_config`` -- build a hierarchical PRB or tomography integrator
* ``create_environment_config`` -- envmap or constant emitter setup
* ``envmap_to_world_transform`` -- shared transform helper
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Asset detection / loading
# ---------------------------------------------------------------------------


def get_asset_name(asset_path: str) -> str:
    """Extract a clean asset name from an asset path.

    Normalises separators and strips trailing slashes; spaces become underscores.
    """
    name = os.path.basename(os.path.normpath(asset_path))
    return name.replace(" ", "_")


@dataclass
class AssetInfo:
    """Information about a Volumetric Primitives asset (pyramid-style)."""

    path: str
    npy_folder: str
    num_levels: int
    has_omega: Dict[int, bool]
    has_albedo: Dict[int, bool]
    num_primitives: Dict[int, int]
    extents: Dict[int, Any]  # float or np.ndarray (per-primitive)
    is_steerable: bool = False
    ply_folder: Optional[str] = None  # canonical .ply store (data/); None -> legacy npy
    # Per-level boolean keep-masks from a render-time ``--refine`` prune pass.
    # Empty unless pruning is active; consumed by the render CLIs as a
    # ``primitive_subset`` when building each level's scene entry.
    prune_masks: Dict[int, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D401
        extent_summary: Dict[int, Any] = {}
        for lvl, ext in self.extents.items():
            if isinstance(ext, np.ndarray):
                extent_summary[lvl] = f"array(n={len(ext)}, mean={ext.mean():.2f})"
            else:
                extent_summary[lvl] = ext
        return (
            f"AssetInfo(path={self.path}, levels={self.num_levels}, "
            f"primitives={self.num_primitives}, extents={extent_summary})"
        )


@dataclass
class SteerableAssetInfo:
    """Information about a Steerable Pyramid asset."""

    path: str
    npy_folder: str
    num_frequency_levels: int
    num_orientations: int
    total_subbands: int
    primitives_dict: Dict[str, Any]
    is_steerable: bool = True

    def __repr__(self) -> str:  # noqa: D401
        total_prims = sum(p["centers"].shape[0] for p in self.primitives_dict.values())
        return (
            f"SteerableAssetInfo(path={self.path}, "
            f"freq_levels={self.num_frequency_levels}, "
            f"orientations={self.num_orientations}, "
            f"subbands={self.total_subbands}, "
            f"total_primitives={total_prims})"
        )


def detect_asset_info(asset_path: str | os.PathLike[str]) -> AssetInfo:
    """Auto-detect a pyramid asset's structure (num levels, primitive counts, extents).

    Reads the unified per-level store ``<asset>/data/root.primitives_pyr*.ply``.
    Legacy per-array ``npy_data/*.npy`` assets must be converted first with
    ``gfields migrate assets``.
    """
    asset_path = str(asset_path)
    import gfields.io

    # Canonical layout: one .ply per pyramid level under <asset>/data/ (written
    # by gfields.io.dict_to_asset). Preferred over the legacy per-array npy.
    ply_folder = os.path.join(asset_path, "data")
    ply_levels = (gfields.io.ply_asset_levels(ply_folder)
                  if os.path.isdir(ply_folder) else [])
    if ply_levels:
        num_levels = 1 + max(ply_levels)
        has_omega, has_albedo, num_primitives, extents = {}, {}, {}, {}
        for level in range(num_levels):
            a = gfields.io.load_ellipsoid_ply(
                os.path.join(ply_folder, f"root.primitives_pyr{level}.ply"))
            num_primitives[level] = len(a["centers"])
            has_omega[level] = "omega" in a
            has_albedo[level] = "albedo" in a
            ext = a.get("extent")
            extents[level] = (ext if (ext is not None and np.ndim(ext) > 0 and np.size(ext) > 1)
                              else (float(np.ravel(ext)[0]) if ext is not None else 3.0))
        return AssetInfo(
            path=asset_path, npy_folder=os.path.join(asset_path, "npy_data"),
            ply_folder=ply_folder, num_levels=num_levels, has_omega=has_omega,
            has_albedo=has_albedo, num_primitives=num_primitives, extents=extents)

    # Deprecated fallback: read the legacy per-array npy_data/ layout directly.
    npy_folder = os.path.join(asset_path, "npy_data")
    npy_levels = (gfields.io.npy_asset_levels(npy_folder)
                  if os.path.isdir(npy_folder) else [])
    if npy_levels:
        num_levels = 1 + max(npy_levels)
        has_omega, has_albedo, num_primitives, extents = {}, {}, {}, {}
        for level in range(num_levels):
            a = gfields.io.load_ellipsoid_npy_level(npy_folder, level)
            num_primitives[level] = len(a["centers"])
            has_omega[level] = "omega" in a
            has_albedo[level] = "albedo" in a
            ext = a.get("extent")
            extents[level] = (ext if (ext is not None and np.ndim(ext) > 0 and np.size(ext) > 1)
                              else (float(np.ravel(ext)[0]) if ext is not None else 3.0))
        return AssetInfo(
            path=asset_path, npy_folder=npy_folder, ply_folder=None,
            num_levels=num_levels, has_omega=has_omega, has_albedo=has_albedo,
            num_primitives=num_primitives, extents=extents)

    raise ValueError(
        f"No asset found under {asset_path}: expected data/root.primitives_pyr*.ply "
        f"or a legacy npy_data/ directory."
    )


def _ellipsoidsmesh_entry(
    *, centers, scales, quaternions, opacities, omega, albedo, extent,
    visibility_mask: int,
) -> Dict[str, Any]:
    """Pack per-primitive numpy arrays into an ``ellipsoidsmesh`` scene-dict entry.

    Shared by :func:`load_pyramid_level` and :func:`load_steerable_asset`; all
    array args are numpy (scales already applied by the caller). ``extent`` is a
    scalar or a per-primitive array (packed as a TensorXf column). Gabor vs
    Gaussian is derived from ``omega`` by the shape, so no flag is emitted.
    """
    import mitsuba as mi

    n = int(np.asarray(centers).reshape(-1, 3).shape[0])
    extent_value: Any = (mi.TensorXf(np.asarray(extent, np.float32).reshape(-1, 1))
                         if isinstance(extent, np.ndarray) else float(extent))
    return {
        "type": "ellipsoidsmesh",
        "shell": "uv_sphere",
        "centers": mi.TensorXf(np.ravel(centers), shape=(n, 3)),
        "scales": mi.TensorXf(np.ravel(scales), shape=(n, 3)),
        "quaternions": mi.TensorXf(np.ravel(quaternions), shape=(n, 4)),
        "opacities": mi.TensorXf(np.ravel(opacities), shape=(n, 1)),
        "omega": mi.TensorXf(np.ravel(omega), shape=(n, 1)),
        "albedo": mi.TensorXf(np.ravel(albedo), shape=(n, 3)),
        "extent": extent_value,
        "visibility_mask": visibility_mask,
    }


def load_steerable_asset(
    asset_path: str | os.PathLike[str],
    *,
    opacity_scale: float = 1.0,
    albedo_scale: float = 0.0,
    default_extent: float = 3.0,
    world_scale: float = 1.0,
    albedo_rgb: Optional[List[float]] = None,
) -> SteerableAssetInfo:
    """Load a steerable-pyramid asset (one Mitsuba dict per orientation sub-band)."""
    import gfields.io

    asset_dir = Path(str(asset_path))
    ply_path = gfields.io.steerable_ply_path(asset_dir)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Steerable decomposition not found: {ply_path}")

    # ``load_steerable_ply`` reconstructs the sub-band grid and its config
    # (num freq levels / orientations) straight from the per-primitive flag
    # columns -- no sidecar. ``total_subbands`` is the theoretical bit budget
    # (1 Gaussian + Gabor levels x orientations), as the retired
    # steerable_metadata.json recorded it.
    subbands, cfg = gfields.io.load_steerable_ply(ply_path)
    num_freq_levels = cfg["num_frequency_levels"]
    num_orientations = cfg["num_orientations"]
    total_subbands = 1 + max(num_freq_levels - 1, 0) * num_orientations

    print(f"Loading steerable asset: {asset_path}")
    print(f"  Frequency levels: {num_freq_levels}")
    print(f"  Orientations:     {num_orientations}")
    print(f"  Total sub-bands:  {total_subbands}")

    primitives_dict: Dict[str, Any] = {}
    total_primitives = 0
    ws = float(world_scale)

    for sub in subbands:
        freq_level = sub["freq_level"]
        subband_key = f"f{freq_level}_o{sub['orientation_index']}"
        subband_info = {"visibility_mask": int(sub["visibility_mask"])}

        centers = np.ascontiguousarray(sub["centers"], dtype=np.float32)
        scales = np.ascontiguousarray(sub["scales"], dtype=np.float32)
        quaternions = np.ascontiguousarray(sub["quaternions"], dtype=np.float32)
        opacities = np.ravel(sub["opacities"]).astype(np.float32) * opacity_scale

        if ws != 1.0:
            centers = np.ascontiguousarray(centers * ws, dtype=np.float32)
            scales = np.ascontiguousarray(scales * ws, dtype=np.float32)

        omega = np.asarray(sub["omega"]).reshape(len(centers), -1) if "omega" in sub \
            else np.zeros((len(centers), 1))

        if "extent" in sub:
            extent = np.ravel(sub["extent"]).astype(np.float32)
        else:
            extent = default_extent

        n_prims = len(centers)
        total_primitives += n_prims

        if albedo_rgb is not None:
            rgb = np.asarray(albedo_rgb, dtype=np.float32).reshape(3) * float(albedo_scale)
            albedo_values = np.tile(rgb[None, :], (n_prims, 1))
        else:
            albedo_values = np.ones((n_prims, 3), dtype=np.float32) * float(albedo_scale)

        primitives_dict[subband_key] = _ellipsoidsmesh_entry(
            centers=centers, scales=scales, quaternions=quaternions,
            opacities=opacities, omega=omega, albedo=albedo_values, extent=extent,
            visibility_mask=subband_info["visibility_mask"])

        mask_bin = f"0b{subband_info['visibility_mask']:08b}"
        prim_type = "Gabor" if freq_level > 0 else "Gaussian"
        extent_str = (
            f", extent=per-prim(mean={extent.mean():.2f})"
            if isinstance(extent, np.ndarray)
            else f", extent={extent:.2f}"
        )
        print(f"    {subband_key}: {n_prims} {prim_type} primitives, mask={mask_bin}{extent_str}")

    print(f"  Total primitives: {total_primitives}")

    return SteerableAssetInfo(
        path=str(asset_path),
        npy_folder=str(asset_dir / "data"),
        num_frequency_levels=num_freq_levels,
        num_orientations=num_orientations,
        total_subbands=total_subbands,
        primitives_dict=primitives_dict,
    )


def load_pyramid_level(
    asset_info: AssetInfo,
    level: int,
    *,
    albedo_scale: float = 1.0,
    opacity_scale: float = 1.0,
    extent: Any = 3.0,
    primitive_subset: Optional[slice] = None,
    world_scale: float = 1.0,
    albedo_rgb: Optional[List[float]] = None,
    use_asset_albedo: bool = False,
) -> Dict[str, Any]:
    """Build an ``ellipsoidsmesh`` scene-dict entry for one pyramid level."""
    num_prims = asset_info.num_primitives[level]
    import gfields.io
    if getattr(asset_info, "ply_folder", None):
        a = gfields.io.load_ellipsoid_ply(
            os.path.join(asset_info.ply_folder, f"root.primitives_pyr{level}.ply"))
    elif getattr(asset_info, "npy_folder", None):
        # Deprecated legacy per-array npy_data/ layout (emits a DeprecationWarning).
        a = gfields.io.load_ellipsoid_npy_level(asset_info.npy_folder, level)
    else:
        raise ValueError(
            f"Asset {asset_info.path} has no data/*.ply store or legacy npy_data/ dir.")
    centers, scales, quaternions = a["centers"], a["scales"], a["quaternions"]
    # Opacity column name varies by producer (regression -> 'opacities',
    # optimize_hierarchical -> 'sigma_t'); accept either.
    opacity_arr = next((a[k] for k in ("opacities", "sigma_t", "sigmat") if k in a), None)
    if opacity_arr is None:
        raise KeyError(
            f"No opacity column (opacities/sigma_t/sigmat) in "
            f"root.primitives_pyr{level}.ply for {asset_info.path}")
    opacities = np.ravel(opacity_arr).astype(np.float32)
    omega = np.ravel(a["omega"]).astype(np.float32) if "omega" in a else np.zeros(num_prims, dtype=np.float32)
    asset_albedo = a.get("albedo")

    if albedo_rgb is not None:
        rgb = np.asarray(albedo_rgb, dtype=np.float32).reshape(3)
        albedo = np.tile(rgb[None, :], (num_prims, 1))
    elif use_asset_albedo and asset_albedo is not None:
        albedo = np.ascontiguousarray(asset_albedo, dtype=np.float32).reshape(num_prims, 3)
    else:
        albedo = np.ones((num_prims, 3), dtype=np.float32)

    if primitive_subset is not None:
        centers = np.ascontiguousarray(centers[primitive_subset])
        scales = np.ascontiguousarray(scales[primitive_subset])
        quaternions = np.ascontiguousarray(quaternions[primitive_subset])
        opacities = np.ascontiguousarray(opacities.flatten()[primitive_subset])
        omega = np.ascontiguousarray(omega.flatten()[primitive_subset])
        albedo = np.ascontiguousarray(albedo.reshape(-1, 3)[primitive_subset])

    ws = float(world_scale)
    if ws != 1.0:
        centers = np.ascontiguousarray(centers * ws, dtype=np.float32)
        scales = np.ascontiguousarray(scales * ws, dtype=np.float32)

    n_out = int(centers.shape[0])

    if isinstance(extent, np.ndarray):
        ext = np.asarray(extent, dtype=np.float32).reshape(-1)
        if ext.size != n_out and primitive_subset is not None:
            ext = np.ascontiguousarray(ext[primitive_subset])
        if ext.size != n_out:
            raise ValueError(
                f"extent length {ext.size} does not match {n_out} primitives for level {level}"
            )
        extent_out: Any = ext
    else:
        extent_out = float(extent)

    return _ellipsoidsmesh_entry(
        centers=centers, scales=scales, quaternions=quaternions,
        opacities=opacities * opacity_scale, omega=omega,
        albedo=albedo * albedo_scale, extent=extent_out,
        visibility_mask=2 ** level)


# ---------------------------------------------------------------------------
# Integrator + environment configuration
# ---------------------------------------------------------------------------

# Canonical presets for interactive viewing and the example render scripts.
MS_INTEGRATOR_DEFAULTS: Dict[str, Any] = {
    "max_depth": 10,
    "rr_depth": 10,
    "rr_nee_depth": 64,
    "stochastic_selection_criteria": "deterministic",
    "stochastic_selection_criteria_nee": "deterministic",
    "solver_type": "bisection",
    "solver_max_iterations": 4,
    "max_overlaps": 256,
}
MS_ALBEDO_DEFAULT = 0.8
TOMOGRAPHY_MAX_DEPTH_DEFAULT = -1


def apply_asset_render_defaults(args) -> None:
    """Fill integrator/asset defaults that depend on tomography vs MS mode."""
    ms = bool(getattr(args, 'multiple_scattering', False))
    if getattr(args, 'max_depth', None) is None:
        args.max_depth = (MS_INTEGRATOR_DEFAULTS['max_depth'] if ms
                          else TOMOGRAPHY_MAX_DEPTH_DEFAULT)
    if ms:
        if getattr(args, 'rr_depth', None) is None:
            args.rr_depth = MS_INTEGRATOR_DEFAULTS['rr_depth']
        if getattr(args, 'rr_nee_depth', None) is None:
            args.rr_nee_depth = MS_INTEGRATOR_DEFAULTS['rr_nee_depth']
        if getattr(args, 'solver_max_iterations', None) is None:
            args.solver_max_iterations = MS_INTEGRATOR_DEFAULTS[
                'solver_max_iterations']
        if getattr(args, 'albedo', None) is None:
            args.albedo = MS_ALBEDO_DEFAULT
    elif getattr(args, 'albedo', None) is None:
        args.albedo = 0.0


def create_integrator_config(
    *,
    multiple_scattering: bool,
    kernel_type: str = "gabor_gaussian_mixture",
    max_depth: Optional[int] = None,
    pyramid_level_count: int = 1,
    stochastic_selection_criteria: str = MS_INTEGRATOR_DEFAULTS[
        "stochastic_selection_criteria"],
    stochastic_selection_criteria_nee: str = MS_INTEGRATOR_DEFAULTS[
        "stochastic_selection_criteria_nee"],
    power_law_beta: float = 0.8,
    power_law_beta_nee: float = 0.8,
    rr_depth: int = MS_INTEGRATOR_DEFAULTS["rr_depth"],
    rr_nee_depth: int = MS_INTEGRATOR_DEFAULTS["rr_nee_depth"],
    max_overlaps: int = MS_INTEGRATOR_DEFAULTS["max_overlaps"],
    rr_depth_primitive: int = 1024,
    max_depth_primitive: int = 2048,
    solver_type: str = "bisection",
    solver_max_iterations: int = 4,
    asset_encoding: str = "simple",
    num_orientations: int = 3,
    orientation_selection: str = "deterministic",
    orientation_selection_nee: str = "deterministic",
    orientation_threshold: float = 0.7,
    orientation_decay_rate: float = 1.5,
    importance_mix_weight: float = 0.5,
    bin_rotation: Optional[List[float]] = None,
    isotropic: bool = True,
    stochastic_sampling: bool = False,
    nee_transmittance: str = "analytic",
    use_anyhit: Optional[bool] = None,
    use_anyhit_sampling: Optional[bool] = None,
    truncate_at_shell: bool = False,
    nee_truncate_at_shell: Optional[bool] = None,
    use_kernel_freeze: bool = False,
    residual_weight_clamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Build an integrator scene-dict entry.

    ``multiple_scattering=True`` selects ``gfields_prb`` (the unified PRB
    integrator); ``False`` selects ``gfields_tomography``.
    ``stochastic_sampling=True`` selects ``gfields_prb_stochastic`` instead of
    ``gfields_prb`` (decomposition-tracking distance sampling, see
    docs/prb_stochastic.md); ``nee_transmittance`` picks its shadow-ray
    estimator ('analytic': deterministic optical-depth traversal, best at
    equal variance; 'stochastic': per-primitive Bernoulli tests).
    ``use_anyhit`` / ``use_anyhit_sampling`` override the stochastic
    integrator's any-hit fast paths (None keeps the plugin defaults: NEE
    on, sampling off); for tomography, ``use_anyhit`` opts into the
    primal-only OptiX any-hit traversal (adjoint pass is unaffected).
    ``use_kernel_freeze`` enables the tomography integrator's internal
    ``@dr.freeze`` replay. ``residual_weight_clamp`` caps the stochastic
    integrator's envelope delta-tracking null-weight amplification (None
    keeps the plugin default of 1.0).
    """
    import mitsuba as mi

    if max_depth is None:
        max_depth = (MS_INTEGRATOR_DEFAULTS["max_depth"] if multiple_scattering
                     else TOMOGRAPHY_MAX_DEPTH_DEFAULT)

    if multiple_scattering:
        config: Dict[str, Any] = {
            # gfields_prb_stochastic is Gaussian-only (it errors on Gabor/mixture
            # assets -- see the integrator); gfields_prb is the general path.
            "type": ("gfields_prb_stochastic" if stochastic_sampling
                     else "gfields_prb"),
            "max_depth": max_depth,
            "kernel_normalized": True,
            "kernel_type": kernel_type,
            "pyramid_level_count": pyramid_level_count,
            "asset_encoding": asset_encoding,
            "stochastic_selection_criteria": stochastic_selection_criteria,
            "power_law_beta": power_law_beta,
            "stochastic_selection_criteria_nee": stochastic_selection_criteria_nee,
            "power_law_beta_nee": power_law_beta_nee,
            "solver_type": solver_type,
            "solver_max_iterations": solver_max_iterations,
            "max_overlaps": max_overlaps,
            "rr_depth_primitive": rr_depth_primitive,
            "max_depth_primitive": max_depth_primitive,
            "use_nee": True,
            "num_orientations": num_orientations,
            "orientation_selection": orientation_selection,
            "orientation_selection_nee": orientation_selection_nee,
            "orientation_threshold": orientation_threshold,
            "orientation_decay_rate": orientation_decay_rate,
            "importance_mix_weight": importance_mix_weight,
            "phasefunction": {"type": "isotropic"} if isotropic else {"type": "hg", "g": 0.857},
        }
        # NEE origin-overlap support convention (both integrators; see
        # docs/prb_stochastic.md §1.4). None keeps the plugin default (off for
        # gfields_prb; follows truncate_at_shell for gfields_prb_stochastic).
        if nee_truncate_at_shell is not None:
            config["nee_truncate_at_shell"] = nee_truncate_at_shell
        if stochastic_sampling:
            config["nee_transmittance"] = nee_transmittance
            # Shell-truncated distance sampler (Gaussian-only path); see §1.3.
            config["truncate_at_shell"] = truncate_at_shell
            if use_anyhit is not None:
                config["use_anyhit"] = use_anyhit
            if use_anyhit_sampling is not None:
                config["use_anyhit_sampling"] = use_anyhit_sampling
            if residual_weight_clamp is not None:
                config["residual_weight_clamp"] = residual_weight_clamp
        if rr_depth > -1:
            config["rr_depth"] = rr_depth
        if rr_nee_depth > -1:
            config["rr_nee_depth"] = rr_nee_depth
        if bin_rotation is not None:
            config["bin_rotation"] = mi.ScalarQuaternion4f(*bin_rotation)
        return config

    tomography_config: Dict[str, Any] = {
        "type": "gfields_tomography",
        "max_depth": max_depth,
        "kernel_normalized": True,
        "kernel_full_range": True,
        "kernel_type": kernel_type,
        # LOD parameters -- WITHOUT these the LODSelector defaults to
        # pyramid_level_count=1, so its per-ray visibility mask collapses to bit 0
        # (the Gaussian base) and the Gabor levels (visibility_mask=2**level) are
        # never traversed. These must mirror the multiple-scattering config above
        # for the stochastic-Laplacian / orientation-selection strategies to work.
        "pyramid_level_count": pyramid_level_count,
        "asset_encoding": asset_encoding,
        "stochastic_selection_criteria": stochastic_selection_criteria,
        "stochastic_selection_criteria_nee": stochastic_selection_criteria_nee,
        "power_law_beta": power_law_beta,
        "power_law_beta_nee": power_law_beta_nee,
        "num_orientations": num_orientations,
        "orientation_selection": orientation_selection,
        "orientation_selection_nee": orientation_selection_nee,
        "orientation_threshold": orientation_threshold,
        "orientation_decay_rate": orientation_decay_rate,
        "importance_mix_weight": importance_mix_weight,
    }
    if bin_rotation is not None:
        tomography_config["bin_rotation"] = mi.ScalarQuaternion4f(*bin_rotation)
    if use_kernel_freeze:
        tomography_config["use_kernel_freeze"] = True
    if use_anyhit:
        tomography_config["use_anyhit"] = True
    return tomography_config


def envmap_to_world_transform(
    spin_y_deg: float,
    envmap_rotation_x: float,
    envmap_rotation_y: float,
    envmap_rotation_z: float,
):
    """World transform for envmap emitters (shared by static + video renders)."""
    import mitsuba as mi

    T = mi.ScalarTransform4f
    total_y = 180.0 + spin_y_deg + envmap_rotation_y
    return (
        T.rotate([0, 1, 0], total_y)
        @ T.rotate([1, 0, 0], envmap_rotation_x)
        @ T.rotate([0, 0, 1], envmap_rotation_z)
    )


def create_environment_config(
    *,
    multiple_scattering: bool,
    envmap_path: Optional[str] = None,
    envmap_scale: float = 0.35,
    rotation_angle: float = 0.0,
    envmap_rotation_x: float = 0.0,
    envmap_rotation_y: float = 0.0,
    envmap_rotation_z: float = 0.0,
) -> Dict[str, Any]:
    """Build an emitter scene-dict entry (envmap when MS + path provided, else constant)."""
    if not multiple_scattering or envmap_path is None:
        return {"type": "constant"}
    return {
        "type": "envmap",
        "filename": envmap_path,
        "scale": envmap_scale,
        "to_world": envmap_to_world_transform(
            rotation_angle, envmap_rotation_x, envmap_rotation_y, envmap_rotation_z
        ),
    }


# ---------------------------------------------------------------------------
# Pyramid-asset plumbing (numpy side): path resolution, band classification,
# per-level numpy loading and ellipsoidsmesh entry / PLY export. Shared by the
# rendering CLIs (render.py), the animation loops and the interactive viewer.
# ---------------------------------------------------------------------------

def _is_asset_root(p: Path) -> bool:
    """True if ``p`` is itself an asset directory (not a parent of pyramid levels).

    This is just to support a legacy asset layout before the public code release
    implemented proper PLY file handling. I'll just keep it here for now but will
    remove once I am sure we have migrated all assets to the new layout - Jorge

    """
    if (p / "npy_data").is_dir():
        return True
    data_dir = p / "data"
    if not data_dir.is_dir():
        return False
    return (any(data_dir.glob("root.primitives_pyr*.ply"))
            or (data_dir / "root.primitives_steerable.ply").exists())


def resolve_asset_path(asset_path: str, *, asset_pyr: int | None = None) -> Path:
    p = Path(asset_path)
    if not p.exists():
        raise FileNotFoundError(f"Asset path not found: {asset_path}")
    if _is_asset_root(p):
        return p
    optimized = sorted(
        p.glob("optimized_asset_pyr*"),
        key=lambda path: int(re.search(r"pyr(\d+)", path.name).group(1)),
    )
    if optimized:
        if asset_pyr is not None:
            chosen = p / f"optimized_asset_pyr{asset_pyr}"
            if not chosen.is_dir():
                available = [d.name for d in optimized]
                raise FileNotFoundError(
                    f"optimized_asset_pyr{asset_pyr} not found under {asset_path}. "
                    f"Available: {available}"
                )
            return chosen
        return optimized[-1]
    raise FileNotFoundError(
        f"No data/root.primitives_pyr*.ply, data/root.primitives_steerable.ply, "
        f"npy_data, or optimized_asset_pyr* folder under {asset_path}"
    )

__all__ = [
    "AssetInfo",
    "SteerableAssetInfo",
    "detect_asset_info",
    "load_steerable_asset",
    "load_pyramid_level",
    "create_integrator_config",
    "MS_INTEGRATOR_DEFAULTS",
    "MS_ALBEDO_DEFAULT",
    "TOMOGRAPHY_MAX_DEPTH_DEFAULT",
    "apply_asset_render_defaults",
    "envmap_to_world_transform",
    "create_environment_config",
    "resolve_asset_path",
]
