# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Offline asset decomposition into LOD pyramids.

Two decompositions, both bounded by OptiX's 8-bit visibility mask and sharing
the orientation/quaternion and band-partitioning helpers in
:mod:`gfields.lod.utils`:

* **steerable** (:func:`decompose_asset`, ``gfields decompose steerable``) splits
  each Gabor level into orientation sub-bands -- one bin per Gabor by alignment
  between the bin direction and the Gabor's world-space frequency direction. A
  bit-encoded ``visibility_mask`` keeps each sub-band addressable.
* **laplacian** (:func:`lod_partition_main`, ``gfields decompose lod-partition``)
  splits Gabor (or Gaussian) primitives into frequency-band LOD levels by an
  effective-frequency criterion, with optional adaptive per-primitive extent and
  peak-density pruning.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from gfields.lod.utils import (
    compute_visibility_mask,
    compute_world_frequency_direction,
    double_spaced,
    equal_partitioning,
    generate_orientation_bins,
    is_gabor_from_omega,
)


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------


def _np(v: Any) -> Optional[np.ndarray]:
    """Detach a drjit/Mitsuba tensor (or ``None``) to numpy."""
    if v is None:
        return None
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def _extent_array(extent: Any, n: int) -> np.ndarray:
    if hasattr(extent, "numpy"):
        return extent.numpy().flatten()
    if isinstance(extent, np.ndarray):
        return extent.flatten()
    return np.full(n, float(extent))


# ---------------------------------------------------------------------------
# Steerable orientation decomposition
# ---------------------------------------------------------------------------


def _load_merged_levels(asset_info, asset_levels, load_pyramid_level, *, albedo_scale, opacity_scale):
    """Load one or more asset levels and concatenate them into a single arrays dict."""
    keys = ("centers", "scales", "quaternions", "opacities")
    acc: Dict[str, list] = {k: [] for k in (*keys, "omega", "extent")}
    for lvl in asset_levels:
        data = load_pyramid_level(
            asset_info, lvl,
            albedo_scale=albedo_scale, opacity_scale=opacity_scale,
            extent=asset_info.extents.get(lvl, 3.0),
        )
        n = len(data["centers"])
        for k in keys:
            acc[k].append(np.asarray(data[k]))
        omega = np.asarray(data.get("omega", np.zeros((n, 1))))
        acc["omega"].append(omega.reshape(-1, 1) if omega.ndim == 1 else omega)
        acc["extent"].append(_extent_array(data.get("extent", 3.0), n))
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def _bin_by_orientation(quaternions, orientation_bins, num_orientations):
    """Assign each primitive to the best-aligned orientation bin.

    Returns ``{ori_idx: index_array}``. Alignment is ``|world_freq_dir . bin_dir|``,
    argmax over bins (first bin wins ties, matching the runtime selector).
    """
    if len(quaternions) == 0:
        return {ori: np.empty(0, dtype=np.int64) for ori in range(num_orientations)}
    dirs = np.stack([compute_world_frequency_direction(q) for q in quaternions])
    bin_dirs = np.stack([b.direction for b in orientation_bins])
    best = np.argmax(np.abs(dirs @ bin_dirs.T), axis=1)
    return {ori: np.nonzero(best == ori)[0] for ori in range(num_orientations)}


def decompose_asset(
    asset_path: str,
    output_path: str,
    *,
    num_orientations: int = 3,
    num_freq_levels: Optional[int] = None,
    opacity_scale: float = 1.0,
    albedo_scale: float = 0.0,
) -> Dict[str, Any]:
    """Decompose a pyramid asset into steerable orientation sub-bands.

    Writes a single self-describing ``<output_path>/data/root.primitives_steerable.ply``
    (each primitive carries its own visibility_mask / freq_level / orientation_index)
    and returns a metadata dict describing the decomposition for programmatic callers.
    """
    import gfields.io
    from gfields.rendering.pipeline import detect_asset_info, load_pyramid_level

    asset_info = detect_asset_info(asset_path)
    if num_freq_levels is None:
        num_freq_levels = asset_info.num_levels
    else:
        num_freq_levels = min(num_freq_levels, asset_info.num_levels)

    total_subbands = 1 + (num_freq_levels - 1) * num_orientations
    if total_subbands > 8:
        raise ValueError(
            f"Configuration exceeds 8-bit mask limit: {num_freq_levels} freq x "
            f"{num_orientations} ori = {total_subbands} sub-bands (max 8)"
        )

    orientation_bins = generate_orientation_bins(num_orientations)
    prim_keys = ("centers", "scales", "quaternions", "opacities", "omega", "extent")

    subbands: list[Dict[str, Any]] = []
    stats: Dict[Any, Dict[str, Any]] = {}

    for freq_level in range(num_freq_levels):
        # Merge any asset levels beyond the last steerable level into it.
        if freq_level == num_freq_levels - 1 and num_freq_levels < asset_info.num_levels:
            asset_levels = list(range(freq_level, asset_info.num_levels))
        else:
            asset_levels = [freq_level]
        prims = _load_merged_levels(
            asset_info, asset_levels, load_pyramid_level,
            albedo_scale=albedo_scale, opacity_scale=opacity_scale)

        if freq_level == 0:
            # Gaussian base: a single sub-band, no orientation split.
            mask = compute_visibility_mask(0, -1, num_orientations)
            subbands.append({**prims, "visibility_mask": mask,
                             "freq_level": 0, "orientation_index": -1})
            stats[(0, -1)] = {"num_primitives": len(prims["centers"]), "visibility_mask": mask}
            continue

        for ori_idx, sel in _bin_by_orientation(prims["quaternions"], orientation_bins, num_orientations).items():
            if sel.size == 0:
                continue
            mask = compute_visibility_mask(freq_level, ori_idx, num_orientations)
            subbands.append({
                **{k: prims[k][sel] for k in prim_keys},
                "visibility_mask": mask, "freq_level": freq_level, "orientation_index": ori_idx,
            })
            stats[(freq_level, ori_idx)] = {"num_primitives": int(sel.size), "visibility_mask": mask}

    ply_path = gfields.io.steerable_ply_path(Path(output_path))
    gfields.io.save_steerable_ply(subbands, ply_path)

    total = sum(v["num_primitives"] for v in stats.values())
    print(f"Steerable decomposition: {total} primitives, {len(stats)} sub-bands -> {ply_path}")

    return {
        "source_asset": asset_path,
        "num_frequency_levels": num_freq_levels,
        "num_orientations": num_orientations,
        "total_subbands": total_subbands,
        "orientation_bins": [{"index": b.index, "direction": b.direction.tolist()} for b in orientation_bins],
        "subbands": {f"f{k[0]}_o{k[1]}": v for k, v in stats.items()},
    }


def _steerable_setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose a volumetric asset into steerable pyramid sub-bands")
    parser.add_argument("--asset", required=True, help="Path to input asset folder")
    parser.add_argument("--output", required=True, help="Path to output steerable asset folder")
    parser.add_argument("--num_orientations", type=int, default=3,
                        help="Number of orientation bins per Gabor level (default: 3)")
    parser.add_argument("--num_freq_levels", type=int, default=None,
                        help="Number of frequency levels (default: use all from the asset). "
                             "For 7 orientations use 2 levels (1 Gaussian + 1 Gabor).")
    parser.add_argument("--opacity_scale", type=float, default=1.0, help="Scale factor for opacities")
    parser.add_argument("--albedo_scale", type=float, default=0.0, help="Scale factor for albedo")
    return parser.parse_args()


def steerable_main() -> None:
    """CLI: ``gfields decompose steerable`` -- split a pyramid asset's Gabor
    levels into orientation sub-bands (one combined self-describing .ply)."""
    args = _steerable_setup()
    decompose_asset(
        asset_path=args.asset,
        output_path=args.output,
        num_orientations=args.num_orientations,
        num_freq_levels=args.num_freq_levels,
        opacity_scale=args.opacity_scale,
        albedo_scale=args.albedo_scale,
    )


# ---------------------------------------------------------------------------
# Frequency-band LOD partitioning
# ---------------------------------------------------------------------------


def render_decomposition(
    decomposed_model: Dict[str, Any],
    cam_array,
    *,
    spp: int,
    n_bands: int,
    output_path,
) -> None:
    """Render an LoD pyramid by progressively dropping the highest-frequency level."""
    import drjit as dr
    import mitsuba as mi
    from tqdm import tqdm

    decomposed_scene = mi.load_dict(decomposed_model)
    level_imgs = []
    for i in tqdm(list(range(n_bands))[::-1], desc="Rendering LoD levels"):
        level_imgs.append(mi.render(decomposed_scene, sensor=cam_array, spp=spp, seed=42))
        decomposed_model.pop(f"primitives_pyr{i}", None)
        decomposed_scene = mi.load_dict(decomposed_model)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(
        str(Path(output_path) / f"pyramid_{n_bands}.exr"),
        dr.concat(level_imgs, axis=0),
    )


def _lod_partition_setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to Gauss or Gabor checkpoint")
    parser.add_argument("--output", "-o", type=str, default="lod_checkpoints", help="Path to output directory")
    parser.add_argument("--output_levels", "-nl", type=int, default=4, help="number of levels to decompose mixture into")
    parser.add_argument("--num_cams", "-nc", type=int, help="number of cams to render")
    parser.add_argument("--cam_res_x", "-cx", type=int, help="width of rendered images")
    parser.add_argument("--cam_res_y", "-cy", type=int, help="height of rendered images")
    parser.add_argument("--same_number_of_prims_per_level", "-equal", action="store_true",
                        help="use an equal number of primitives per level for the partition criteria")
    parser.add_argument("--dot_b", action="store_true",
                        help="use the sum instead of the norm for the Gabor partition criteria")
    parser.add_argument("--spp", type=int, help="spp when rendering images")

    # Adaptive extent parameters
    parser.add_argument("--adaptive_extent", "-ae", action="store_true",
                        help="Enable adaptive extent computation based on kernel contribution threshold")
    parser.add_argument("--extent_threshold", type=float, default=1e-4,
                        help="Minimum kernel contribution at boundary (default 1e-4)")
    parser.add_argument("--min_extent", type=float, default=0.1,
                        help="Minimum extent value to prevent degeneracy")
    parser.add_argument("--base_extent", type=float, default=3.0,
                        help="Maximum/base extent value (training setup)")
    parser.add_argument("--envelope_only", action="store_true",
                        help="Ignore frequency term, use only the Gaussian envelope for extent")

    # Pruning parameters
    parser.add_argument("--prune_threshold", type=float, default=None,
                        help="Prune primitives with peak density below this threshold (default: disabled)")
    parser.add_argument("--prune_include_frequency", action="store_true",
                        help="Include frequency decay factor when computing peak density for pruning")

    args = parser.parse_args()
    args.path = Path(args.path)
    return args


def _lod_partition_prepare(path: Path, num_cams=None, res_x=None, res_y=None):
    import gfields.training.regression as gu
    import mitsuba as mi

    with open(str(path / "args.json"), "r") as f:
        params = json.load(f)

    num_cams = num_cams or params["cam_count"]
    res_x = res_x or params["cam_res_x"]
    res_y = res_y or params["cam_res_y"]

    model = gu.load_model(path, -1)
    scene: mi.Scene = mi.load_dict(model)
    cam_array = gu.create_cam_array(gu.prepare_cameras(num_cams, res_x, res_y, 1.0))

    has_gabor = "primitives_pyr1" in model
    assert "primitives_pyr2" not in model, (
        "More than 2 levels are available. This is not supported and likely a bug, "
        "as this model might already be decomposed.")

    return params, model, scene, cam_array, has_gabor


# --- shared level plumbing (Gabor + Gaussian) ------------------------------


def _add_level(decomposed_model, level_primitives, source, mask, level):
    """Extract the ``mask`` subset of ``source`` as pyramid level ``level``.

    Gabor vs Gaussian is carried entirely by the ``omega`` column: Gaussian
    levels have omega==0 (or no omega), so nothing else marks the kernel type."""
    import mitsuba as mi
    import gfields.training.regression as gu

    level_prims = {k: mi.TensorXf(v.numpy()[mask]) for k, v in source.items()
                   if isinstance(v, mi.TensorXf) and k != "extent"}
    level_primitives[level] = {
        "scales": level_prims["scales"].numpy(),
        "omega": level_prims["omega"].numpy() if "omega" in level_prims else None,
        "opacities": level_prims["opacities"].numpy() if "opacities" in level_prims else None,
    }
    entry = gu.create_scene_dict_from_primitives(-1, level, **level_prims)[f"primitives_pyr{level}"]
    decomposed_model[f"primitives_pyr{level}"] = entry


def _apply_adaptive_extents(level_primitives, decomposed_model, args) -> Dict[int, np.ndarray]:
    """Compute per-primitive extents per level and write them onto the model."""
    if not args.adaptive_extent:
        return {}
    import mitsuba as mi
    from gfields.training.adaptive import clamp_extents

    all_extents: Dict[int, np.ndarray] = {}
    for lvl, prims in sorted(level_primitives.items()):
        is_gabor_level = is_gabor_from_omega(prims.get("omega"))
        extents = clamp_extents(
            scales=prims["scales"],
            omega=prims.get("omega"),
            opacities=prims.get("opacities"),
            base_extent=args.base_extent,
            min_extent=args.min_extent,
            threshold=args.extent_threshold,
            envelope_only=args.envelope_only or not is_gabor_level,
        )
        all_extents[lvl] = extents
        key = f"primitives_pyr{lvl}"
        if key in decomposed_model:
            decomposed_model[key]["extent"] = mi.TensorXf(extents.reshape(-1, 1))
    return all_extents


def _apply_pruning(level_primitives, decomposed_model, all_extents, args) -> None:
    """Drop primitives whose peak density falls below ``args.prune_threshold``."""
    if args.prune_threshold is None:
        return
    import mitsuba as mi
    from gfields.training.adaptive import compute_pruning_mask

    for lvl in list(level_primitives.keys()):
        prims = level_primitives[lvl]
        is_gabor_level = is_gabor_from_omega(prims.get("omega"))
        keep = compute_pruning_mask(
            scales=prims["scales"],
            opacities=prims.get("opacities"),
            omega=prims.get("omega") if is_gabor_level else None,
            threshold=args.prune_threshold,
            include_frequency=args.prune_include_frequency and is_gabor_level,
        )
        key = f"primitives_pyr{lvl}"
        n_before, n_kept = len(keep), int(np.sum(keep))
        print(f"Level {lvl}: pruned {n_before - n_kept}/{n_before}")

        if n_kept == 0:
            del level_primitives[lvl]
            decomposed_model.pop(key, None)
            all_extents.pop(lvl, None)
            continue

        level_primitives[lvl] = {
            "scales": prims["scales"][keep],
            "omega": prims["omega"][keep] if prims.get("omega") is not None else None,
            "opacities": prims["opacities"][keep] if prims.get("opacities") is not None else None,
        }
        if lvl in all_extents:
            all_extents[lvl] = all_extents[lvl][keep]
        if key in decomposed_model:
            new_prims = {k: (mi.TensorXf(v.numpy()[keep]) if isinstance(v, mi.TensorXf) else v)
                         for k, v in decomposed_model[key].items()}
            if args.adaptive_extent and lvl in all_extents:
                new_prims["extent"] = mi.TensorXf(all_extents[lvl].reshape(-1, 1))
            decomposed_model[key] = new_prims


def _finalize_lod(decomposed_model, scene, cam_array, output_path, n_bands, spp) -> None:
    """Save the decomposed model, report PSNR vs the original, and dump the LoD pyramid."""
    import mitsuba as mi
    import gfields.training.regression as gu
    from gfields.losses import psnr_frozen

    output_path.mkdir(parents=True, exist_ok=True)
    decomposed_scene = mi.load_dict(decomposed_model)
    gu.save_model(output_path, decomposed_scene, n_bands)

    img_dec = mi.render(decomposed_scene, sensor=cam_array, spp=spp, seed=42)
    img = mi.render(scene, sensor=cam_array, spp=spp, seed=42)
    print(f"PSNR vs original: {psnr_frozen(img, img_dec).array[0]:.2f} dB")
    mi.util.write_bitmap(str(output_path / f"optimized_pyr{n_bands}.exr"), img_dec)

    # Per-primitive extents ride on the primitives and are packed into the
    # unified data/root.primitives_pyr*.ply by save_model (no extent_pyr*.npy).
    render_decomposition(decomposed_model, cam_array, spp=spp, n_bands=n_bands, output_path=output_path)


# --- the two partition modes ----------------------------------------------


def _lod_partition_gabor(args, model, scene, cam_array, grid, volume_config, a, b, gauss_target_level, n_bands, output_path):
    import gfields.training.regression as gu

    angular_nyquist = np.max(gu.compute_max_freq(grid.shape, volume_config.volume_scale))
    angular_gauss_cutoff = gu.angular_cutoff_frequency(gauss_target_level, a, b, eps=0.25)

    decomposed_model = deepcopy(model)
    gabor = decomposed_model.pop("primitives_pyr1")  # current Gabor level, re-split below
    gauss_pyr0 = decomposed_model.get("primitives_pyr0")

    if args.dot_b:
        criteria = np.sum(gabor["omega"] / gabor["scales"], axis=-1)
    else:
        criteria = np.linalg.norm(gabor["omega"] / gabor["scales"], axis=-1)

    if args.same_number_of_prims_per_level:
        lod_bounds = equal_partitioning(criteria, n_bands)
    else:
        lod_bounds = double_spaced(angular_gauss_cutoff, angular_nyquist, n_bands)
    lod_bounds = np.asarray(lod_bounds, dtype=float)
    lod_bounds[0], lod_bounds[-1] = 0.0, np.inf

    level_primitives: Dict[int, dict] = {}
    # The Gaussian base (pyr0) stays in the model but also joins adaptive extent / pruning.
    if gauss_pyr0 is not None and gauss_pyr0.get("scales") is not None:
        level_primitives[0] = {
            "scales": _np(gauss_pyr0["scales"]),
            "omega": None,
            "opacities": _np(gauss_pyr0.get("opacities")),
        }

    for i in range(1, n_bands):
        mask = (criteria >= lod_bounds[i - 1]) & (criteria < lod_bounds[i])
        n = int(np.sum(mask))
        print(f"Level {i}: {n} components in [{lod_bounds[i - 1]:.2f}, {lod_bounds[i]:.2f})")
        if n:
            _add_level(decomposed_model, level_primitives, gabor, mask, i)

    all_extents = _apply_adaptive_extents(level_primitives, decomposed_model, args)
    _apply_pruning(level_primitives, decomposed_model, all_extents, args)
    _finalize_lod(decomposed_model, scene, cam_array, output_path, n_bands, args.spp)


def _lod_partition_gaussian(args, model, scene, cam_array, grid, a, b, gauss_target_level, n_bands, output_path):
    import drjit as dr
    import mitsuba as mi
    from tqdm import tqdm
    import gfields.training.regression as gu

    decomposed_model = deepcopy(model)
    gauss = decomposed_model.pop("primitives_pyr0")  # current Gaussian level, re-split below
    criteria = gauss["scales"].numpy()

    min_scale = gu.gaussian_min_spatial_scale(gauss_target_level, a, b)
    nyquist_scale = gu.gaussian_nyquist_spatial_scale(a, b, grid.shape, eps=0.25)

    # Analytic Gaussian pyramid: blur level 0 by adding sigma in quadrature and
    # save each blurred level as its own standalone model.
    analytic_path = output_path / str(n_bands)
    analytic_path.mkdir(parents=True, exist_ok=True)
    lod_scales = double_spaced(nyquist_scale, min_scale, n_bands)  # scales invert frequency
    lod_scales[0] = 0
    gauss_images = []
    for i, sigma in tqdm(list(enumerate(lod_scales)), desc="Gaussian LOD (analytic)", total=n_bands):
        lvl_model = deepcopy(model)
        lvl_model["primitives_pyr0"]["scales"] = mi.TensorXf(np.sqrt(criteria ** 2 + sigma ** 2))
        lvl_scene = mi.load_dict(lvl_model)
        img_lvl = mi.render(lvl_scene, sensor=cam_array, spp=args.spp, seed=42)
        gauss_images.append(img_lvl)
        mi.util.write_bitmap(str(analytic_path / f"optimized_pyr{i}.exr"), img_lvl)
        gu.save_model(analytic_path / str(i), lvl_scene, level=0)
    mi.util.write_bitmap(str(analytic_path / f"pyramid_{n_bands}.exr"), dr.concat(gauss_images, axis=0))

    # Pruned/partitioned pyramid.
    lod_bounds = list(lod_scales)
    lod_bounds.insert(1, nyquist_scale)
    lod_bounds[-1] = np.inf
    if args.same_number_of_prims_per_level:
        criteria = criteria.min(-1).reshape(-1, 1)
        lod_bounds = equal_partitioning(criteria.min(-1), n_bands + 1)

    level_primitives: Dict[int, dict] = {}
    for i in range(n_bands):
        mask = ((lod_bounds[i] <= criteria) & (criteria < lod_bounds[i + 1])).any(axis=-1)
        n = int(np.sum(mask))
        print(f"Level {i}: {n} components in [{lod_bounds[i]:.2f}, {lod_bounds[i + 1]:.2f})")
        if n:
            _add_level(decomposed_model, level_primitives, gauss, mask, i)

    all_extents = _apply_adaptive_extents(level_primitives, decomposed_model, args)
    _apply_pruning(level_primitives, decomposed_model, all_extents, args)
    _finalize_lod(decomposed_model, scene, cam_array, output_path, n_bands, args.spp)

    print(
        f"Analytic per-level models saved under {analytic_path} (load /0, /1, ... for a single LOD level).\n"
        f"Pruned LOD pyramid saved under {output_path} (load the directory for the full pyramid)."
    )


def lod_partition_main() -> None:
    """CLI: ``gfields decompose lod-partition`` -- split a trained Gauss or Gabor
    checkpoint into frequency-band LOD levels, with optional adaptive per-primitive
    extent and peak-density pruning."""
    import gfields.training.configs as volume_configs

    args = _lod_partition_setup()
    params, model, scene, cam_array, has_gabor = _lod_partition_prepare(
        args.path, args.num_cams, args.cam_res_x, args.cam_res_y)
    grid = np.load(str(args.path / "grid_pyr0.npy"))
    volume_config = volume_configs.get_volume_config(params["volume_grid"])
    a, b = 0, volume_config.volume_scale.max()
    gauss_target_level = 3
    n_bands = args.output_levels
    output_path = Path(args.output)

    if has_gabor:
        _lod_partition_gabor(args, model, scene, cam_array, grid, volume_config,
                             a, b, gauss_target_level, n_bands, output_path)
    else:
        _lod_partition_gaussian(args, model, scene, cam_array, grid,
                                a, b, gauss_target_level, n_bands, output_path)
    print("Done.")


__all__ = [
    "decompose_asset",
    "steerable_main",
    "render_decomposition",
    "lod_partition_main",
]


if __name__ == "__main__":
    import sys

    # ``gfields.cli`` runs this module with sys.argv[0] set to the routed
    # label ("gfields decompose steerable" / "gfields decompose lod-partition")
    # so it can select which of the two fused CLIs to run.
    if sys.argv and sys.argv[0].endswith("lod-partition"):
        lod_partition_main()
    else:
        steerable_main()
