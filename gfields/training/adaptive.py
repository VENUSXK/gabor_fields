# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive extent clamping for Gabor / Gaussian primitives (math + CLI).

From the Gabor density integral::

    density(E) = opacity * exp(-0.5*(E^2 + omega_eff^2)) / (2*pi * prod(scales))

solving ``density(E) = tau`` for E gives a per-primitive bounding extent that
respects an absolute density threshold ``tau`` (accounting for opacity, scale
normalization and, optionally, the frequency decay term). Tighter extents mean
smaller OptiX shells -> faster traversal with negligible visual change.

This is a post-training asset-refinement pass: :func:`clamp_asset_level`
computes the extents for one on-disk pyramid level, and ``train.py`` can run it
automatically at the end of optimization (``--post_clamp``). The CLI
(``gfields decompose clamp``) exposes the same helper standalone.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def _to_numpy(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)


def compute_effective_frequency(
    omega: np.ndarray,
    scales: np.ndarray,
    chi: float = 1.0,
) -> np.ndarray:
    """``omega_eff = |beta| * chi`` (scales argument kept for API symmetry)."""
    del scales  # not used; kept for API parity with prior call sites
    omega = _to_numpy(omega)
    omega = np.atleast_1d(omega).flatten()
    return np.abs(omega) * chi


def clamp_extents(
    scales: np.ndarray,
    omega: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
    *,
    base_extent: float = 3.0,
    min_extent: float = 0.1,
    threshold: float = 1e-4,
    envelope_only: bool = False,
    freq_chi: float = 1.0,
) -> np.ndarray:
    """Optimal bounding extent per primitive against an absolute density threshold.

    The output is clamped to ``[min_extent, base_extent]``.
    """
    scales = _to_numpy(scales)
    scales = np.atleast_2d(scales)

    N = scales.shape[0]
    E0_sq = base_extent ** 2
    E_min_sq = min_extent ** 2

    prod_scales = np.prod(scales, axis=-1)

    if opacities is not None:
        opacities = _to_numpy(opacities)
        opacities = np.atleast_1d(opacities).flatten()
        opacities = np.maximum(opacities, 1e-10)
    else:
        opacities = np.ones(N)

    if omega is not None and not envelope_only:
        omega_eff = compute_effective_frequency(omega, scales, chi=freq_chi)
        freq_term = 3 * omega_eff ** 2
    else:
        freq_term = np.zeros(N)

    normalization = threshold * 2.0 * np.pi * prod_scales
    abs_opacities = np.abs(opacities)
    log_arg = abs_opacities / normalization

    E_sq = np.where(
        log_arg > 1.0,
        2.0 * np.log(log_arg) - freq_term,
        E_min_sq,
    )
    E_sq = np.clip(E_sq, E_min_sq, E0_sq)
    return np.sqrt(E_sq)


def compute_peak_density(
    scales: np.ndarray,
    opacities: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    *,
    include_frequency: bool = False,
) -> np.ndarray:
    """``peak = |opacity| / (2*pi * prod(scales))`` with optional Gabor decay factor."""
    scales = _to_numpy(scales)
    scales = np.atleast_2d(scales)
    N = scales.shape[0]
    prod_scales = np.prod(scales, axis=-1)

    if opacities is not None:
        opacities = _to_numpy(opacities)
        opacities = np.atleast_1d(opacities).flatten()
    else:
        opacities = np.ones(N)

    peak = np.abs(opacities) / (2.0 * np.pi * prod_scales)
    if include_frequency and omega is not None:
        omega_eff = compute_effective_frequency(omega, scales)
        peak = peak * np.exp(-0.5 * omega_eff ** 2)
    return peak


def compute_pruning_mask(
    scales: np.ndarray,
    opacities: Optional[np.ndarray] = None,
    omega: Optional[np.ndarray] = None,
    *,
    threshold: float = 1e-6,
    include_frequency: bool = False,
) -> np.ndarray:
    """Boolean keep-mask based on :func:`compute_peak_density`."""
    return compute_peak_density(scales, opacities, omega, include_frequency=include_frequency) >= threshold


# ---------------------------------------------------------------------------
# Asset-level helper (used by the CLI and the post-training auto-clamp hook)
# ---------------------------------------------------------------------------

def clamp_asset_level(
    asset_dir: Path | str,
    level: int,
    *,
    base_extent: float = 3.0,
    min_extent: float = 0.1,
    threshold: float = 1e-4,
    envelope_only: bool = False,
    freq_chi: float = 1.0,
    output: Optional[Path] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Compute per-primitive extents for one on-disk pyramid level and save them.

    Reads ``<asset_dir>/data/root.primitives_pyr{level}.ply``, computes the
    adaptive extents, and writes them to ``output`` (default
    ``<asset_dir>/npy_data/extents_pyr{level}.npy``). Returns the extent array.
    """
    import gfields.io

    asset_dir = Path(asset_dir)
    ply_path = asset_dir / "data" / f"root.primitives_pyr{level}.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"{ply_path} not found (expects the unified per-level .ply store).")
    a = gfields.io.load_ellipsoid_ply(ply_path)
    scales = np.asarray(a["scales"], dtype=np.float32)
    omega = np.asarray(a["omega"], dtype=np.float32) if "omega" in a else None
    opac_key = next((k for k in ("opacities", "sigma_t", "sigmat") if k in a), None)
    opacities = np.asarray(a[opac_key], dtype=np.float32) if opac_key else None

    extents = clamp_extents(
        scales, omega=omega, opacities=opacities,
        base_extent=base_extent, min_extent=min_extent, threshold=threshold,
        envelope_only=envelope_only, freq_chi=freq_chi)

    if verbose:
        peak = compute_peak_density(scales, opacities=opacities, omega=omega,
                                    include_frequency=not envelope_only)
        print(f"[adaptive] level {level}: {len(extents)} primitives from {ply_path}")
        print(f"[adaptive]   extent min/mean/max: {extents.min():.4f} / {extents.mean():.4f} / {extents.max():.4f}")
        print(f"[adaptive]   peak   min/mean/max: {peak.min():.6f} / {peak.mean():.6f} / {peak.max():.6f}")

    out = Path(output) if output is not None else (asset_dir / "npy_data" / f"extents_pyr{level}.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, extents)
    if verbose:
        print(f"[adaptive]   wrote {out}")
    return extents


def _resolve_level(asset_dir: Path, level: int | None) -> int:
    if level is not None:
        return level
    m = re.search(r"pyr(\d+)$", asset_dir.name)
    return int(m.group(1)) if m else 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``gfields decompose clamp`` -- compute per-primitive adaptive extents."""
    parser = argparse.ArgumentParser(
        description="Compute per-primitive adaptive bounding extents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("asset", type=Path,
                        help="Asset directory with data/root.primitives_pyr{level}.ply.")
    parser.add_argument("--level", type=int, default=None,
                        help="Pyramid level to clamp. Defaults to the trailing "
                             "_pyr{N} in the asset directory name, else 0.")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="Absolute density threshold tau.")
    parser.add_argument("--base-extent", type=float, default=3.0,
                        help="Cap on the per-primitive extent in sigma units.")
    parser.add_argument("--min-extent", type=float, default=0.1,
                        help="Floor on the per-primitive extent in sigma units.")
    parser.add_argument("--envelope-only", action="store_true",
                        help="Ignore Gabor frequency decay (treat all primitives as Gaussians).")
    parser.add_argument("--freq-chi", type=float, default=1.0,
                        help="Ray-direction statistic chi for the frequency term.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Where to write the per-primitive extent array. "
                             "Defaults to <asset>/npy_data/extents_pyr{level}.npy.")
    args = parser.parse_args(argv)

    level = _resolve_level(args.asset, args.level)
    clamp_asset_level(
        args.asset, level,
        base_extent=args.base_extent, min_extent=args.min_extent,
        threshold=args.threshold, envelope_only=args.envelope_only,
        freq_chi=args.freq_chi, output=args.output)
    return 0


__all__ = [
    "compute_effective_frequency",
    "clamp_extents",
    "compute_peak_density",
    "compute_pruning_mask",
    "clamp_asset_level",
    "main",
]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
