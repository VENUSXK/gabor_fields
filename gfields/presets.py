# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Baked-in presets for the example workflows (train / render).

This is the single source of truth for the paper's winning training recipe
(default primitive counts + density baked in), the named-asset volume lookup,
and the render-mode presets (tomography / multiple scattering). It also
handles volume-file resolution (converting Mitsuba ``.vol`` / OpenVDB ``.vdb``
grids to cached ``.npy``).

The ``examples/train.{sh,ps1}`` and ``examples/render.{sh,ps1}`` shims only
locate a python interpreter and run ``python -m gfields.train`` /
``python -m gfields.render``; those entry points pull the recipe / mode flags
from here (see ``build_train_argv`` and ``RENDER_MODES``). Every flag injected
here can be overridden from the command line — argparse keeps the last
occurrence, and user flags always come after the injected defaults.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from gfields._paths import REPO_ROOT as REPO


# ---------------------------------------------------------------------------
# Default resources
# ---------------------------------------------------------------------------

def default_envmap() -> Path:
    return REPO / "resources" / "environment_maps" / "rosendal_park_sunset_puresky_2k.exr"


def _abspath(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO / p)


# ---------------------------------------------------------------------------
# Volume resolution (.npy passthrough, .vol / .vdb -> cached .npy)
# ---------------------------------------------------------------------------

def _convert_vol(vol_path: Path, cache_path: Path) -> Path:
    import mitsuba as mi
    import numpy as np

    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

    grid = mi.VolumeGrid(str(vol_path))
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected Mitsuba volume shape {arr.shape} for {vol_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.ascontiguousarray(arr, dtype=np.float32))
    return cache_path


def _convert_vdb(vdb_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gfields.references.vdb",
           "--input", str(vdb_path), "--output", str(cache_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "VDB conversion failed (needs OpenVDB: "
            "conda install -c conda-forge openvdb).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    candidates = sorted(cache_dir.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .npy produced from {vdb_path} in {cache_dir}")
    return candidates[0]


def resolve_volume(volume: str | Path) -> Path:
    """Return a ``.npy`` path, converting Mitsuba ``.vol`` / OpenVDB ``.vdb`` via a cache."""
    source = _abspath(volume)
    if not source.is_file():
        raise FileNotFoundError(f"Volume not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".npy":
        return source.resolve()

    cache_root = REPO / "results" / "volume_cache"
    digest = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:12]
    cache_path = cache_root / f"{source.stem}_{digest}.npy"
    if cache_path.exists() and cache_path.stat().st_mtime >= source.stat().st_mtime:
        return cache_path.resolve()

    if suffix == ".vol":
        return _convert_vol(source, cache_path).resolve()
    if suffix == ".vdb":
        return _convert_vdb(source, cache_root).resolve()
    raise ValueError(f"Unsupported volume format {suffix!r} ({source}). "
                     "Use .npy, Mitsuba .vol, or OpenVDB .vdb.")


# ---------------------------------------------------------------------------
# Training: the winning ("champion") recipe + named-asset volume lookup
# ---------------------------------------------------------------------------
# Three non-obvious, critical settings baked in:
#   * per-parameter LRs are set EXPLICITLY (the CLI default of 1.0 x global_lr
#     is ~100-100000x too high and plateaus quality around 39 dB);
#   * --limit_max_scale + --max_scale_mult 0.5 hard-cap primitive size: tighter
#     primitives are BOTH ~6x faster per step AND higher quality;
#   * 1200 Gabor iterations + a light L1 scale regularizer (2026-07-16 sweep,
#     results/sweep_20260716/): 600 iterations stops ~1 dB short of the plateau,
#     and lambda_scale=1e-6 trims mean primitive scale for ~6% faster tomography
#     renders at no measurable quality cost. (Also measured there: densifying
#     up to a fixed budget yields FATTER primitives — 73% slower renders and
#     slightly lower PSNR than starting at the budget — so densification stays
#     off; slower MCMC noise decay and gentler LRs bring no gain.)
#
# Primitive counts and density (--opacity_scale) are basically the only knobs
# that differ across volumes in practice, so a single default recipe covers
# most assets; override --gauss_count/--gabor_count/--opacity_scale from the
# CLI for anything unusual (they win over the injected defaults below). Other
# known-good (gauss, gabor) combos, roughly increasing in quality/cost:
#   512 + 8192, 1024 + 24576, 2048 + 32768.
# --opacity_scale 10.0 is a reasonable middle-of-the-road density for most
# volumes; if a render looks too washed out or too opaque, nudge it down/up
# (values 7.5-70 have been used for individual paper assets, see
# gfields.training.configs).
TRAIN_RECIPE = """
    --cam_count 32 --sample_single_hemisphere
    --cam_res_x 512 --cam_res_y 512 --cam_scale 0.95 --cam_y_offset -0.25
    --importance_sample_means --pad_grid
    --gauss_target_level 3 --gabor_target_level 0
    --gauss_count 512 --gabor_count 4096
    --iterations_gaussian 240 --iterations_gabor 1200
    --opt_spp 2 --ref_spp 8192 --ref_kernel_spp 64
    --init_opacity 0.00008 --init_omega 1.5 --init_scale 0.0175
    --limit_max_scale --max_scale_mult 0.5
    --global_lr_gabor 4.0 --global_lr_gaussian 4.0
    --centers_lr 0.003 --scales_lr 0.0002 --quats_lr 0.0003
    --opacity_lr 0.00002 --omega_lr 0.0005
    --lambda_opacity 0.0 --lambda_scale 1e-6
    --noise_init 50 --noise_decay 0.75 --relocate_until 1000
    --densify_every 30 --densify_count 1024 --densify_until 0
    --start_pruning_from 200000000000000 --prune_every 30 --opacity_decay 0.0
    --checkpoint_every 50
    --opacity_scale 10.0
""".split()

# Gaussian-only mode: fit full-detail level-0 Gaussians, skip the Gabor phase.
GAUSSIAN_MODE = "--gauss_target_level 0 --gauss_count 24576 --skip_gabor_optim".split()

# Default asset used when the training scripts are invoked without a target
# (only "smoke" ships in the repo; the others require external volume data).
DEFAULT_ASSET = "smoke"

# Named assets -> volume-grid path. This is just a shorthand for known volume
# files; per-asset training flags aren't baked in here anymore (see
# TRAIN_RECIPE above) — pass --gauss_count/--gabor_count/--opacity_scale etc.
# on the CLI to tune an individual asset.
TRAIN_ASSETS = {
    "wdas8": "data/volumes/grids/wdas8.npy",
    "embergen": "data/volumes/grids/embergen.npy",
    "explosion": "data/volumes/grids/explosion.npy",
    "fire": "data/volumes/grids/fire.npy",
    "smoke": "data/volumes/smoke.vol",
    "smokev2": "data/volumes/grids/smokev2.npy",
    "tornado": "data/volumes/grids/tornado.npy",
    "dust": "data/volumes/grids/dust.npy",
}


def build_train_argv(raw: list[str]) -> list[str]:
    """Expand the ``gfields.train`` front-end into a full argparse argv.

    Front-end forms (the leading non-flag token is the *target*)::

        python -m gfields.train <preset>            # e.g. wdas8
        python -m gfields.train <volume.npy|.vol|.vdb> --opacity_scale S
        python -m gfields.train <target> --gaussian # Gaussian-only baseline

    A recognized *target* injects the champion recipe (+ per-asset preset, +
    Gaussian-mode flags), a resolved ``--volume_grid`` and a default
    ``--output``, followed verbatim by every remaining user flag (which wins on
    conflict). With no target and no explicit ``--volume_grid``, the target
    defaults to ``DEFAULT_ASSET`` (``smoke``, the only asset shipped with the
    repo). The raw ``--volume_grid ... --output ...`` interface still works
    when ``--volume_grid`` is passed explicitly.
    """
    raw = list(raw)

    gaussian = False
    kept: list[str] = []
    for tok in raw:
        if tok == "--gaussian":
            gaussian = True
        else:
            kept.append(tok)
    raw = kept

    target = None
    if raw and not raw[0].startswith("-"):
        target = raw.pop(0)
    elif not raw or "--volume_grid" not in raw:
        # No positional target and no explicit --volume_grid: fall back to the
        # only asset shipped with the repo instead of erroring out.
        target = DEFAULT_ASSET
    if target is None:
        return raw  # legacy direct mode (user supplies --volume_grid/--output)

    if target in TRAIN_ASSETS:
        name, volume = target, TRAIN_ASSETS[target]
    else:
        name, volume = Path(target).stem, target

    resolved = str(resolve_volume(volume))
    mode_flags = GAUSSIAN_MODE if gaussian else []
    suffix = "_gaussian" if gaussian else ""
    default_output = str(REPO / "results" / "checkpoints" / f"{name}{suffix}")

    injected = [*TRAIN_RECIPE, *mode_flags,
                "--volume_grid", resolved, "--output", default_output]
    return [*injected, *raw]


# ---------------------------------------------------------------------------
# Rendering: mode presets (tomography / multiple scattering)
# ---------------------------------------------------------------------------
# `video` is handled by `python -m gfields.render video` (gfields.render.run_video,
# which already carries its own defaults); only the still-image modes need presets.
RENDER_MODES = {
    "tomo": [
        "--spp", "64", "--max_depth", "-1",
        "--output", str(REPO / "results" / "renders" / "tomo"),
    ],
    "ms": [
        "--multiple_scattering", "--max_depth", "10",
        "--rr_depth", "10", "--rr_nee_depth", "64",
        "--solver_max_iterations", "4", "--albedo", "0.8",
        "--spp", "64", "--envmap", str(default_envmap()),
        "--output", str(REPO / "results" / "renders" / "ms"),
    ],
}
