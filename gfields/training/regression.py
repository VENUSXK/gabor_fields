# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gabor/Gaussian regression training loop and its helpers.

The public surface (``setup``, ``prepare_cameras``, ``generate_random_kernels``,
``render_reference_images``, ``densify``, ``prune``, ``save_model`` …) used by
``gfields optimize gabor``. Formerly split across a thin ``regression`` shim and
a private ``_regression_impl``; merged into one module. Importing it initializes
Mitsuba (``mi.set_variant``) and pulls the training stack, matching what any
``from ... import *`` consumer already triggered.
"""
import json
import os, sys, argparse
import hashlib
from os.path import join
from typing import Callable, List, Tuple, Any, Dict, Optional, Literal, Union
from collections import defaultdict

from colorist import green, yellow, effect_dim
from pathlib import Path
import numpy as np

try:
    import wandb
except ImportError:  # wandb is an optional training/analysis dependency
    wandb = None

from scipy.ndimage import gaussian_filter

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
# mi.set_log_level(mi.LogLevel.Debug)
T = mi.ScalarTransform4f
from tqdm import tqdm
import matplotlib.pyplot as plt

import gfields
from gfields.integrators.common import *
from gfields.training.configs import VolumeConfig

np.random.seed(0)


def setup() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Optimize volumetric primitives from a 3D grid. Pass a preset '
                    'name or a volume file (.npy/.vol/.vdb) as the leading positional '
                    'argument to auto-inject the paper recipe (see gfields.presets); '
                    'or drive it directly with --volume_grid/--output.')
    # Target-driven mode (gfields.presets.build_train_argv) fills these in; they
    # stay required in spirit but are validated after parsing so the recipe /
    # preset injection can supply them.
    parser.add_argument('--output',              type=str, default=None, help='Path to the result output folder')
    parser.add_argument('--volume_grid',         type=str, default=None, help='Path to the .npy volume grid file')
    parser.add_argument('--write_image_every',   type=int,   default=25, help='Frequency at which images and stats are computed')
    parser.add_argument('--log_every',           type=int,   default=10,  help='Compute and display loss/PSNR metrics every N iterations (forces a device sync; 1 = every iteration)')
    parser.add_argument('--checkpoint_every',    type=int,   default=20,  help='Save the optimized model every N iterations so an interrupted/killed run leaves a usable, resumable asset (--from_gabor_checkpoint / --from_gaussian_checkpoint)')

    # Cameras and reference images parameters
    parser.add_argument('--cam_count',           type=int, default=16,     help='Number of cameras to use in the optimization process')
    parser.add_argument('--sample_single_hemisphere', action='store_true', help='Sample cameras from a single hemisphere only instead of full sphere.')
    parser.add_argument('--cam_res_x',           type=int, default=800,    help='Camera resolution')
    parser.add_argument('--cam_res_y',           type=int, default=800,    help='Camera resolution')
    parser.add_argument('--cam_y_offset',        type=float, default=0.0,  help='Height offset for the camera position')
    parser.add_argument('--vol_size_scale',      type=float, default=1.0,  help='Scale factor for the volume size in the scene')
    parser.add_argument('--cam_scale',           type=float, default=1.0,  help='Scale factor for the camera position in the scene')
    parser.add_argument('--opacity_scale',       type=float, default=None, help='Override the per-asset density (opacity) scale. Required to train a volume with no named VolumeConfig (uses the generic isotropic config).')
    
    # Integrator parameters
    parser.add_argument('--ref_spp',             type=int,   default=2048,                     help='Number of samples per pixel used for rendering the reference images')
    parser.add_argument('--ref_spp_chunk',       type=int,   default=1024,                   help='Max SPP per volpath reference pass (lower avoids GPU OOM at high ref_spp)')
    parser.add_argument('--ref_kernel_spp',      type=int,   default=8,                        help='Number of samples per pixel used for rendering the kernel reference images')
    parser.add_argument('--opt_spp',             type=int,   default=1,                        help='Number of samples per pixel used for rendering during the optimization')
    parser.add_argument('--grad_spp',            type=int,   default=1,                        help='Number of samples per pixel used for gradient computation during the optimization')
    parser.add_argument('--max_depth',           type=int,   default=-1,                       help='Maximum path depth for integrator')
    parser.add_argument('--integrator',          type=str,   default='gfields_tomography',
                        choices=['gfields_tomography'],
                        help='Tomography integrator plugin for optimization renders')
    parser.add_argument('--opt_cam_batch',       type=int,   default=0,
                        help='Cameras per optimization render batch (0 = all cameras in one batch sensor)')
    parser.add_argument('--max_scale_mult',      type=float, default=1.0,
                        help='Multiplier on the gabor max-scale bound (with --limit_max_scale). '
                             '<1 tightens the cap: primitives saturate the default cap, and large '
                             'primitives dominate ray-hit cost, so a tighter cap trades quality for '
                             'render/training speed.')
    parser.add_argument('--cam_subsample',       type=int,   default=0,
                        help='Stochastic camera subsampling for the gabor phase: render only this many '
                             'randomly-chosen cameras per iteration (0 = use all cameras). Speeds up '
                             'high-primitive-count training by trading full-batch gradients for SGD-over-views; '
                             'the final/post-prune PSNR is still measured over all cameras.')
    parser.add_argument('--use_kernel_freeze', action='store_true',
                        help='Enable @dr.freeze on the tomography integrator kernels')
    parser.add_argument('--use_anyhit', action='store_true', default=True,
                        help='Evaluate primal tomography renders with one OptiX '
                             'any-hit traversal per ray (order-independent optical-'
                             'depth accumulation in the shader) instead of the '
                             'sequential hit walk; adjoint passes keep the walk. '
                             'Default ON (the optimization fast path). '
                             'See docs/prb_stochastic.md section 5.')
    parser.add_argument('--no_anyhit', dest='use_anyhit', action='store_false',
                        help='Disable the primal any-hit fast path; fall back to '
                             'the sequential hit walk for primal tomography renders.')
    parser.add_argument('--seed_offset',         type=int,   default=0,
                        help='Added to every optimization render seed; vary it across otherwise '
                             'identical runs to measure run-to-run MC (seed) noise.')
    parser.add_argument('--pad_grid',            action='store_true',                          help='Pad the volume grid with zeros to avoid artifacts at the boundaries')
    
    # checkpointing
    parser.add_argument('--from_gaussian_checkpoint',       type=str, default='',              help='Skip Gaussian optimization and load the checkpoint from this path')
    parser.add_argument('--skip_gaussian_optim', action='store_true', help='Skip Gaussian optimization and only optimize Gabor primitives')
    parser.add_argument('--from_gabor_checkpoint',          type=str, default='',              help='Load Gabor optimization checkpoint from this path')
    parser.add_argument('--skip_gabor_optim',    action='store_true', help='Skip Gabor optimization and only optimize Gaussian primitives')
    parser.add_argument('--epoch_start',         type=int,   default=0,       help='Start epoch for the optimization')
    parser.add_argument('--track_grads',         action='store_true', help='Track gradients during optimization')

    # Post-training asset-refinement passes (run once optimization finishes)
    parser.add_argument('--post_clamp',          action='store_true', help='After training, compute per-primitive adaptive extents for each level (gfields.training.adaptive) and save them.')
    parser.add_argument('--post_prune',          action='store_true', help='After training, run PSNR-budget pruning (gfields.training.post_pruning) on the trained asset.')
    parser.add_argument('--post_prune_budget',   type=float, default=0.1,     help='PSNR budget (dB) for --post_prune: prune until PSNR would drop by this much. Safe default 0.1 dB.')

    # Optimization parameters
    parser.add_argument('--iterations_gaussian', type=int,   default=4096,    help='Number of iterations for the gradient descent optimization')
    parser.add_argument('--iterations_gabor',    type=int,   default=4096,    help='Number of iterations for the gradient descent optimization')
    parser.add_argument('--regen_pyramid',       action="store_true",         help='Re-generate the pyramid even if it already exists')
    parser.add_argument('--gauss_count',         type=int,   default=12**3,   help='Resolution of the 3D grid of volumetric primitives') # 16 for gabor
    parser.add_argument('--gabor_count',         type=int,   default=12**3,   help='Resolution of the 3D grid of volumetric primitives') # 16 for gabor
    parser.add_argument('--init_albedo',         type=float, default=0.0,     help='Initial value for the volumetric primitive albedo')
    parser.add_argument('--init_omega',          type=float, default=np.pi/2, help='Initial value for the volumetric primitive frequency weights')
    parser.add_argument('--init_opacity',         type=float, default=1.2e-5,    help='Initial value for the volumetric primitive opacity')
    parser.add_argument('--init_scale',          type=float, default=1.0,     help='Initial value for the volumetric primitive scale')
    parser.add_argument('--frequency_bounds_regularization', type=float, default=0.0, help='Regularization term for the frequency weights')
    parser.add_argument('--importance_sample_means', action='store_true', help='Use importance sampling for the initial means of the Gabor kernels')
    parser.add_argument('--limit_max_scale',     action='store_true',         help='Limit the maximum scale of the Gabor primitives. This can improve rendering times when the number of primitives is high but harm if the number is low.')

    parser.add_argument('--lambda_opacity',      type=float, default=0.0,     help='Weight for the opacity regularization term')
    parser.add_argument('--lambda_scale',        type=float, default=0.0,     help='Weight for the scale regularization term')
    parser.add_argument('--lambda_frequency',    type=float, default=0.1,     help='Weight for the super-Nyquist frequency regularizer (fraction of primitives whose frequency exceeds the volume band limit). Small-volume assets have a low band limit, so an unweighted count would swamp the image loss; keep this modest.')
    parser.add_argument('--save_best', action='store_true', default=True, help='Save the best clean-PSNR Gabor mixture seen during training, not just the final iterate (guards against the MCMC-noise oscillation leaving the last step below the peak). On by default.')
    parser.add_argument('--no_save_best', dest='save_best', action='store_false', help='Disable best-checkpoint selection; save the final iterate.')
    parser.add_argument('--noise_decay',         type=float, default=0.75,    help='Decay factor for the noise added to the gradients during optimization')
    parser.add_argument('--noise_init',          type=float, default=5e1,     help='Initial standard deviation of the noise added to the gradients during optimization')
    parser.add_argument('--relocate_until',      type=int,   default=1000,    help='Relocate dead primitives until this iteration')

    # Learning rates
    parser.add_argument('--global_lr_gaussian',  type=float, default=2.0,     help='Global learning rate')
    parser.add_argument('--global_lr_gabor',     type=float, default=10.0,    help='Global learning rate')
    parser.add_argument('--centers_lr',          type=float, default=1.0,     help='Learning rate multiplier for the volumetric primitive center parameters')
    parser.add_argument('--scales_lr',           type=float, default=1.0,     help='Learning rate multiplier for the volumetric primitive scale parameters')
    parser.add_argument('--quats_lr',            type=float, default=1.0,     help='Learning rate multiplier for the volumetric primitive quat parameters')
    parser.add_argument('--opacity_lr',          type=float, default=1.0,     help='Learning rate multiplier for the volumetric primitive opacity parameters')
    parser.add_argument('--albedo_lr',           type=float, default=0.0,     help='Learning rate multiplier for the volumetric primitive albedo parameters')
    parser.add_argument('--omega_lr',            type=float, default=1.0,     help='Learning rate multiplier for the volumetric primitive frequency weight parameters')

    parser.add_argument('--warmup_ratio',        type=float, default=0.15,     help='Percentage of steps spent in warmup phase (0 to 1)')
    parser.add_argument('--min_lr_ratio',        type=float, default=0.1,   help='Minimum learning rate relative to initial_lr at end of schedule (0 to 1)')

    # Primitive control
    parser.add_argument('--start_pruning_from', type=int,   default=10,       help='Start pruning primitives from this iteration')
    parser.add_argument('--prune_every',        type=int,   default=400,      help='Prune the primitives every N iterations')
    parser.add_argument('--opacity_decay',      type=float, default=0.001,    help='Decay rate for the primitive opacity')
    parser.add_argument('--densify_every',      type=int,   default=200,      help='Densify the primitives every N iterations')
    parser.add_argument('--densify_until',      type=int,   default=1000,     help='Densify the primitives until this iteration')
    parser.add_argument('--densify_count',      type=int,   default=400,      help='Number of primitives to add during densification')

    parser.add_argument('--gauss_target_level', type=int,   default=3,        help='Target pyramid level to optimize')
    parser.add_argument('--gabor_target_level', type=int,   default=0,        help='Target pyramid level to optimize')

    # Expand the target-driven front-end (preset / volume + --gaussian) into a
    # full argv, injecting the champion recipe. With no leading target this is a
    # no-op and the raw --volume_grid/--output interface is used unchanged.
    from gfields.presets import build_train_argv
    args = parser.parse_args(build_train_argv(sys.argv[1:]))

    if args.volume_grid is None:
        parser.error("no volume: pass a preset/volume target (e.g. `wdas8` or "
                     "`data/volumes/grids/mine.npy`) or --volume_grid PATH")
    if args.output is None:
        parser.error("no output: a preset target sets one automatically; in the "
                     "raw --volume_grid mode you must also pass --output DIR")

    # create all output directories
    output = Path(args.output)  # we don't overwrite the output in the args yet, so we can serialize the args to json file
    output.mkdir(parents=True, exist_ok=True)

    (output / 'frames_pyr0').mkdir(exist_ok=True)
    (output / 'frames_pyr1').mkdir(exist_ok=True)
    (output / 'refs_pyr0').mkdir(exist_ok=True)
    (output / 'refs_pyr1').mkdir(exist_ok=True)

    # save the arguments to a json file
    args_path = output / "args.json"
    with open(str(args_path), 'w') as f:
        json.dump(vars(args), f, indent=4)
    green(f"Wrote args to {str(args_path)}")

    if wandb is not None:
        wandb.init(project='gabor', entity='nihermann', config=vars(args))

    args.output = output  # now we can overwrite the output in the args

    if args.skip_gaussian_optim and args.from_gaussian_checkpoint == "":
        if (ckpt_path := (output / "optimized_asset_pyr0" / "npy_data")).exists():
            yellow("Warning: --skip_gaussian_optim was set but not --from_gaussian_checkpoint but optimized_asset_pyr0 folder exists. Assuming you want to continue a failed run. If this is not the case, please provide a path to the checkpoint explicitly.")
            args.from_gaussian_checkpoint = str(ckpt_path)
        else:
            raise ValueError("If you want to skip Gaussian optimization, you must provide a checkpoint path with --from_gaussian_checkpoint")

    if args.from_gaussian_checkpoint and not os.path.exists(args.from_gaussian_checkpoint):
        if args.skip_gaussian_optim:
            raise FileNotFoundError(f"Checkpoint file {args.from_gaussian_checkpoint} does not exist.")
        yellow(f"Gaussian checkpoint {args.from_gaussian_checkpoint} not found; will optimize from scratch.")
        args.from_gaussian_checkpoint = ""
    
    if args.from_gabor_checkpoint and not os.path.exists(args.from_gabor_checkpoint):
        raise FileNotFoundError(f"Checkpoint file {args.from_gabor_checkpoint} does not exist.")
    
    if not os.path.exists(args.volume_grid):
        raise FileNotFoundError(f"Volume grid file {args.volume_grid} does not exist.")
    
    if args.gauss_target_level < 0:
        raise ValueError(f"Gaussian target level must be non-negative. Got {args.gauss_target_level}.")
    
    if args.gabor_target_level < 0:
        raise ValueError(f"Gabor target level must be non-negative. Got {args.gabor_target_level}.")

    return args


# region Initialization


def prepare_cameras(cam_count: int, cam_res_x: int, cam_res_y: int, cam_scale: float = 1.0, cam_y_offset: float = 0.0, sample_single_hemisphere: bool = True) -> List[mi.ProjectiveCamera]:
    if sample_single_hemisphere:
        directions = sample_uniform_hemisphere(cam_count)
        cameras = []
        for i in range(cam_count):
            sensor_to_world = T().look_at(target=[0, cam_y_offset, 0], origin=directions[i] * 4.0, up=[0, 1, 0])
            sensor_to_world = sensor_to_world @ T().scale([cam_scale, cam_scale, cam_scale]) # 0.6 for bunny 1.3 for cloud
            d = {
                'type': 'orthographic',
                # 'fov': 40,
                'to_world': sensor_to_world,
                'film': {
                    'type': 'hdrfilm',
                    'width': cam_res_x,
                    'height': cam_res_y,
                    'filter': { 'type': 'gaussian' }
                }
            }
            cameras.append(mi.load_dict(d))
        return cameras

    cameras = []
    for i in range(cam_count):
        angle = 180.0 / cam_count * i - 90.0
        sensor_rotation_y = T().rotate([0, 1, 0], angle)
        sensor_rotation_x = T().rotate([1, 0, 0], 90.0 * np.random.rand() - 45.0) # Randomize camera's elevation a bit
        sensor_to_world = T().look_at(target=[0, 0, 0], origin=[0, 0, 4], up=[0, 1, 0])
        sensor_to_world = sensor_rotation_y @ sensor_rotation_x @ sensor_to_world
        sensor_to_world = sensor_to_world @ T().scale([cam_scale, cam_scale, cam_scale]) # 0.6 for bunny 1.3 for cloud
        d = {
            'type': 'orthographic',
            # 'fov': 40,
            'to_world': sensor_to_world,
            'film': {
                'type': 'hdrfilm',
                'width': cam_res_x, 
                'height': cam_res_y,
                'filter': { 'type': 'gaussian' }
            }
        }
        cameras.append(mi.load_dict(d))
    return cameras


def prepare_ref_volume(volume_grid_path: str, output_dir: Path, gauss_target_level: int, gabor_target_level: int, *, pad_grid: bool, regen_pyramid: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the reference volume and pyramid.

    Args:
        volume_grid_path: Path to the volume grid file
        output_dir: Output directory for caching
        gauss_target_level: Target level for Gaussian pyramid
        gabor_target_level: Target level for Gabor pyramid
        pad_grid: Whether to pad the grid with zeros
        regen_pyramid: Force regeneration of pyramid

    Returns:
        Tuple of (voxel_grid, pyramid). voxel_grid may be None if skip_grid_load=True and cache is valid.
    """
    # Shared content-addressed pyramid cache (keyed by volume + target levels),
    # reused across output dirs / runs like the reference-image cache.
    current_params = _get_pyramid_cache_params(volume_grid_path, gauss_target_level, gabor_target_level)
    cache_dir = _shared_cache_dir('pyramid', current_params)
    pyr_paths = [(cache_dir / f'grid_pyr{lvl}.npy') for lvl in [gauss_target_level, gabor_target_level]]
    cache_metadata_path = cache_dir / 'cache_metadata.json'
    cache_valid = _cache_is_valid(cache_metadata_path, current_params) and all(f.exists() for f in pyr_paths)

    if cache_valid and not regen_pyramid:
        green("Pyramid cache is valid (parameters unchanged), loading from disk.")
        pyr = [np.load(f) for f in pyr_paths]
        effect_dim(f'Voxel grid shapes: {pyr[0].shape} & {pyr[1].shape}')
        return pyr[0], pyr[1]
    else:
        # Need to regenerate - must load the grid
        voxel_grid = np.load(volume_grid_path).squeeze()
        orig_shape = voxel_grid.shape
        if pad_grid:
            # pad the grid to a cube to avoid streching dimensions during optimization
            max_dim = max(orig_shape)
            half_pad = [(max_dim - s) // 2 for s in orig_shape]
            voxel_grid = np.pad(
                voxel_grid,
                pad_width=[(half_pad[0], max_dim - orig_shape[0] - half_pad[0]),  # make sure to pad correctly for odd sizes
                           (half_pad[1], max_dim - orig_shape[1] - half_pad[1]),
                           (half_pad[2], max_dim - orig_shape[2] - half_pad[2])],
            )
        effect_dim(f'Voxel grid shape: {voxel_grid.shape} (original: {orig_shape})')

        if not regen_pyramid and not cache_valid:
            yellow("Cache invalidated: pyramid parameters changed. Regenerating pyramid...")
        pyr = []
        for i, level in enumerate([gauss_target_level, gabor_target_level]):
            pyr.append(gaussian_downsample(voxel_grid, downsample=False, level=level) if level > 0 else voxel_grid)
            np.save(pyr_paths[i], pyr[-1])
        # Save cache metadata after successful generation
        _save_cache_metadata(cache_metadata_path, current_params)

        return pyr[0], pyr[1]


def render_reference_images(
    grid: Optional[np.ndarray],
    volume_config: VolumeConfig,
    cameras: List[mi.ProjectiveCamera],
    output_dir: Path,
    level: int = -1,
    spp: int = 128,
    ref_spp_chunk: int = 1024,
    init_albedo: float = 1.,
    regen: bool = False,
    sample_single_hemisphere: bool = True,
    # Cache validation parameters
    volume_grid_path: Optional[str] = None,
    cam_res_x: Optional[int] = None,
    cam_res_y: Optional[int] = None,
    cam_scale: float = 1.0,
    target_level: int = -1,
) -> List[mi.TensorXf]:
    """
    Render or load reference images for a given pyramid level.

    Args:
        grid: The voxel grid to render. Can be None if cache is valid.
        volume_config: Volume configuration
        cameras: List of cameras
        output_dir: Output directory
        level: Pyramid level
        spp: Samples per pixel for rendering
        init_albedo: Initial albedo value
        regen: Force regeneration
        sample_single_hemisphere: Hemisphere sampling flag (for cache validation)
        volume_grid_path: Path to volume grid (for cache validation)
        cam_res_x, cam_res_y: Camera resolution (for cache validation)
        cam_scale: Camera scale (for cache validation)
        target_level: Target pyramid level (for cache validation)
    Returns:
        List of reference images as mi.TensorXf
    """
    ref_images = []
    # Resolve the cache location. With the full cache params available, store /
    # look up references in the SHARED content-addressed cache so they are
    # reused across output dirs and runs (re-optimizing the same asset never
    # re-renders them); otherwise fall back to the legacy per-output-dir path.
    current_params = None
    if volume_grid_path is not None and cam_res_x is not None and cam_res_y is not None:
        current_params = _get_reference_cache_params(
            volume_grid_path=volume_grid_path,
            cam_res_x=cam_res_x,
            cam_res_y=cam_res_y,
            cam_count=len(cameras),
            ref_spp=spp,
            cam_scale=cam_scale,
            sample_single_hemisphere=sample_single_hemisphere,
            init_albedo=init_albedo,
            opacity_scale=volume_config.opacity_scale,
            level=level,
            target_level=target_level
        )
        output_path = _shared_cache_dir('refs', current_params)
        cache_metadata_path = output_path / 'cache_metadata.json'
    else:
        output_path = output_dir / f'refs_pyr{level}'
        output_path.mkdir(parents=True, exist_ok=True)
        cache_metadata_path = output_dir / f'refs_pyr{level}_cache_metadata.json'
    ref_image_paths = get_exr_files(output_path)

    if current_params is not None:
        cache_valid = (
            _cache_is_valid(cache_metadata_path, current_params) and
            len(ref_image_paths) == len(cameras)
        )
    else:
        # Fall back to old behavior if cache params not provided
        cache_valid = len(ref_image_paths) == len(cameras)

    if cache_valid and not regen:
        green(f'Reference images cache valid for level {level} (parameters unchanged), loading from disk.')
        for i, path in enumerate(sorted(ref_image_paths)):
            ref_images.append(mi.TensorXf(mi.Bitmap(path)))
    else:
        # Cache invalid - must render. Grid is required.
        if grid is None:
            raise ValueError(f"Cannot render reference images: grid is None but cache is invalid for level {level}. "
                           "This likely means the cache parameters changed but the grid wasn't loaded.")

        if not regen and not cache_valid and volume_grid_path is not None:
            yellow(f'Cache invalidated for level {level}: reference image parameters changed. Re-rendering...')
        print(f'Rendering reference images for level {level} at {str(output_path)}')
        # train.py fits an absorption-only (no scattering!) mixture, so the
        # common init_albedo=0 case never needs a scattering path tracer: the
        # RRT + supervoxel-majorant integrator (gfields.references) computes
        # the same absorption-only transmittance far faster. Nonzero albedo
        # (scattering experiments) still needs the full volpath tracer.
        if init_albedo == 0.0:
            # GfieldsReferenceIntegrator reads its sigma_t volume (and no
            # separate 'scale' factor) straight off the integrator props, so
            # opacity_scale is baked into the grid data instead.
            volume_grid_scene: mi.Scene = mi.load_dict({
                'type': 'scene',
                'integrator': {
                    'type': 'gfields_reference',
                    'background': 1.0,
                    'to_world': volume_config.to_world,
                    'sigma_t': {
                        'type': 'gridvolume',
                        'data': mi.TensorXf(grid * volume_config.opacity_scale),
                        'to_world': volume_config.to_world
                    },
                },
            })
        else:
            volume_grid_scene = mi.load_dict({
                'type': 'scene',
                'integrator': {
                    'type': 'volpath',
                },
                'object': {
                    'type': 'cube',
                    'bsdf': { 'type': 'null' },
                    'interior': {
                        'type': 'heterogeneous',
                        'sigma_t': {
                            'type': 'gridvolume',
                            'data': mi.TensorXf(grid),
                            'to_world': volume_config.to_world
                        },
                        'albedo': init_albedo,
                        'scale': volume_config.opacity_scale
                    },
                    'to_world': T()
                },
                'environment': { 'type': 'constant' }
            })
        for i, camera in tqdm(enumerate(cameras), desc=f"Rendering reference images of level {level}", leave=False, total=len(cameras)):
            img = render_sensor_chunked(
                volume_grid_scene, camera, spp,
                seed=i, chunk_spp=ref_spp_chunk)
            mi.util.write_bitmap(join(output_path, f'{i:04d}.exr'), img)
            ref_images.append(img)
            del img
            dr.flush_malloc_cache()
        del volume_grid_scene
        dr.flush_malloc_cache()

        # Save cache metadata after successful generation
        if current_params is not None:
            _save_cache_metadata(cache_metadata_path, current_params)

    ref_images = [dr.clip(image, 0.0, 1.0) for image in ref_images]
    return ref_images


def generate_random_kernels(num_kernels: int, init_opacities: float, init_omega: float, init_scale: float,

                            init_albedo: float, level: int, importance_sample_means: bool = False,
                            volume: Optional[np.ndarray] = None, volume_config: Optional[VolumeConfig] = None, opacity_sampling_grid: Optional[np.ndarray] = None, target_level: int = 0) -> Dict[str, mi.TensorXf]:
    if importance_sample_means:
        if volume is None or volume_config is None:
            raise ValueError("Volume and volume_config must be provided for importance sampling.")
        centers = importance_sample_positions(volume, volume_config, n_gmm=num_kernels, threshold=1e-6)
        centers = np.ascontiguousarray(centers[:, ::-1])
    else:
        centers = np.random.rand(num_kernels, 3) * 2.0 - 1.0

    quaternions = mi.Point3f(np.random.rand(3, num_kernels) * 2 * np.pi)
    quaternions = np.ascontiguousarray(dr.euler_to_quat(quaternions).numpy().T)

    # omega = np.abs(np.random.randn(num_kernels, 1)) * 0.4 + init_omega
    # scales = np.abs(np.random.randn(num_kernels, 3)) * 0.05 + init_scale

    # uniform between nyquist of different levels
    # scales = np.random.rand(num_kernels, 3) * (upper_scale_bound - nyquist_scale) + nyquist_scale
    omega = np.random.rand(num_kernels, 1) * (1.5 - 0.7) + 0.7

    if target_level == 0:
        nyquist_scale = gaussian_nyquist_spatial_scale(-1, 1, volume.shape, eps=0.25) * 1.5 * 4
    else:
        nyquist_scale = gaussian_min_spatial_scale(2, -1, 1)
    upper_scale_bound = gaussian_min_spatial_scale(3, -1, 1)

    scales = reflect_bounds(
        np.random.randn(num_kernels, 3) / 3 * (upper_scale_bound - nyquist_scale) + nyquist_scale,
        min_val=nyquist_scale, max_val=upper_scale_bound
    )
    # L = np.array([[0.30909814, 0., 0., 0.],
    #               [0.06358993, 0.30143564, 0., 0.],
    #               [0.06638153, 0.05370173, 0.29809306, 0.],
    #               [-0.05714649, -0.04542464, -0.04118995, 0.2398241]])
    # mu = np.array([-3.4845378, -3.4861038, -3.4865131, 0.40290555])

    # scales, omega = sample_joint_lognormal(mu, L, num_kernels)

    albedo = np.ones((num_kernels, 3)) * init_albedo
    # omega = np.ones_like(omega) * init_omega

    opacities = (np.random.randn(num_kernels, 1) * 0.25 + 1) * init_opacities

    return {
        'centers': mi.TensorXf(centers), 'scales': mi.TensorXf(scales), 'quaternions': mi.TensorXf(quaternions),
        'opacities': mi.TensorXf(opacities), 'albedo': mi.TensorXf(albedo), 'omega': mi.TensorXf(omega)
    }


def importance_sample_positions(volume: np.ndarray, volume_config: VolumeConfig, n_gmm: int = 1_000, threshold: float = 1e-6) -> np.ndarray:
    # sampling randomly from occupied voxels, not full importance sampling, to ensure wide coverage when using low numbers of kernels
    volume_slice = np.where(np.abs(volume) > threshold)
    volume_slice = np.stack(
        [volume_slice[0], volume_slice[1], volume_slice[2]], axis=1
    )
    starting_points = volume_slice[np.random.permutation(volume_slice.shape[0])[:n_gmm]]

    # project indices to world space
    starting_points_norm = (starting_points / np.array(volume.shape) - 0.5) * volume_config.volume_scale

    return starting_points_norm

# endregion Initialization


# region Theory

def sample_joint_lognormal(mu, L, n_samples):
    z = np.random.randn(n_samples, 4)
    y = mu[None, :] + z @ L.T
    x = np.exp(y)
    s = x[:, :3]
    omega = x[:, [3]]
    return s, omega


def gaussian_downsample(
    voxel_grid: np.ndarray,
    level: int = 4,
    downsample: bool = False,
    c: float = 60.0,
) -> np.ndarray:
    """
    Apply a Gaussian blur (and optionally downsample) to a 3D voxel grid.

    Design goals:
    - `level` controls how strong the blur is: each increment doubles the blur radius
      in *normalized* coordinates (i.e. as a fraction of the domain size).
    - The perceived blur is roughly independent of the grid resolution: a given
      `(level, c)` pair produces similar blur on 64^3 and 256^3 grids when viewed
      in a normalized coordinate system.
    - If `downsample=True`, the grid is decimated by 2**level after blurring.

    Parameters
    ----------
    voxel_grid : np.ndarray
        3D array (H, W, D) representing the voxel grid.
    level : int, default=4
        Blur / pyramid level. Higher levels => stronger blur.
    downsample : bool, default=False
        If True, decimate the voxel grid by a factor of 2**level along each axis
        after blurring.
    c : float, default=60.0
        Global blur-strength constant. Larger `c` => weaker blur at each level.
        Smaller `c` => stronger blur. It controls the base frequency-domain
        standard deviation in normalized units.

    Returns
    -------
    blurred_grid : np.ndarray
        The blurred (and optionally downsampled) voxel grid.
    """

    # -----------------------------------------------
    # 1. Frequency-domain std in normalized units
    # -----------------------------------------------
    # We interpret sigma_f_norm as the frequency-domain standard deviation
    # of a Gaussian in units of "cycles per normalized unit" (e.g. over [0,1]).
    #
    # We choose it to shrink by a factor of 2 with each level:
    #     sigma_f_norm(level) = (c / 2**level)
    #
    # So level+1 has half the frequency std of level; equivalently, twice the
    # spatial blur radius in normalized coordinates.
    sigma_f_norm = c / (2 ** level)  # [cycles / normalized_unit]

    # -----------------------------------------------
    # 2. Convert to spatial std in normalized coords
    # -----------------------------------------------
    # For a Gaussian, spatial and frequency stds satisfy:
    #     sigma_x * sigma_f = 1 / (2 * pi)
    # when sigma_f is in cycles per unit (not angular).
    #
    # Thus:
    #     sigma_x_norm = 1 / (2 * pi * sigma_f_norm)
    #
    # This sigma_x_norm is dimensionless: it is a fraction of the domain size,
    # independent of the actual grid resolution.
    sigma_x_norm = 1.0 / (2.0 * np.pi * sigma_f_norm)  # [normalized_length]

    # -----------------------------------------------
    # 3. Map normalized std to voxel units
    # -----------------------------------------------
    # Suppose the longest grid dimension has N_max voxels, and normalized
    # coordinates u in [0,1] map to voxel coordinates x via x = u * N_max.
    #
    # Then a normalized std sigma_x_norm corresponds to a voxel std:
    #     sigma_vox = sigma_x_norm * N_max
    #
    # This makes the *normalized* blur radius resolution-invariant.
    N_max = max(voxel_grid.shape)
    sigma_vox = sigma_x_norm * N_max  # [voxels]

    # Apply an isotropic Gaussian blur in voxel units.
    blurred = gaussian_filter(voxel_grid, sigma=sigma_vox)

    # -----------------------------------------------
    # 4. Optional downsampling
    # -----------------------------------------------
    if downsample:
        step = 2 ** level
        blurred = blurred[::step, ::step, ::step]

    return blurred


def angular_cutoff_frequency(l: int, a: float, b: float, *, c: float = 60, eps: float = 0.5) -> float:
    """
    Continuous angular cutoff frequency for Gaussian pyramid level `l`.

    Parameters
    ----------
    l : int
        Pyramid / blur level.
    a, b : float
        Bounds of the continuous domain [a, b] into which the voxel grid
        is embedded.
    c : float
        Global blur parameter (as in your original construction).
    eps : float
        Attenuation threshold. Frequencies at and above this omega_c
        are attenuated by at least `eps` in amplitude.

    Returns
    -------
    omega_c : float
        Cutoff angular frequency in radians per unit x.
    """
    L = b - a
    return (2.0 * np.pi * c) / (L * 2**l) * np.sqrt(2.0 * np.log(1.0 / eps))


def gaussian_nyquist_spatial_scale(a: float, b: float, grid_shape: Tuple[int, int, int], *, eps: float = 0.5) -> float:
    """
    Nyquist spatial scale for a voxel grid embedded in [a, b].

    Parameters
    ----------
    a, b : float
        Bounds of the continuous domain [a, b] into which the voxel grid
        is embedded.
    grid_shape : Tuple[int, int, int]
        Shape of the voxel grid (H, W, D).
    eps : float
        Attenuation threshold. Scales at and below this sigma_min
        are attenuated by at least `eps` in amplitude.

    Returns
    -------
    sigma_min : float
        Minimum spatial scale in normalized coordinates.
    """
    L = b - a
    N_max = max(grid_shape)
    return L / (N_max * np.pi) * np.sqrt(2.0 * np.log(1.0 / eps))


def compute_max_freq(volume_shape: Union[Tuple[int, int, int], np.ndarray], continuous_scale: np.ndarray) -> np.ndarray:
    """Compute the maximum frequency that can be represented in the volume given its shape and scale.

    Args:
        volume_shape (np.ndarray): The shape of the volume (depth, height, width).
        continuous_scale (np.ndarray): The continuous scale of the volume (scale_z, scale_y, scale_x).

    Returns:
        np.ndarray: The maximum frequency in each dimension (freq_z, freq_y, freq_x).
    """
    if not isinstance(volume_shape, np.ndarray):
        volume_shape = np.array(volume_shape)
    assert volume_shape.shape == (3,)
    assert continuous_scale.shape == (3,)
    max_freq = np.pi * volume_shape / continuous_scale
    return max_freq


def gaussian_min_spatial_scale(l: int | float, a: float, b: float, *, c: float = 60) -> float:
    """
    Minimum spatial scale for Gaussian pyramid level `l`.

    Parameters
    ----------
    l : int
        Pyramid / blur level.
    a, b : float
        Bounds of the continuous domain [a, b] into which the voxel grid
        is embedded.
    c : float
        Global blur parameter (as in your original construction).

    Returns
    -------
    sigma_min : float
        Minimum spatial scale in normalized coordinates.
    """
    L = b - a
    return (L * 2**l) / (2.0 * np.pi * c)

# endregion Theory


# region Optimization Helpers

def create_lr_dict(args: argparse.Namespace, global_lr: float, level: int) -> Dict[str, float]:
    lr_dict = dict()
    lr_dict[f'centers_{level}'] = global_lr * args.centers_lr
    lr_dict[f'scales_{level}']  = global_lr * args.scales_lr
    lr_dict[f'quaternions_{level}']   = global_lr * args.quats_lr
    lr_dict[f'opacities_{level}']  = global_lr * args.opacity_lr
    lr_dict[f'albedo_{level}']  = global_lr * args.albedo_lr
    lr_dict[f'omega_{level}'] = global_lr * args.omega_lr
    return lr_dict


def set_opt_bounds(opt: gfields.optimizers.BoundedAdam, bounds_dict: Dict[str, float], level: int) -> None:
    for i in range(level+1):
        opt.set_bounds(f'scales_{i}', lower = float(bounds_dict[f'min_scale_{i}']), upper = float(bounds_dict[f'max_scale_{i}'])) # if done correctly, this should cover roughly the Gaussian band passed region
        opt.set_bounds(f'albedo_{i}', lower=1e-9, upper=1.0)
        opt.set_bounds(f'centers_{i}', lower=-1.0, upper=1.0)
        opt.set_bounds(f'opacities_{i}', lower= float(bounds_dict[f'min_opacity_{i}']), upper= float(bounds_dict[f'max_opacity_{i}']))
        opt.set_bounds(f'omega_{i}', lower = float(bounds_dict[f'min_frequency_{i}']), upper = float(bounds_dict[f'max_frequency_{i}'])) # this may not make much sense. Enabling Gabor kernels to be placed in fourier on the edges of the band pass might not be great


def create_scene_dict_from_primitives(
    max_depth: int, 
    level: int, 
    *, 
    centers: mi.TensorXf, 
    scales: mi.TensorXf, 
    quaternions: mi.TensorXf, 
    opacities: mi.TensorXf, 
    albedo: mi.TensorXf, 
    omega: mi.TensorXf,
    extent: float = 3.0,
    is_gabor: Optional[bool] = None,
    kernel_type: Optional[int] = None,
    integrator_type: str = 'gfields_tomography',
    use_kernel_freeze: bool = False,
    use_anyhit: bool = False,
) -> Dict[str, Any]:
    """
    Create a Mitsuba scene dictionary from primitive parameters.
    
    Args:
        max_depth: Maximum integrator depth (-1 for unlimited)
        level: Pyramid level index
        centers: (N, 3) primitive center positions
        scales: (N, 3) primitive scale parameters
        quaternions: (N, 4) primitive rotation quaternions
        opacities: (N, 1) primitive opacity/sigmat values
        albedo: (N, 3) primitive albedo colors
        omega: (N, 1) or (N, 3) Gabor frequency parameters
        extent: Bounding extent in sigma units (default 3.0)
            Higher frequency Gabors can use smaller extents due to exp(-ω²/2) decay.
            See gfields.training.adaptive for computing optimal per-level extents.
        is_gabor: Whether primitives are Gabor (True) or Gaussian (False).
            Default: level != 0 (level 0 is Gaussian, higher levels are Gabor)
        integrator_type: Mitsuba integrator plugin name.

    Returns:
        Scene dictionary ready for mi.load_dict()
    """
    if is_gabor is None:
        is_gabor = level != 0

    integrator: Dict[str, Any] = {
        'type': integrator_type,
        'max_depth': max_depth,
        'kernel_normalized': True,
        'kernel_full_range': True,
        'kernel_type': "gabor_gaussian_mixture",
        'render_marginalization': False,
    }
    if use_kernel_freeze:
        integrator['use_kernel_freeze'] = True
    if use_anyhit:
        # Primal renders via one OptiX any-hit traversal per ray; the adjoint
        # pass keeps the sequential walk (identical estimand).
        integrator['use_anyhit'] = True
        
    # Kernel marker. The C++ shape accepts the general integer ``kernel_type``
    # (0=Gaussian, 1=Gabor, ...) and the legacy boolean ``is_gabor`` as a
    # shorthand for kernel_type=1. We always emit ``is_gabor`` (understood by
    # every build) and additionally emit the integer ``kernel_type`` only when a
    # caller explicitly requests one (requires a build that supports it).
    prim_dict: Dict[str, Any] = {
        'type': 'ellipsoidsmesh',
        'shell': 'uv_sphere',
        'visibility_mask': 255,
        'omega': omega,
        'centers': centers,
        'scales': scales,
        'quaternions': quaternions,
        'opacities': opacities,
        'albedo': albedo,
        'extent': float(extent),
        # 'extent_adaptive_clamping': True,
        'is_gabor': bool(is_gabor) if is_gabor is not None else (kernel_type == 1),
    }
    if kernel_type is not None:
        prim_dict['kernel_type'] = int(kernel_type)
    scene_dict = {
        'type': 'scene',
        'integrator': integrator,
        f'primitives_pyr{level}': prim_dict,
        'environment': { 'type': 'constant' }
    }
    return scene_dict


def write_scene_params_to_optimizer(params: mi.SceneParameters, opt: mi.ad.Optimizer, level: int) -> None:
    """
    Write the scene parameters to the optimizer for the given level.
    This function unravels the parameters and updates the optimizer variables.
    """
    opt[f'omega_{level}'] = params[f'primitives_pyr{level}.omega']
    ellipsoid = Ellipsoid.unravel(params[f'primitives_pyr{level}.data'], params[f'primitives_pyr{level}.omega'])
    opt[f'centers_{level}'] = ellipsoid.center
    opt[f'scales_{level}']  = ellipsoid.scale
    opt[f'quaternions_{level}']   = mi.Vector4f(ellipsoid.quat)
    opt[f'opacities_{level}']  = params[f'primitives_pyr{level}.opacities']
    opt[f'albedo_{level}']  = dr.unravel(mi.Color3f, params[f'primitives_pyr{level}.albedo'])


def write_optimizer_params_to_scene(opt: mi.ad.Optimizer, params: mi.SceneParameters, level: int) -> None:
    """
    Write the optimized parameters to the params dictionary for the given level.
    This function normalizes the quaternions and updates the parameters for the ellipsoids mesh.
    """
    opt[f'quaternions_{level}'] = dr.normalize(opt[f'quaternions_{level}'])
    params[f'primitives_pyr{level}.data']    = Ellipsoid.ravel(opt[f'centers_{level}'], opt[f'scales_{level}'], mi.Quaternion4f(opt[f'quaternions_{level}']))
    params[f'primitives_pyr{level}.opacities'] = opt[f'opacities_{level}']
    params[f'primitives_pyr{level}.albedo']  = dr.ravel(opt[f'albedo_{level}'])
    params[f'primitives_pyr{level}.omega']   = opt[f'omega_{level}']
    params.update()

# endregion Opimization Helpers


# region I/O

def save_model(output_path: Path, scene: mi.Scene, level: int) -> None:
    """
    Save the final optimized model to the output path in numpy format.
    The folder structure is created as follows:
    - output_path/optimized_asset_pyr{level}/
        - npy_data/
            - omega_pyr{0..level}.npy
            - centers_pyr{0..level}.npy
            - scales_pyr{0..level}.npy
            - quaternions_pyr{0..level}.npy
            - opacities_pyr{0..level}.npy
            - albedo_pyr{0..level}.npy
        - data/
            - root.primitives_pyr{0..level}.ply
        - __init__.py
    """
    asset_dir = output_path / f'optimized_asset_pyr{level}'
    (asset_dir / 'npy_data').mkdir(parents=True, exist_ok=True)  # opt_state (Adam moments) still lives here
    final_asset = gfields.io.object_to_dict(scene)
    # Canonical asset storage: one .ply per pyramid level under <asset>/data/
    # (packs centers/scales/quaternions/opacities/omega/albedo/extent). The
    # former per-array npy_data/*_pyr*.npy duplicated this and has been retired.
    gfields.io.dict_to_asset(final_asset, str(asset_dir))


def save_optimizer_state(opt: 'gfields.optimizers.BoundedAdam', output_path: Path, level: int) -> None:
    '''Persist the Adam per-parameter moments (t, m, v) and the BoundedAdam noise
    level alongside the asset checkpoint, so a --from_gabor_checkpoint resume can
    restore them and avoid the Adam cold-start dip (first step uses v=0 -> ~grad/sqrt(eps),
    which kicks a converged solution off by ~10 dB).  Stored under
    optimized_asset_pyr{level}/npy_data/opt_state/ so the same path used for
    --from_gabor_checkpoint also locates the optimizer state.'''
    state_dir = Path(output_path) / f'optimized_asset_pyr{level}' / 'npy_data' / 'opt_state'
    state_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {'noise_lr': float(getattr(opt, 'noise_lr', 0.0)), 'steps': {}}
    for key, (_val, _bool, _lr, extra) in opt.state.items():
        t, m, v, *_rest = extra
        np.save(state_dir / f'{key}__m.npy', np.array(m))
        np.save(state_dir / f'{key}__v.npy', np.array(v))
        meta['steps'][key] = int(np.array(t).reshape(-1)[0])
    (state_dir / 'meta.json').write_text(json.dumps(meta))


def load_optimizer_state(opt: 'gfields.optimizers.BoundedAdam', ckpt_path: Path, level: int) -> int:
    '''Restore Adam moments written by save_optimizer_state into a freshly-built
    optimizer (params already set from the checkpoint, so m/v are zero with the
    right shape).  ckpt_path is the --from_gabor_checkpoint dir (the npy_data
    folder).  Only restores entries whose shape matches the current optimizer
    (so a count change from densify/prune/gabor_count safely skips them).
    Returns the number of parameters restored.'''
    ckpt = Path(ckpt_path)
    state_dir = ckpt / 'opt_state'
    if not state_dir.exists():
        state_dir = ckpt / 'npy_data' / 'opt_state'
    if not (state_dir / 'meta.json').exists():
        return 0
    meta = json.loads((state_dir / 'meta.json').read_text())
    if hasattr(opt, 'noise_lr') and 'noise_lr' in meta:
        opt.noise_lr = float(meta['noise_lr'])
    restored = 0
    for key in list(opt.state.keys()):
        mf, vf = state_dir / f'{key}__m.npy', state_dir / f'{key}__v.npy'
        if not (mf.exists() and vf.exists()):
            continue
        val, b, lr, extra = opt.state[key]
        t, m, v, *rest = extra
        m_np, v_np = np.load(mf), np.load(vf)
        if m_np.shape != np.array(m).shape:
            continue  # primitive count changed; skip (cold-start that param)
        t_val = int(meta.get('steps', {}).get(key, 0))
        opt.state[key] = (val, b, lr, (type(t)(t_val), type(m)(m_np), type(v)(v_np), *rest))
        restored += 1
    return restored


def load_single_level(path: Path, level: int) -> Dict[str, np.ndarray]:
    # Canonical unified .ply store, under <asset>/data/root.primitives_pyr{level}.ply.
    import gfields.io
    ply = Path(path).parent / 'data' / f'root.primitives_pyr{level}.ply'
    if not ply.exists():
        raise FileNotFoundError(
            f"Unified asset ply not found: {ply}. Legacy per-array npy assets are "
            f"no longer read directly -- convert them with `gfields migrate assets`.")
    a = gfields.io.load_ellipsoid_ply(ply)
    n = len(a['centers'])
    return {p: mi.TensorXf(np.ascontiguousarray(a[p], dtype=np.float32).reshape(n, -1))
            for p in ['omega', 'centers', 'scales', 'quaternions', 'opacities', 'albedo'] if p in a}


def load_model(path: Path, max_depth: int, integrator_type: str = 'gfields_tomography') -> dict:
    if len(output_root := list(path.glob("optimized_asset_pyr*"))) > 0:
        # * will be one number, pick the path with the highest number
        path = sorted(output_root, key=lambda x: int(x.name.split('_pyr')[1]))[-1]
        yellow(f"- Found multiple optimized asset paths, loading the latest one: {str(path)}. If you want to load a specific one, please provide the path directly.")

    if (data_dir := path / "npy_data").exists():
        path = data_dir

    integrator: Dict[str, Any] = {
        'type': integrator_type,
        'max_depth': max_depth,
        'kernel_normalized': True,
        'kernel_full_range': True,
        'kernel_type': "gabor_gaussian_mixture",
    }

    prims_dict = {
        'type': 'scene',
        'integrator': integrator,
        'environment': { 'type': 'constant' }
    }
    # Discover pyramid levels from the canonical unified store: one
    # root.primitives_pyr*.ply per level under <asset>/data/ (written by
    # save_model). Legacy per-array centers_pyr*.npy are no longer read.
    import gfields.io
    ply_dir = (path.parent if path.name == 'npy_data' else path) / 'data'
    levels = sorted(int(f.name.split('_pyr')[-1].split('.')[0])
                    for f in ply_dir.glob('root.primitives_pyr*.ply'))
    for level in levels:
        layer_dict: Dict[str, Any] = load_single_level(path, level)

        # Extent is packed into the canonical .ply by dict_to_asset; default 3.0.
        ply_path = ply_dir / f'root.primitives_pyr{level}.ply'
        if 'extent' in (a := gfields.io.load_ellipsoid_ply(ply_path)):
            extent = mi.TensorXf(np.ascontiguousarray(
                a['extent'], dtype=np.float32).reshape(len(a['centers']), -1))
        else:
            extent = 3.0

        layer_dict.update({
            'type': 'ellipsoidsmesh',
            'shell': 'uv_sphere',
            'visibility_mask': 255,
            'extent': extent,
            # 'extent_adaptive_clamping': True,
            'is_gabor': level != 0,  # Gabor primitives are only used for levels > 0
        })
        prims_dict[f'primitives_pyr{level}'] = layer_dict

    return prims_dict

# endregion I/O


# region Densify, Prune and Regularize


def relocate_dead_primitives(opt: gfields.optimizers.BoundedAdam, params: mi.SceneParameters, level: int, threshold: float = 1e-6,
                             dead_scale: float = 1e-4) -> int:
    """
    Relocate dead primitives to positions of contributing ones (MCMC-style
    recycling). A primitive is dead when its opacity has decayed below
    ``threshold`` *or* its scale has collapsed below ``dead_scale`` (in the
    [-1, 1] volume a 1e-4-sigma primitive is far below a pixel at any
    realistic training resolution): collapsed primitives receive no data
    gradient and, because the MCMC center noise is scaled by the covariance,
    essentially no exploration noise either -- without this criterion they
    stay frozen at the optimizer's scale bound forever. ``dead_scale`` is
    clamped to never exceed the level's own lower scale bound (the Gabor
    Nyquist bound is a *healthy* size; sitting on it is not degeneracy).
    Returns the number of relocated primitives.
    """
    opacities = opt[f'opacities_{level}']
    scales = opt[f'scales_{level}']

    # Only scales *strictly below* a legitimate lower bound are degenerate;
    # if the bound itself is >= dead_scale (e.g. the Gabor Nyquist scale),
    # the scale criterion is disabled for this level.
    bnd = opt.bounds.get(f'scales_{level}') if hasattr(opt, 'bounds') else None
    lower = bnd[1] if bnd is not None and bnd[1] is not None else 0.0
    scale_floor = dead_scale if float(lower) < dead_scale else 0.0

    dead_mask = dr.abs(opacities) < threshold
    if scale_floor > 0.0:
        dead_mask |= dr.min(dr.abs(scales)) < scale_floor
    dead_indices = dr.compress(dead_mask)
    num_dead = int(dr.width(dead_indices))

    if num_dead == 0:
        return 0  # No dead primitives to relocate

    num_prims = int(dr.width(opacities))

    alive_mask = ~dead_mask
    alive_indices = dr.compress(alive_mask)
    if int(dr.width(alive_indices)) == 0:
        return 0  # degenerate state: nothing to clone from
    alive_opacities = dr.abs(opacities[alive_indices])

    # Sample live primitives with replacement to get target indices for the
    # dead ones, proportional to |opacity| (the 3DGS-MCMC convention).
    # NOTE: this was previously proportional to opacity / volume, which is
    # catastrophic in the presence of near-zero scales: a collapsed primitive
    # has astronomical point density, dominates the target distribution, and
    # every relocation round then *clones its degenerate geometry onto all
    # dead primitives* (legacy assets accumulated up to 33% collapsed
    # primitives this way).
    sampling_probs = alive_opacities / dr.sum(alive_opacities)
    probs = np.nan_to_num(np.asarray(sampling_probs.numpy(), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    total = probs.sum()
    probs = probs / total if total > 0 else np.full(probs.shape, 1.0 / probs.size)
    sampled_alive_indices = np.random.choice(alive_indices.numpy(), size=num_dead, replace=True, p=probs)
    counts = np.bincount(sampled_alive_indices, minlength=num_prims)

    mi_sampled_alive_indices = mi.UInt(sampled_alive_indices)
    opt[f'opacities_{level}'] /= dr.maximum(mi.Float(counts), 1.0)


    # Adam per-parameter state inner tuple is (t, m, v) on Dr.Jit <=1.3 and
    # (t, m, v, extra) on 1.4 (the trailing slot is reserved, e.g. AMSGrad's
    # running max); accept either arity.
    for key, (_val, _bool, _lr, (_t, m, v, *_rest)) in opt.state.items():
        if str(level) not in key:
            continue  # skip unrelated parameters
        new_prop = dr.gather(type(opt[key]), opt[key], mi_sampled_alive_indices)
        dr.scatter(opt[key], new_prop, dead_indices)
        
        # reset moment statistics for resampled live primitives to keep them stable
        dr.scatter(m, type(m)(0.), mi_sampled_alive_indices)
        dr.scatter(v, type(v)(0.), mi_sampled_alive_indices)
    
    write_optimizer_params_to_scene(opt, params, level)

    print(f"Relocated {num_dead} dead primitives at level {level}.")

    dr.enable_grad(opt.variables)
    return num_dead



def add_primitives_to_opt_and_scene(opt: mi.ad.Optimizer, params: mi.SceneParameters, level: int, kernels: Dict[str, mi.TensorXf]) -> None:
    """
    sample a new set of primitives to add during optimization
    """
    num_prev = dr.width(opt[f'opacities_{level}'])
    num_kernels = dr.width(kernels['opacities'])
    old_idx = dr.arange(mi.UInt32, num_prev)
    new_idx = dr.arange(mi.UInt32, num_kernels) + num_prev

    for key, new_vals in kernels.items():
        old_vals = opt[f'{key}_{level}']
        # Build the new-primitive values as type(old_vals), robust to the source
        # layout. 1D params (Float: opacities/omega) need a flat (num_kernels,)
        # array; vector params (Vector3f/Quaternion4f) need component-major
        # (C, num_kernels) -- np.array(Vector3f) is (C, N). The old code did
        # type(old_vals)(2D) which crashed for the Float params.
        nv = np.asarray(new_vals.numpy() if hasattr(new_vals, 'numpy') else new_vals)
        nv = np.squeeze(nv)
        if old_vals.ndim == 1:
            nv = nv.reshape(-1)
        else:
            comp = int(np.asarray(old_vals).shape[0])  # 3 (Vector3f) or 4 (Quaternion4f)
            if nv.ndim == 1:
                nv = nv.reshape(comp, -1)
            if nv.shape[0] != comp and nv.shape[-1] == comp:
                nv = nv.T  # (N, C) -> (C, N)
        new_typed = type(old_vals)(nv)

        # Grab the current Adam moments so densification does NOT reset them
        # (opt[key] = buf zeroes m/v at the new size, which would cold-start every
        # existing primitive -> a localized version of the resume dip).
        had_state = f'{key}_{level}' in opt.state
        if had_state:
            _v, _b, _lr, (t_old, m_old, v_old, *rest) = opt.state[f'{key}_{level}']

        buf = dr.empty(type(old_vals), num_prev + num_kernels)
        dr.scatter(buf, old_vals, old_idx)
        dr.scatter(buf, new_typed, new_idx)
        opt[f'{key}_{level}'] = buf  # resets m/v to zero at the grown size

        if had_state:
            _v2, _b2, _lr2, (t_new, m_new, v_new, *rest2) = opt.state[f'{key}_{level}']
            dr.scatter(m_new, m_old, old_idx)  # restore old momentum into old slots
            dr.scatter(v_new, v_old, old_idx)  # new slots stay zero (correct cold start)
            opt.state[f'{key}_{level}'] = (_v2, _b2, _lr2, (t_old, m_new, v_new, *rest2))

    write_optimizer_params_to_scene(opt, params, level)


def densify(opt: gfields.optimizers.BoundedAdam, scene_params: mi.SceneParameters, level: int, args: argparse.Namespace, volumes: List[np.ndarray], volume_config: VolumeConfig) -> None:
    """
    Densify the primitives by adding new ones based on the current volume.
    This function generates new primitives and appends them to the optimizer.
    """
    new_prims = generate_random_kernels(
        args.densify_count, args.init_opacity, args.init_omega, args.init_scale, args.init_albedo, level,
        importance_sample_means=args.importance_sample_means,
        volume=np.abs(volumes[0] - volumes[-1]),
        volume_config=volume_config
    )

    if level > 0:
        scales: np.ndarray = opt[f'scales_{level}'].numpy()
        scales_mean = np.mean(scales, axis=1)
        scales_std = np.std(scales, axis=1)
        new_scales = (
            (np.random.randn(args.densify_count, 3) * scales_std * 0.7 + scales_mean)
            .clip(min=opt.bounds[f'scales_{level}'][0])
        )
        new_prims['scales'] = mi.TensorXf(new_scales)

        # shift newly added frequencies up by one stddev of the current frequencies
        b: mi.Float = dr.norm(opt['omega_1'] / opt['scales_1'])
        b = b.numpy()
        bmean = b.mean()
        bstd = b.std()

        omega = (
            (np.random.randn(args.densify_count) * bstd + bmean + 2. * bstd)
            .clip(min=opt.bounds[f'omega_{level}'][0])
        )
        new_prims['omega'] = mi.TensorXf(omega) * dr.norm(1 / mi.TensorXf(new_scales.T))

    add_primitives_to_opt_and_scene(
        opt, scene_params, level, new_prims
    )


def prune(opt: mi.ad.Optimizer, scene_params: mi.SceneParameters, level: int, rule: Literal['density', 'opacity'] = 'density', custom_th: Optional[float] = None,
          dead_scale: float = 1e-4) -> int:
    if rule == 'density':
        # Clamp the volume so collapsed scales cannot inflate the weight
        # (a near-zero-scale primitive would otherwise look infinitely dense
        # and be *protected* by this rule).
        weight = opt[f'opacities_{level}'] / (dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi)
                                              * dr.maximum(dr.prod(opt[f'scales_{level}']), dead_scale ** 3))
        th = 0.001
    elif rule == 'opacity':
        weight = dr.abs(opt[f'opacities_{level}'])
        th = 1e-5
    else:
        raise ValueError(f"Unknown pruning rule: {rule}")

    if custom_th is not None:
        th = custom_th

    valid = (weight > th) # Pruning threshold based on density and scale 1e-6 and 1e-4 originally

    # Scale-collapsed primitives are invisible at render resolution (no data
    # gradient, ~zero covariance-scaled MCMC noise) and must not survive to
    # the final asset. Only applies when the level's own lower scale bound is
    # itself degenerate (< dead_scale): sitting on a healthy bound (e.g. the
    # Gabor Nyquist scale) is not collapse.
    bnd = getattr(opt, 'bounds', {}).get(f'scales_{level}')
    lower = bnd[1] if bnd is not None and bnd[1] is not None else 0.0
    if dead_scale > 0.0 and float(lower) < dead_scale:
        valid &= dr.min(dr.abs(opt[f'scales_{level}'])) >= dead_scale
    valid_indices = dr.compress(valid)
    num_pruned: int = dr.width(opt[f'opacities_{level}']) - dr.width(valid_indices)
    print(f"\nPruning ({num_pruned}/{dr.width(weight)}) volumetric primitives")
    if dr.width(valid_indices) == dr.width(weight):
        return 0  # nothing to prune, all primitives are valid
    opt[f'centers_{level}'] = dr.gather(mi.Point3f,  opt[f'centers_{level}'], valid_indices)
    opt[f'scales_{level}']  = dr.gather(mi.Vector3f, opt[f'scales_{level}'],  valid_indices)
    opt[f'opacities_{level}']  = dr.gather(mi.Float,    opt[f'opacities_{level}'],  valid_indices)
    opt[f'albedo_{level}']  = dr.gather(mi.Color3f,  opt[f'albedo_{level}'],  valid_indices)
    opt[f'quaternions_{level}']   = dr.gather(mi.Vector4f, dr.ravel(opt[f'quaternions_{level}']), valid_indices)
    opt[f'omega_{level}']   = dr.gather(mi.Float,  opt[f'omega_{level}'], valid_indices)

    write_optimizer_params_to_scene(opt, scene_params, level)
    return num_pruned


def frequency_regularization(opt: mi.ad.Optimizer, ref_volume: np.ndarray, volume_config: VolumeConfig, level: int, weight: float = 0.1) -> mi.TensorXf:
    """Penalize primitives whose spatial frequency ``‖ω/scale‖`` exceeds the
    volume's representable band limit (`max_freq = π·shape/scale`), to keep the
    mixture band-limited / anti-aliased.

    Returns ``weight`` times the **mean** super-Nyquist indicator over primitives
    — a fraction in [0, 1], independent of the primitive count. (The previous
    ``dr.sum`` returned a raw count that both scaled with the primitive count and
    carried no coefficient, so on low-resolution volumes — e.g. smoke, whose band
    limit is ~4× lower than the paper clouds — it reached hundreds and swamped the
    normalized image loss, collapsing quality at high budgets. High-resolution
    assets have ~0 violators, so this change is a no-op for them.)"""
    loss = mi.TensorXf(0.0)
    for lv in range(1, level + 1):
        f = dr.norm(opt[f'omega_{lv}'] / opt[f'scales_{lv}'])
        max_freq = np.max(compute_max_freq(ref_volume.shape, volume_config.volume_scale))
        loss += dr.mean(sigmoid(f - max_freq, a=10))
    return weight * loss


# region Cache Metadata Functions


def _compute_file_hash(file_path: str) -> str:
    """Compute a hash of a file's contents for cache invalidation."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cache_root() -> Path:
    """Root of the SHARED reference / pyramid cache.

    Reference images and the pyramid decomposition are fully determined by their
    cache params (volume hash, camera set, spp, albedo, target levels, ...), so
    they are content-addressed under this root and reused across output dirs and
    across runs -- re-optimizing the same asset never re-renders the references.
    Override the location with the ``GFIELDS_CACHE_DIR`` environment variable.
    """
    env = os.environ.get('GFIELDS_CACHE_DIR')
    if env:
        return Path(env)
    from .._paths import REPO_ROOT
    return REPO_ROOT / 'results' / 'reference_cache'


def _shared_cache_dir(kind: str, params: Dict[str, Any]) -> Path:
    """Content-addressed directory for one cache entry (``kind`` is 'refs' or
    'pyramid'). The params already include everything that changes the output
    (e.g. the pyramid ``level``), so each distinct configuration maps to its own
    directory and a matching run finds it regardless of ``--output``."""
    digest = hashlib.sha1(
        json.dumps(params, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
    d = _cache_root() / kind / digest
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_pyramid_cache_params(volume_grid_path: str, gauss_target_level: int, gabor_target_level: int) -> Dict[str, Any]:
    """Get the parameters that affect pyramid generation for caching."""
    return {
        'volume_grid_path': str(Path(volume_grid_path).resolve()),
        'volume_grid_hash': _compute_file_hash(volume_grid_path),
        'gauss_target_level': gauss_target_level,
        'gabor_target_level': gabor_target_level,
    }


def _get_reference_cache_params(
        volume_grid_path: str,
        cam_res_x: int,
        cam_res_y: int,
        cam_count: int,
        ref_spp: int,
        cam_scale: float,
        sample_single_hemisphere: bool,
        init_albedo: float,
        opacity_scale: float,
        level: int,
        target_level: int,
) -> Dict[str, Any]:
    """Get the parameters that affect reference image generation for caching."""
    return {
        'volume_grid_path': str(Path(volume_grid_path).resolve()),
        'volume_grid_hash': _compute_file_hash(volume_grid_path),
        'cam_res_x': cam_res_x,
        'cam_res_y': cam_res_y,
        'cam_count': cam_count,
        'ref_spp': ref_spp,
        'cam_scale': cam_scale,
        'sample_single_hemisphere': sample_single_hemisphere,
        'init_albedo': init_albedo,
        'opacity_scale': opacity_scale,
        'level': level,
        'target_level': target_level,
    }


def _save_cache_metadata(cache_path: Path, params: Dict[str, Any]) -> None:
    """Save cache metadata to a JSON file."""
    with open(cache_path, 'w') as f:
        json.dump(params, f, indent=2)


def _load_cache_metadata(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load cache metadata from a JSON file, returns None if file doesn't exist."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _cache_is_valid(cache_path: Path, current_params: Dict[str, Any]) -> bool:
    """Check if the cache is valid by comparing current params with cached params."""
    cached_params = _load_cache_metadata(cache_path)
    if cached_params is None:
        return False
    return cached_params == current_params


def check_pyramid_cache_valid(volume_grid_path: str, output_dir: Path, gauss_target_level: int, gabor_target_level: int) -> bool:
    """Check if pyramid cache is valid without loading the grid."""
    pyr_paths = [(output_dir / f'grid_pyr{lvl}.npy') for lvl in range(5)]
    cache_metadata_path = output_dir / 'pyramid_cache_metadata.json'
    current_params = _get_pyramid_cache_params(volume_grid_path, gauss_target_level, gabor_target_level)
    return _cache_is_valid(cache_metadata_path, current_params) and all(f.exists() for f in pyr_paths)


def check_reference_cache_valid(
        volume_grid_path: str,
        output_dir: Path,
        level: int,
        cam_res_x: int,
        cam_res_y: int,
        cam_count: int,
        ref_spp: int,
        cam_scale: float,
        sample_single_hemisphere: bool,
        init_albedo: float,
        opacity_scale: float,
        target_level: int,
) -> bool:
    """Check if reference image cache is valid without loading anything."""
    output_path = output_dir / f'refs_pyr{level}'
    cache_metadata_path = output_dir / f'refs_pyr{level}_cache_metadata.json'
    ref_image_paths = get_exr_files(output_path)

    current_params = _get_reference_cache_params(
        volume_grid_path=volume_grid_path,
        cam_res_x=cam_res_x,
        cam_res_y=cam_res_y,
        cam_count=cam_count,
        ref_spp=ref_spp,
        cam_scale=cam_scale,
        sample_single_hemisphere=sample_single_hemisphere,
        init_albedo=init_albedo,
        opacity_scale=opacity_scale,
        level=level,
        target_level=target_level
    )
    return _cache_is_valid(cache_metadata_path, current_params) and len(ref_image_paths) == cam_count


def load_cached_reference_images(output_dir: Path, level: int) -> List[mi.TensorXf]:
    """Load cached reference images from disk."""
    output_path = output_dir / f'refs_pyr{level}'
    ref_image_paths = sorted(get_exr_files(output_path))
    green(f'Loading {len(ref_image_paths)} cached reference images for level {level}.')
    ref_images = [mi.TensorXf(mi.Bitmap(path)) for path in ref_image_paths]
    return [dr.clip(image, 0.0, 1.0) for image in ref_images]


def load_cached_pyramid(output_dir: Path) -> List[np.ndarray]:
    """Load cached pyramid from disk without loading the original voxel grid."""
    pyr_paths = [(output_dir / f'grid_pyr{lvl}.npy') for lvl in range(5)]
    green("Loading cached pyramid (skipping original voxel grid load).")
    return [np.load(f) for f in pyr_paths]


# endregion Cache Metadata Functions


# region Utils

def reflect_bounds(values: np.ndarray, min_val: float | None, max_val: float | None):
    if min_val is not None:
        values = np.where(values < min_val, values + 2 * (min_val - values), values)
    if max_val is not None:
        values = np.where(values > max_val, values - 2 * (values - max_val), values)
    return values


def estimate_memory_budget(
    *,
    cam_res_x: int,
    cam_res_y: int,
    cam_count: int,
    ref_spp: int,
    ref_spp_chunk: int = 1024,
    opt_cam_batch: int = 0,
    gabor_count: int = 0,
    gauss_count: int = 0,
    integrator: str = 'gfields_tomography',
) -> dict[str, float]:
    '''Rough GPU memory estimates (MiB) for planning large runs.'''
    opt_batch = cam_count if opt_cam_batch <= 0 else min(opt_cam_batch, cam_count)
    n_opt_rays = cam_res_x * cam_res_y * opt_batch
    n_ref_rays = cam_res_x * cam_res_y

    opt_image_mib = n_opt_rays * 3 * 4 / (1024 ** 2)
    ref_image_mib = n_ref_rays * 3 * 4 / (1024 ** 2)
    concat_ref_mib = cam_count * ref_image_mib

    chunk = max(1, min(ref_spp_chunk, ref_spp))
    volpath_state_mib = n_ref_rays * chunk * 16 / (1024 ** 2)

    num_primitives = gauss_count + gabor_count

    return {
        'opt_rays_per_render': float(n_opt_rays),
        'opt_image_mib': opt_image_mib,
        'ref_volpath_peak_mib_est': volpath_state_mib,
        'ref_images_concat_mib': concat_ref_mib,
        'primitive_count': float(num_primitives),
    }


def render_sensor_chunked(
    scene: mi.Scene,
    sensor: mi.Sensor,
    spp: int,
    *,
    seed: int = 0,
    chunk_spp: int = 1024,
) -> mi.TensorXf:
    '''
    Render with ``spp`` total samples using multiple lower-SPP passes.

    Peak device memory scales with ``chunk_spp`` instead of ``spp``.
    '''
    if spp <= 0:
        raise ValueError('spp must be positive')
    chunk_spp = max(1, chunk_spp)
    if spp <= chunk_spp:
        return mi.render(scene, sensor=sensor, spp=spp, seed=seed)

    accum: mi.TensorXf | None = None
    done = 0
    pass_id = 0
    while done < spp:
        n = min(chunk_spp, spp - done)
        img = mi.render(scene, sensor=sensor, spp=n, seed=seed + pass_id)
        contrib = img * float(n)
        accum = contrib if accum is None else (accum + contrib)
        done += n
        pass_id += 1
        dr.eval(accum)
        del img, contrib
        dr.flush_malloc_cache()
    return accum / float(spp)


def create_cam_array(cameras: List[mi.ProjectiveCamera]) -> mi.Sensor:
    cam_res_x, cam_res_y = cameras[0].film().size()
    batch_sensor_dict: Dict[str, Any] = {
        'type': 'batch',
        'film': {
            'type': 'hdrfilm',
            'width': cam_res_x * len(cameras), 'height': cam_res_y,
            'filter': { 'type': 'tent' },
        }
    }
    for i, camera in enumerate(cameras):
        batch_sensor_dict[f'cam_{i:04d}'] = camera
    batch_sensor: mi.Sensor = mi.load_dict(batch_sensor_dict)
    return batch_sensor


def iter_cam_batches(
    cameras: List[mi.ProjectiveCamera],
    batch_size: int,
) -> list[tuple[mi.Sensor, slice]]:
    '''Yield (batch_sensor, slice) groups for memory-bounded optimization.'''
    if batch_size <= 0 or batch_size >= len(cameras):
        return [(create_cam_array(cameras), slice(0, len(cameras)))]
    out: list[tuple[mi.Sensor, slice]] = []
    for start in range(0, len(cameras), batch_size):
        batch = cameras[start:start + batch_size]
        out.append((create_cam_array(batch), slice(start, start + len(batch))))
    return out


def slice_batched_ref_image(
    ref_image: mi.TensorXf,
    cam_res_x: int,
    sl: slice,
) -> mi.TensorXf:
    '''Extract the horizontal strip of a concatenated multi-camera reference image.'''
    start = sl.start or 0
    stop = sl.stop
    return ref_image[:, start * cam_res_x:stop * cam_res_x, :]


def from_camera_array_to_batched(tensor: mi.TensorXf, cam_count: int) -> mi.TensorXf:
    H, Wn, C = tensor.shape
    W = Wn // cam_count
    assert Wn % cam_count == 0, f"Tensor width {Wn} is not divisible by camera count {cam_count}"
    return dr.reshape(type(tensor), tensor, (H, cam_count, W, C)).transpose(1, 2, 0, 3)


def batch(l: list, fn: Callable, batch_size: int) -> list:
    n = len(l) // batch_size
    if len(l) % batch_size != 0:
        n += 1
    return [fn(l[i*batch_size:(i+1)*batch_size]) for i in range(n)]


def sigmoid(x: mi.TensorXf, a: float = 1) -> mi.TensorXf:
    return 1 / (1 + dr.exp(-a * x))


def sample_uniform_hemisphere(k, normal=(0, 0, 1)):
    uv = halton_2d(k)
    z = uv[:, 0]
    phi = 2 * np.pi * uv[:, 1]
    r = np.sqrt(1 - z * z)
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    n = np.asarray(normal) / np.linalg.norm(normal)
    if not np.allclose(n, [0, 0, 1]):
        h = np.array([1, 0, 0]) if abs(n[2]) > 0.7 else np.array([0, 0, 1])
        t = np.cross(h, n); t /= np.linalg.norm(t)
        b = np.cross(n, t)
        dirs = dirs @ np.stack([t, b, n], 1)
    return dirs


def halton_2d(n, b1=2, b2=3):
    def vdc(k, base):
        x, f = 0.0, 1.0 / base
        while k:
            x += f * (k % base)
            k //= base
            f /= base
        return x
    return np.array([[vdc(i, b1), vdc(i, b2)] for i in range(n)])


def visualize_slices(original, gaussian, aggregated, mode = 'slice'):
    if aggregated is not None:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        titles = ['Original', 'Gaussian Downsampled', 'Box Downsampled']
        data = [original, gaussian, aggregated]
    else:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        titles = ['Original', 'Downsampled']
        data = [original, gaussian]
    if mode == 'slice':
        for i, grid in enumerate(data):
            mid_x, mid_y, mid_z = [s // 2 for s in grid.shape]
            axes[i, 0].imshow(grid[mid_x, :, :], cmap='inferno')  # X slice
            axes[i, 1].imshow(grid[:, mid_y, :], cmap='inferno')  # Y slice
            axes[i, 2].imshow(grid[:, :, mid_z], cmap='inferno')  # Z slice
            for j in range(3):
                axes[i, j].set_title(f'{titles[i]} - Slice {"XYZ"[j]}')
                axes[i, j].axis('off')
            axes[i, 3].imshow((np.sum(grid, axis=2) - np.sum(original, axis=2)) ** 2.0, cmap='inferno')
            axes[i, 3].set_title(f'Squared Error between {titles[i]} and original (Integrals over Z)')
    elif mode == 'marginalize':
        for i, grid in enumerate(data):
            axes[i, 0].imshow(np.sum(grid, axis=0), cmap='inferno')
            axes[i, 1].imshow(np.sum(grid, axis=1), cmap='inferno')
            axes[i, 2].imshow(np.sum(grid, axis=2), cmap='inferno')
            for j in range(3):
                axes[i, j].set_title(f'{titles[i]} - Integral over {"XYZ"[j]}')
                axes[i, j].axis('off')
            axes[i, 3].imshow((np.sum(grid, axis=2) - np.sum(original, axis=2)) ** 2.0, cmap='inferno')
            axes[i, 3].set_title(f'Squared Error between {titles[i]} and original (Integrals over Z)')

    plt.tight_layout()
    plt.show()


def plot_loss(data: List[float], label: str, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(data)
    fig.canvas.toolbar_visible = 'fade-in-fade-out'
    fig.canvas.footer_visible = False
    fig.canvas.header_visible = False
    ax.set_xlabel('Iteration')
    plt.ylabel(label)
    plt.title(label + ' plot')
    plt.savefig(str(output_file))
    plt.close()


def get_exr_files(folder_path: str) -> List[str]:
    """
    Gather all .exr files in a specified folder and return them as a list.

    :param folder_path: Path to the folder containing .exr files
    :return: List of .exr file paths
    """
    return [join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".exr")]


class GradientTracker:
    """
    A class to track parameter statistics during optimization.
    It collects the gradients and computes statistics like mean, variance, and maximum.
    """
    def __init__(self, track_grads: bool = True) -> None:
        self.statistics: Dict[str, Dict[str, List[float]]] = {}
        self.track_grads = track_grads

    def track(self, opt: mi.ad.Optimizer) -> None:
        if not self.track_grads:
            return
        for var_name, var in opt.variables.items():
            if var_name not in self.statistics:
                self.statistics[var_name] = defaultdict(list)
            stats = self.statistics[var_name]
            # Collect statistics for the variable
            varm = dr.select(dr.isnan(var), 0, var)
            varm = dr.select(dr.isnan(var), dr.mean(varm), varm)
            stats["min"].append(dr.min(var)[0])
            stats["max"].append(dr.max(var)[0])
            stats["mean"].append(dr.mean(var)[0])

            # Collect statistics for the gradient of the variable
            grad = dr.grad(var)
            grad = dr.select(dr.isnan(grad), dr.mean(dr.mean(grad)), grad)
            grad_norm = dr.norm(grad) if len(grad.shape) > 1 else dr.abs(grad)
            norm_np = grad_norm.numpy()
            grad_mean = np.mean(norm_np[norm_np < 5])
            stats["grad_mean"].append(grad_mean)
            stats["grad_variance"].append(np.mean(grad_norm ** 2) - grad_mean ** 2)
            stats["grad_min"].append(dr.min(grad)[0])
            stats["grad_max"].append(dr.max(grad)[0])

    def plot_statistics(self, output_dir: Path) -> None:
        """
        Plot the collected statistics and save them to the output directory.
        """
        if not self.track_grads:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        for var_name, stats in self.statistics.items():
            fig, ax_left  = plt.subplots(figsize=(10, 6))
            x = np.arange(len(stats["grad_mean"])) + 1
            m = np.array(stats["grad_mean"])
            m = np.nan_to_num(m, nan=np.nanmean(m), posinf=np.nanmax(m), neginf=np.nanmin(m))

            var = np.array(stats["grad_variance"]).clip(0, 2.0)

            ax_left.plot(x, m, label='Grad Mean', color='blue')
            # ax_left.fill_between(x, m - var, m + var, color='blue', alpha=0.3)
            ax_left.set_title(f'Statistics for {var_name}')
            ax_left.set_xlabel('Iteration')
            ax_left.set_ylabel('Gradient')

            ax_right = ax_left.twinx()
            ax_right.plot(x, stats["min"], label='Param Min', color='green', linestyle='--')
            ax_right.plot(x, stats["mean"], label='Param Mean', color='green', linestyle='-')
            ax_right.plot(x, stats["max"], label='Param Max', color='green', linestyle='--')
            ax_right.set_ylabel('Param Value')
            ax_right.legend()
            plt.title(f'Statistics for {var_name}')
            plt.tight_layout()
            plt.savefig(output_dir / f'{var_name}_statistics.png')
            plt.close(fig)


class LRScheduler:
    def __init__(self, optimizer: mi.ad.Optimizer, initial_lr: float, num_steps: int, base_lr_dict: Dict[str, float], warmup_ratio: float = 0., min_lr_ratio: float = 0.03):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.base_lr_dict = base_lr_dict
        self.warmup_steps = int(warmup_ratio * num_steps)
        self.num_steps = num_steps
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

        effect_dim(f"\nUsing LRScheduler with warmup steps: {self.warmup_steps}, min LR: {self.initial_lr * self.min_lr_ratio}, initial LR: {self.initial_lr}")

    def step(self):
        scale = self._get_lr_scale()
        self.apply_lr_scale(scale)
        self.current_step += 1

    def _get_lr_scale(self) -> float:
        if self.current_step < self.warmup_steps:
            return (self.current_step / self.warmup_steps)
        t = min(1.0, (self.current_step - self.warmup_steps) / max(1, int(0.9 * self.num_steps) - self.warmup_steps))
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * t))

    def apply_lr_scale(self, scale: float) -> None:
        for k, v in self.base_lr_dict.items():
            self.optimizer.set_learning_rate({k: v * scale})

        if self.current_step < self.warmup_steps:
            for k in self.base_lr_dict.keys():
                if 'quat' in k:
                    self.optimizer.set_learning_rate({k: 0.})

# endregion Utils