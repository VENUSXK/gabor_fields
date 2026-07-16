# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Render gabor-field assets.

Auto-detects the asset structure and supports:
- tomography / multiple-scattering rendering
- Gaussian / Laplacian pyramid rendering
- benchmarking mode

A turntable / rotating-envmap video is available via the ``video`` subcommand
(``python -m gfields.render video <asset> ...``); camera helpers come from
``gfields.cameras`` and pipeline / integrator helpers from ``gfields.rendering``
"""
import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import gfields as vp

from gfields.training.adaptive import clamp_extents as compute_adaptive_extent_per_primitive
from gfields.rendering.pipeline import (
    AssetInfo,
    apply_asset_render_defaults,
    create_environment_config,
    create_integrator_config,
    detect_asset_info,
    envmap_to_world_transform,
    get_asset_name,
    load_pyramid_level,
    load_steerable_asset,
)
from gfields.io import is_steerable_asset
from gfields.cameras import create_cameras

ADAPTIVE_EXTENT_AVAILABLE = True

np.random.seed(0)

# =============================================================================
# Output Path Utilities
# =============================================================================

@dataclass
class OutputConfig:
    """Configuration for output file naming and organization."""
    base_dir: str
    asset_name: str
    render_mode: str      # 'full', 'gaussian_pyr', 'laplacian_pyr'
    scattering_mode: str  # 'tomography', 'multi'
    
    @property
    def experiment_dir(self) -> str:
        """Get the experiment output directory."""
        return os.path.join(self.base_dir, f"{self.asset_name}_{self.render_mode}_{self.scattering_mode}")
    
    def get_frame_path(self, 
                       frame_idx: int = 0,
                       cam_idx: Optional[int] = None,
                       level_idx: Optional[int] = None,
                       suffix: str = "") -> str:
        """
        Generate a descriptive filename for a rendered frame.
        
        Examples:
            cloud_1_full_tomography_cam11.exr
            cloud_1_gaussian_pyr_multi_level2_cam05.exr
            cloud_1_full_multi_video_frame0042.exr
        """
        parts = [self.asset_name]
        
        if level_idx is not None:
            parts.append(f"level{level_idx}")
        
        if cam_idx is not None:
            parts.append(f"cam{str(cam_idx).zfill(2)}")
        
        if suffix:
            parts.append(suffix)
        
        if frame_idx > 0 or (cam_idx is None and level_idx is None and not suffix):
            parts.append(f"frame{str(frame_idx).zfill(4)}")
        
        filename = "_".join(parts) + ".exr"
        return os.path.join(self.experiment_dir, filename)


# =============================================================================
# Rendering Functions
# =============================================================================

def render_single_frame(scene, camera, spp: int, seed: int = 0):
    """Render a single frame with benchmarking."""
    
    with vp.benchmark.single_run('Rendering'):
        return mi.render(scene, sensor=camera, spp=spp, seed=seed)


# =============================================================================
# Main Script
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Render Volumetric Primitives asset for paper figures'
    )
    
    # Asset path
    parser.add_argument('asset', type=str, nargs='?', default=None,
                        help='Path to the optimized asset directory')
    
    # Rendering mode
    parser.add_argument('--render_pyramid_gaussian', action='store_true',
                        help='Render Gaussian pyramid of the asset')
    parser.add_argument('--render_pyramid_laplacian', action='store_true',
                        help='Render Laplacian pyramid of the asset')
    parser.add_argument('--multiple_scattering', action='store_true',
                        help='Enable multiple scattering rendering')
    parser.add_argument('--stochastic_sampling', action='store_true',
                        help='Use the gfields_prb_stochastic integrator '
                             '(decomposition-tracking distance sampling; '
                             'multiple-scattering mode only)')
    parser.add_argument('--nee_transmittance', type=str, default='analytic',
                        choices=['stochastic', 'analytic'],
                        help='Shadow-ray transmittance estimator of '
                             '--stochastic_sampling (analytic: deterministic '
                             'optical-depth traversal, best at equal '
                             'variance; stochastic: per-primitive Bernoulli '
                             'tests)')
    parser.add_argument('--use_anyhit_sampling', action='store_true',
                        help='Opt into any-hit distance sampling for '
                             '--stochastic_sampling (validated but currently '
                             'slower than the hit walk; NEE any-hit is on by '
                             'default regardless -- prb_stochastic.md §5)')
    parser.add_argument('--no_anyhit', action='store_true',
                        help='Disable the --stochastic_sampling NEE any-hit '
                             'fast path (on by default) and fall back to the '
                             'sequential shadow-ray walk')
    parser.add_argument('--residual_weight_clamp', type=float, default=None,
                        help='Cap the --stochastic_sampling envelope '
                             'delta-tracking null-weight amplification '
                             '(plugin default: 1.0)')
    parser.add_argument('--use_anyhit', action='store_true',
                        help='Tomography mode: render the primal pass with a '
                             'single OptiX any-hit traversal instead of the '
                             'sequential walk (adjoint pass unaffected)')
    parser.add_argument('--use_kernel_freeze', action='store_true',
                        help='Tomography mode: enable the integrator\'s '
                             'internal @dr.freeze for faster frozen replay')
    parser.add_argument('--render_all_cams', action='store_true',
                        help='Render from all cameras')
    parser.add_argument('--max_level', type=int, default=-1,
                        help='Maximum pyramid level to render (-1 for all levels)')
    
    # Camera settings
    parser.add_argument('--cam_index', type=int, default=11,
                        help='Camera index for single-camera rendering')
    parser.add_argument('--cam_count', type=int, default=16,
                        help='Number of cameras to generate')
    parser.add_argument('--cam_res_x', type=int, default=1200,
                        help='Camera resolution width')
    parser.add_argument('--cam_res_y', type=int, default=800,
                        help='Camera resolution height')
    parser.add_argument('--cam_scale', type=float, default=2.35,
                        help='Orthographic only: uniform scale on sensor to_world (larger = wider framing, less zoom)')
    parser.add_argument('--cam_distance', type=float, default=28.0,
                        help='Camera distance from origin (asset / steerable modes; single-kernel uses a fixed closer distance)')
    parser.add_argument('--cam_fov', type=float, default=25.0,
                        help='Perspective only: horizontal field of view in degrees (default ~normal lens; was 12.5 which is very telephoto)')
    parser.add_argument('--use_perspective_cam', action='store_true',
                        help='Use perspective camera instead of default orthographic (see --cam_fov)')

    # Rendering parameters
    parser.add_argument('--spp', type=int, default=1,
                        help='Samples per pixel')
    parser.add_argument('--albedo', type=float, default=None,
                        help='Albedo scale factor (uniform grey, multiplied into per-channel albedo)')
    parser.add_argument('--albedo_rgb', type=float, nargs=3, default=None,
                        metavar=('R', 'G', 'B'),
                        help='Per-primitive RGB albedo applied to all primitives '
                             '(takes precedence over --use_asset_albedo). Final albedo '
                             'in the path tracer is this RGB multiplied by --albedo.')
    parser.add_argument('--use_asset_albedo', action='store_true',
                        help='Load per-primitive albedo from albedo_pyr*.npy when present '
                             '(ignored if --albedo_rgb is set).')
    parser.add_argument('--kernel', type=str, default='gabor_gaussian_mixture',
                        help='Kernel type: gaussian, epanechnikov, gabor_gaussian_mixture')
    parser.add_argument('--variant', type=str, default='cuda_ad_rgb',
                        help='Mitsuba variant')
    parser.add_argument('--kernel_extent', type=float, default=3.0,
                        help='Default extent of the kernel (default: 3.0). Per-level extents from asset take precedence.')
    parser.add_argument('--asset_scale', type=float, default=1.0,
                        help='Uniform scale for primitive centers and semi-axes (scales) only; extent and sigma_t '
                             'unchanged. to_world on ellipsoidsmesh is PLY-only in this Mitsuba build, so scaling is '
                             'applied in buffers.')
    
    # Adaptive extent parameters (for on-the-fly computation)
    parser.add_argument('--adaptive_extent', '-ae', action='store_true',
                        help='Compute adaptive extents based on kernel contribution threshold (overrides asset extents)')
    parser.add_argument('--extent_threshold', type=float, default=1e-4,
                        help='Minimum kernel contribution at boundary for adaptive extent (default: 1e-4)')
    parser.add_argument('--min_extent', type=float, default=0.1,
                        help='Minimum extent value for adaptive extent (default: 0.1)')
    parser.add_argument('--envelope_only', action='store_true',
                        help='Ignore frequency term, use only Gaussian envelope for extent (treats Gabors like Gaussians)')
    
    # Environment
    parser.add_argument('--envmap', type=str,
                        default='./data/envmaps/kloppenheim_02_puresky_2k.exr',
                        help='Path to environment map')
    parser.add_argument('--envmap_scale', type=float, default=1.0,
                        help='Environment map intensity scale')
    parser.add_argument('--envmap_rotation_x', type=float, default=0.0,
                        help='Rotation of environment map around X-axis in degrees (default: 0)')
    parser.add_argument('--envmap_rotation_y', type=float, default=135.0,
                        help='Y rotation in degrees (added to a 180° base). Tune to move the HDRI sun away from dominating directions; applies to static and rotating-lighting video.')
    parser.add_argument('--envmap_rotation_z', type=float, default=0.0,
                        help='Rotation of environment map around Z-axis in degrees (default: 0)')

    # Kernel parameters
    parser.add_argument('--opacity_scaler', type=float, default=1.0,
                        help='Opacity scaler for the kernel')
    
    # Path tracing parameters
    parser.add_argument('--max_depth', type=int, default=None,
                        help='Maximum path depth (-1 = unlimited tomography)')
    parser.add_argument('--stochastic_selection_criteria', type=str,
                        default='deterministic', help='Stochastic primitive selection')
    parser.add_argument('--stochastic_selection_criteria_nee', type=str,
                        default='deterministic', help='Stochastic selection for NEE')
    parser.add_argument('--power_law_beta', type=float, default=0.2,
                        help='Power law beta for selection')
    parser.add_argument('--power_law_beta_nee', type=float, default=0.75,
                        help='Power law beta for NEE selection')
    parser.add_argument('--rr_depth', type=int, default=None,
                        help='Russian roulette start depth (-1 to disable)')
    parser.add_argument('--rr_nee_depth', type=int, default=None,
                        help='Russian roulette depth for NEE')
    parser.add_argument('--max_overlaps', type=int, default=256,
                        help='Maximum primitive overlaps')
    parser.add_argument('--rr_depth_primitive', type=int, default=1024,
                        help='RR depth for primitive traversal')
    parser.add_argument('--max_depth_primitive', type=int, default=8192,
                        help='Maximum primitive traversal depth')
    parser.add_argument('--solver_type', type=str, default='bisection',
                        help='Solver type for volume rendering')
    parser.add_argument('--solver_max_iterations', type=int, default=None,
                        help='Maximum number of solver iterations')
    
    # Orientation selectivity (asset_encoding is auto-detected: 'simple' for pyramid, 'steerable' for steerable assets)
    parser.add_argument('--num_orientations', type=int, default=3,
                        help='Number of orientation bins (steerable mode only)')
    parser.add_argument('--orientation_selection', type=str, default='deterministic',
                        choices=['deterministic', 'threshold', 'importance', 'control_variate_threshold'],
                        help='Orientation selection mode for primary paths')
    parser.add_argument('--orientation_selection_nee', type=str, default='deterministic',
                        choices=['deterministic', 'threshold', 'importance', 'control_variate_threshold'],
                        help='Orientation selection mode for NEE')
    parser.add_argument('--orientation_threshold', type=float, default=0.8,
                        help='Alignment threshold for threshold-based orientation selection')
    parser.add_argument('--orientation_decay_rate', type=float, default=1.5,
                        help='Decay rate for importance-based orientation selection')
    parser.add_argument('--importance_mix_weight', type=float, default=0.5,
                        help='MIS mixing weight for importance sampling (0=uniform, 1=pure importance)')
    parser.add_argument('--bin_rotation', type=float, nargs=4, default=None,
                        metavar=('X', 'Y', 'Z', 'W'),
                        help='Quaternion [x,y,z,w] to rotate orientation bins (for rotated assets)')
    
    # Output settings
    parser.add_argument('--output', type=str, default='output',
                        help='Output path/prefix')
    parser.add_argument('--path_add', type=str, default='',
                        help='Additional path suffix')
    
    # Benchmarking
    parser.add_argument('--benchmark_run_count', type=int, default=0,
                        help='Number of benchmark runs (0 to disable)')
    parser.add_argument('--benchmark_dry_runs', type=int, default=1,
                        help='Number of dry benchmark runs (0 to disable)')

    # Testing
    parser.add_argument('--primitive_subset', type=int, default=None,
                        help='Number of primitives to render (first N primitives)')
    
    return parser.parse_args()


# ===========================================================================
# Video mode (folded from the former `render video` tool)
# ===========================================================================
# Env-map-lit sequence of a volumetric-primitive asset. Two motions:
#   turntable -- camera orbits the static asset under fixed env lighting.
#   envmap    -- camera fixed; the env map spins so the lighting sweeps.
# The scene is built once (sensor swapped / emitter to_world rewritten per
# frame), so kernels are reused across the sequence. Frames are written as EXR
# and optionally compiled to MP4 via ffmpeg.
# Run via `python -m gfields.render video <asset> [options]`.

def _video_parse_args() -> argparse.Namespace:
    from gfields.rendering.pipeline import MS_ALBEDO_DEFAULT
    from gfields._paths import REPO_ROOT as _REPO_ROOT
    default_envmap = (_REPO_ROOT / 'resources' / 'environment_maps'
                      / 'rosendal_park_sunset_puresky_2k.exr')
    p = argparse.ArgumentParser(
        prog='gfields.render video',
        description='Env-map-lit turntable / rotating-envmap video of an asset')
    p.add_argument('asset', help='Optimized asset directory (unified data/root.primitives_pyr*.ply layout)')
    p.add_argument('--motion', choices=['turntable', 'envmap'],
                   default='turntable', help='Camera orbit or rotating env map')
    p.add_argument('--envmap', default=str(default_envmap),
                   help='Environment map (EXR/HDR)')
    p.add_argument('--envmap_scale', type=float, default=0.8)
    p.add_argument('--output', default='results/renders/video',
                   help='Output directory for frames + MP4')
    p.add_argument('--spp', type=int, default=64)
    p.add_argument('--frames', type=int, default=120)
    p.add_argument('--framerate', type=float, default=30.0)
    p.add_argument('--resolution', type=int, default=720,
                   help='Square render resolution')
    p.add_argument('--fov', type=float, default=35.0)
    p.add_argument('--elevation', type=float, default=18.0,
                   help='Turntable camera elevation (degrees)')
    p.add_argument('--azimuth', type=float, default=30.0,
                   help='Fixed camera azimuth for --motion envmap (degrees)')
    p.add_argument('--distance', type=float, default=None,
                   help='Orbit radius; auto-framed from the asset AABB if unset')
    p.add_argument('--distance_scale', type=float, default=1.3,
                   help='Margin multiplier for the auto-framed orbit radius')
    p.add_argument('--max_depth', type=int, default=None)
    p.add_argument('--stochastic_sampling', action='store_true',
                   help='Use the gfields_prb_stochastic integrator '
                        '(decomposition-tracking distance sampling)')
    p.add_argument('--nee_transmittance', type=str, default='analytic',
                   choices=['stochastic', 'analytic'],
                   help='Shadow-ray transmittance estimator of '
                        '--stochastic_sampling')
    p.add_argument('--use_anyhit_sampling', action='store_true',
                   help='Opt into any-hit distance sampling for '
                        '--stochastic_sampling')
    p.add_argument('--no_anyhit', action='store_true',
                   help='Disable the --stochastic_sampling NEE any-hit fast '
                        'path (on by default)')
    p.add_argument('--residual_weight_clamp', type=float, default=None,
                   help='Cap the --stochastic_sampling envelope '
                        'delta-tracking null-weight amplification')
    p.add_argument('--opacity_scale', type=float, default=10.0)
    p.add_argument('--albedo_scale', type=float, default=MS_ALBEDO_DEFAULT)
    p.add_argument('--extent', type=float, default=5.0)
    p.add_argument('--ignore_asset_albedo', action='store_true')
    p.add_argument('--crf', type=int, default=18, help='x264 quality (lower=better)')
    p.add_argument('--no_video', action='store_true', help='Skip the MP4 compile')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def _video_build_scene_dict(args, asset_dir) -> dict:
    from gfields.rendering.pipeline import (
        detect_asset_info, load_pyramid_level, create_integrator_config,
        create_environment_config)
    info = detect_asset_info(str(asset_dir))
    scene_dict: dict = {
        'type': 'scene',
        'integrator': create_integrator_config(
            multiple_scattering=True, max_depth=args.max_depth,
            stochastic_sampling=args.stochastic_sampling,
            nee_transmittance=args.nee_transmittance,
            use_anyhit_sampling=(True if args.use_anyhit_sampling else None),
            use_anyhit=(False if args.no_anyhit else None),
            residual_weight_clamp=args.residual_weight_clamp),
        'environment': create_environment_config(
            multiple_scattering=True,
            envmap_path=args.envmap if os.path.exists(args.envmap) else None,
            envmap_scale=args.envmap_scale),
    }
    for level in range(info.num_levels):
        scene_dict[f'primitives_pyr{level}'] = load_pyramid_level(
            info, level, opacity_scale=args.opacity_scale,
            albedo_scale=args.albedo_scale, extent=args.extent,
            use_asset_albedo=not args.ignore_asset_albedo)
    return scene_dict


def _video_auto_distance(scene_dict, args):
    from gfields.cameras import compute_asset_aabb_from_scene_dict
    aabb = compute_asset_aabb_from_scene_dict(scene_dict)
    if aabb is None:
        return (args.distance or 24.0), [0.0, 0.0, 0.0]
    center = [float(c) for c in aabb['center']]
    if args.distance is not None:
        return float(args.distance), center
    diag = float(np.linalg.norm(aabb['size']))
    half_fov = np.deg2rad(0.5 * args.fov)
    dist = (0.5 * diag) / max(np.tan(half_fov), 1e-3) * args.distance_scale
    return dist, center


def run_video() -> None:
    from gfields.rendering.pipeline import (
        envmap_to_world_transform, get_asset_name, resolve_asset_path)
    from gfields.cameras import create_side_top_camera
    from gfields.rendering import video as video_io
    args = _video_parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = resolve_asset_path(args.asset)
    # Prefer the human-readable asset name (e.g. "smoke") over the resolved
    # "optimized_asset_pyrN" subfolder when naming the outputs.
    name = get_asset_name(asset_dir)
    if name.startswith('optimized_asset_pyr') or name == 'npy_data':
        name = get_asset_name(Path(asset_dir).parent)

    print(f'Rendering {args.motion} video: {asset_dir}')
    print(f'  envmap: {args.envmap}  | frames: {args.frames} @ {args.framerate}fps'
          f'  | {args.resolution}x{args.resolution} spp={args.spp}')

    scene_dict = _video_build_scene_dict(args, asset_dir)
    distance, center = _video_auto_distance(scene_dict, args)
    scene = mi.load_dict(scene_dict)
    res = args.resolution

    # For the envmap motion we drive the emitter transform through traverse.
    params = mi.traverse(scene)
    env_key = next((k for k in params.keys() if k.endswith('.to_world')
                    and ('env' in k.lower() or 'emitter' in k.lower())), None)
    fixed_sensor = create_side_top_camera(
        res, res, azimuth_deg=args.azimuth, elevation_deg=args.elevation,
        distance=distance, fov=args.fov, target=center)

    frame_paths: list = []
    for f in tqdm(range(args.frames), desc=f'{args.motion} frames'):
        u = f / max(args.frames, 1)
        if args.motion == 'turntable':
            sensor = create_side_top_camera(
                res, res, azimuth_deg=360.0 * u, elevation_deg=args.elevation,
                distance=distance, fov=args.fov, target=center)
            img = mi.render(scene, sensor=sensor, spp=args.spp, seed=args.seed)
        else:  # envmap motion
            if env_key is not None:
                params[env_key] = envmap_to_world_transform(360.0 * u, 0.0, 0.0, 0.0)
                params.update()
            img = mi.render(scene, params, sensor=fixed_sensor,
                            spp=args.spp, seed=args.seed)
        frame_path = str(out_dir / f'{name}_{args.motion}_{f:04d}.exr')
        mi.util.write_bitmap(frame_path, img)
        frame_paths.append(frame_path)

    if not args.no_video and frame_paths:
        out_mp4 = str(out_dir / f'{name}_{args.motion}.mp4')
        video_io.assemble(frame_paths, out_mp4, framerate=args.framerate,
                          crf=args.crf)
    print(f'Done: {len(frame_paths)} frames in {out_dir}')


def main():
    # Support `render {asset|tomo|ms|video} ...`; `asset` is the default mode so
    # the bare `render <asset> ...` form keeps working. `tomo` / `ms` inject the
    # preset flags from gfields.presets (user flags follow and override them).
    if len(sys.argv) > 1 and sys.argv[1] in ('asset', 'tomo', 'ms', 'video'):
        _mode = sys.argv.pop(1)
        if _mode == 'video':
            return run_video()
        if _mode in ('tomo', 'ms'):
            from gfields.presets import RENDER_MODES
            sys.argv[1:1] = RENDER_MODES[_mode]  # before user args -> user wins
    args = parse_args()
    apply_asset_render_defaults(args)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    if args.asset is None:
        raise ValueError("Asset path is required")

    # Accept an experiment folder (containing optimized_asset_pyr*) as well as an
    # already-resolved asset directory.
    from gfields.rendering.pipeline import resolve_asset_path
    args.asset = str(resolve_asset_path(args.asset))

    # Check if this is a steerable asset
    if is_steerable_asset(args.asset):
        print("=" * 60)
        print("Steerable Asset Detected")
        print("=" * 60)
        render_steerable_asset(args)
        return
    
    # Detect pyramid asset structure
    asset_info = detect_asset_info(args.asset)
    print(f"Detected asset: {asset_info}")
    
    # Compute per-primitive adaptive extents if requested (overrides asset extents)
    if args.adaptive_extent:
        if not ADAPTIVE_EXTENT_AVAILABLE:
            print("Warning: adaptive_extent module not available, using default extents")
        else:
            print("\nComputing per-primitive adaptive extents...")
            import gfields.io
            for level in range(asset_info.num_levels):
                # Load scales, omega, opacities for this level from the unified
                # per-level .ply store.
                a = gfields.io.load_ellipsoid_ply(
                    os.path.join(asset_info.ply_folder, f'root.primitives_pyr{level}.ply'))
                scales = np.asarray(a['scales'], dtype=np.float32)

                # Load omega if available
                omega = None
                if 'omega' in a:
                    omega = np.asarray(a['omega'], dtype=np.float32)

                # Load opacities (try different naming conventions)
                if 'opacities' in a:
                    opacities = np.asarray(a['opacities'], dtype=np.float32)
                elif 'sigma_t' in a:
                    opacities = np.asarray(a['sigma_t'], dtype=np.float32)
                elif 'sigmat' in a:
                    opacities = np.asarray(a['sigmat'], dtype=np.float32)
                else:
                    opacities = None

                # Compute per-primitive adaptive extents
                per_prim_extents = compute_adaptive_extent_per_primitive(
                    scales=scales,
                    omega=omega,
                    opacities=opacities,
                    base_extent=args.kernel_extent,
                    min_extent=args.min_extent,
                    threshold=args.extent_threshold,
                    envelope_only=args.envelope_only,
                )
                asset_info.extents[level] = per_prim_extents
                print(f"  Level {level}: n={len(per_prim_extents)}, "
                      f"extent range=[{per_prim_extents.min():.3f}, {per_prim_extents.max():.3f}], "
                      f"mean={per_prim_extents.mean():.3f}")
            print()
    
    # Determine rendering mode
    if args.render_pyramid_gaussian:
        render_mode = 'gaussian_pyr'
    elif args.render_pyramid_laplacian:
        render_mode = 'laplacian_pyr'
    else:
        render_mode = 'full'
    
    scattering_mode = 'multi' if args.multiple_scattering else 'tomography'
    
    # Setup output configuration
    output_config = OutputConfig(
        base_dir=args.output,
        asset_name=get_asset_name(args.asset),
        render_mode=render_mode,
        scattering_mode=scattering_mode,
    )
    
    # Create experiment output directory
    os.makedirs(output_config.experiment_dir, exist_ok=True)
    print(f"Output directory: {output_config.experiment_dir}")
    
    # Create cameras
    cameras = create_cameras(
        cam_count=args.cam_count,
        cam_res_x=args.cam_res_x,
        cam_res_y=args.cam_res_y,
        cam_scale=args.cam_scale,
        use_perspective=args.use_perspective_cam,
        fov=args.cam_fov,
        distance=args.cam_distance,
    )
    
    # Render based on mode
    if args.render_pyramid_gaussian or args.render_pyramid_laplacian:
        render_pyramid_mode(args, asset_info, cameras, output_config)
    else:
        render_full_asset(args, asset_info, cameras, output_config)
    
    print(f"Rendering complete. Output saved to: {output_config.experiment_dir}")


def render_pyramid_mode(args, asset_info: AssetInfo, cameras: List, output_config: OutputConfig):
    """Render pyramid levels (Gaussian or Laplacian)."""
    
    # Determine effective number of levels
    max_level = asset_info.num_levels if args.max_level < 0 else min(args.max_level + 1, asset_info.num_levels)
    
    for level_idx in range(max_level):
        print(f"Rendering pyramid level {level_idx}...")
        
        # Build scene dict
        scene_dict = {
            'type': 'scene',
            'integrator': create_integrator_config(
                multiple_scattering=args.multiple_scattering,
                kernel_type=args.kernel,
                max_depth=args.max_depth,
                pyramid_level_count=max_level,
                stochastic_selection_criteria=args.stochastic_selection_criteria,
                stochastic_selection_criteria_nee=args.stochastic_selection_criteria_nee,
                power_law_beta=args.power_law_beta,
                power_law_beta_nee=args.power_law_beta_nee,
                rr_depth=args.rr_depth,
                rr_nee_depth=args.rr_nee_depth,
                max_overlaps=args.max_overlaps,
                rr_depth_primitive=args.rr_depth_primitive,
                max_depth_primitive=args.max_depth_primitive,
                solver_type=args.solver_type,
                solver_max_iterations=args.solver_max_iterations,
                asset_encoding='simple',  # Pyramid assets always use simple encoding
                num_orientations=args.num_orientations,
                orientation_selection=args.orientation_selection,
                orientation_selection_nee=args.orientation_selection_nee,
                orientation_threshold=args.orientation_threshold,
                orientation_decay_rate=args.orientation_decay_rate,
                importance_mix_weight=args.importance_mix_weight,
                bin_rotation=args.bin_rotation,
                stochastic_sampling=args.stochastic_sampling,
                nee_transmittance=args.nee_transmittance,
                use_anyhit_sampling=(True if args.use_anyhit_sampling
                                     else None),
                use_anyhit=(False if args.no_anyhit else
                            (True if args.use_anyhit else None)),
                residual_weight_clamp=args.residual_weight_clamp,
                use_kernel_freeze=args.use_kernel_freeze,
                isotropic=False,  # asset renders use the HG phase (g=0.857)
            ),
            'environment': create_environment_config(
                multiple_scattering=args.multiple_scattering,
                envmap_path=args.envmap,
                envmap_scale=args.envmap_scale,
                envmap_rotation_x=args.envmap_rotation_x,
                envmap_rotation_y=args.envmap_rotation_y,
                envmap_rotation_z=args.envmap_rotation_z,
            ),
        }

        # For Gaussian pyramid: accumulate levels 0..level_idx
        # For Laplacian pyramid: only render level_idx
        if args.render_pyramid_laplacian:
            levels_to_render = [level_idx]
        else:
            levels_to_render = range(level_idx + 1)
        
        for j in levels_to_render:
            # Use per-level extent from asset, or fall back to command-line argument
            level_extent = asset_info.extents.get(j, args.kernel_extent)
            primitives = load_pyramid_level(
                asset_info, j, albedo_scale=args.albedo, opacity_scale=args.opacity_scaler,
                primitive_subset=args.primitive_subset, extent=level_extent, world_scale=args.asset_scale,
                albedo_rgb=args.albedo_rgb, use_asset_albedo=args.use_asset_albedo,
            )
            primitives['visibility_mask'] = 2 ** j
            scene_dict[f'primitives_pyr{j}'] = primitives

        # Render and save
        if args.render_all_cams:
            scene = mi.load_dict(scene_dict)
            for cam_idx, cam in enumerate(cameras):
                img = render_single_frame(scene, cam, args.spp)
                output_path = output_config.get_frame_path(level_idx=level_idx, cam_idx=cam_idx)
                mi.util.write_bitmap(output_path, img)
                print(f"  Saved: {os.path.basename(output_path)}")
        else:
            scene = mi.load_dict(scene_dict)
            img = render_single_frame(scene, cameras[args.cam_index], args.spp)
            output_path = output_config.get_frame_path(level_idx=level_idx, cam_idx=args.cam_index)
            mi.util.write_bitmap(output_path, img)
            print(f"  Saved: {os.path.basename(output_path)}")
        
        # Cleanup
        gc.collect()
        dr.flush_kernel_cache()


def render_full_asset(args, asset_info: AssetInfo, cameras: List, output_config: OutputConfig):
    """Render full asset with all pyramid levels."""
    
    # Determine effective number of levels
    max_level = asset_info.num_levels if args.max_level < 0 else min(args.max_level + 1, asset_info.num_levels)
    
    # Build scene dict with all levels
    scene_dict = {
        'type': 'scene',
        'integrator': create_integrator_config(
            multiple_scattering=args.multiple_scattering,
            kernel_type=args.kernel,
            max_depth=args.max_depth,
            pyramid_level_count=max_level,
            stochastic_selection_criteria=args.stochastic_selection_criteria,
            stochastic_selection_criteria_nee=args.stochastic_selection_criteria_nee,
            power_law_beta=args.power_law_beta,
            power_law_beta_nee=args.power_law_beta_nee,
            rr_depth=args.rr_depth,
            rr_nee_depth=args.rr_nee_depth,
            max_overlaps=args.max_overlaps,
            rr_depth_primitive=args.rr_depth_primitive,
            max_depth_primitive=args.max_depth_primitive,
            solver_type=args.solver_type,
            solver_max_iterations=args.solver_max_iterations,
            asset_encoding='simple',
            num_orientations=args.num_orientations,
            orientation_selection=args.orientation_selection,
            orientation_selection_nee=args.orientation_selection_nee,
            orientation_threshold=args.orientation_threshold,
            orientation_decay_rate=args.orientation_decay_rate,
            importance_mix_weight=args.importance_mix_weight,
            bin_rotation=args.bin_rotation,
            stochastic_sampling=args.stochastic_sampling,
            nee_transmittance=args.nee_transmittance,
            use_anyhit_sampling=(True if args.use_anyhit_sampling else None),
            use_anyhit=(False if args.no_anyhit else
                        (True if args.use_anyhit else None)),
            residual_weight_clamp=args.residual_weight_clamp,
            use_kernel_freeze=args.use_kernel_freeze,
        ),
        'environment': create_environment_config(
            multiple_scattering=args.multiple_scattering,
            envmap_path=args.envmap,
            envmap_scale=args.envmap_scale,
            envmap_rotation_x=args.envmap_rotation_x,
            envmap_rotation_y=args.envmap_rotation_y,
            envmap_rotation_z=args.envmap_rotation_z,
        ),
    }
    
    # Convert primitive_subset from int to slice if provided
    primitive_subset = None
    if args.primitive_subset is not None:
        primitive_subset = slice(0, args.primitive_subset)
    
    # Add pyramid levels up to max_level
    for level in range(max_level):
        # Use per-level extent from asset, or fall back to command-line argument
        level_extent = asset_info.extents.get(level, args.kernel_extent)
        primitives = load_pyramid_level(
            asset_info, level, albedo_scale=args.albedo, opacity_scale=args.opacity_scaler,
            primitive_subset=primitive_subset, extent=level_extent, world_scale=args.asset_scale,
            albedo_rgb=args.albedo_rgb, use_asset_albedo=args.use_asset_albedo,
        )
        # visibility_mask = 2**level: the LODSelector's per-ray mask (driven by
        # pyramid_level_count + the selection criterion) picks which levels a ray
        # traverses. This works once the integrator receives pyramid_level_count
        # (see create_integrator_config -- the tomography config previously dropped
        # it, collapsing the ray mask to bit 0 and hiding every Gabor level).
        vis_bit = 2 ** level
        primitives['visibility_mask'] = vis_bit
        scene_dict[f'primitives_pyr{level}'] = primitives
        # Get actual primitive count from the loaded centers tensor
        actual_count = primitives['centers'].shape[0]
        if isinstance(level_extent, np.ndarray):
            extent_str = f"per-prim (mean={level_extent.mean():.2f}, range=[{level_extent.min():.2f}, {level_extent.max():.2f}])"
        else:
            extent_str = f"{level_extent:.3f}"
        print(f"Level {level}: {actual_count} primitives (visibility_mask={vis_bit}, extent={extent_str})")

    # Render and save based on mode
    if args.benchmark_run_count > 0:
        render_with_benchmark(args, scene_dict, cameras, output_config)
    elif args.render_all_cams:
        scene = mi.load_dict(scene_dict)
        for cam_idx, cam in enumerate(cameras):
            img = render_single_frame(scene, cam, args.spp)
            output_path = output_config.get_frame_path(cam_idx=cam_idx)
            mi.util.write_bitmap(output_path, img)
            print(f"Saved: {os.path.basename(output_path)}")
    else:
        scene = mi.load_dict(scene_dict)
        img = render_single_frame(scene, cameras[args.cam_index], args.spp)
        output_path = output_config.get_frame_path(cam_idx=args.cam_index)
        mi.util.write_bitmap(output_path, img)
        print(f"Saved: {os.path.basename(output_path)}")

    gc.collect()
    dr.flush_kernel_cache()


def render_steerable_asset(args):
    """Render a steerable pyramid asset."""
    
    # Load steerable asset
    steerable_info = load_steerable_asset(
        args.asset,
        opacity_scale=args.opacity_scaler,
        albedo_scale=args.albedo if args.multiple_scattering else 0.0,
        default_extent=args.kernel_extent,
        world_scale=args.asset_scale,
        albedo_rgb=args.albedo_rgb,
    )
    
    # Determine rendering mode
    scattering_mode = 'multi' if args.multiple_scattering else 'tomography'
    
    # Setup output configuration
    output_config = OutputConfig(
        base_dir=args.output,
        asset_name=get_asset_name(args.asset),
        render_mode='steerable',
        scattering_mode=scattering_mode,
    )
    os.makedirs(output_config.experiment_dir, exist_ok=True)
    print(f"Output directory: {output_config.experiment_dir}")
    
    # Create cameras
    cameras = create_cameras(
        cam_count=args.cam_count,
        cam_res_x=args.cam_res_x,
        cam_res_y=args.cam_res_y,
        cam_scale=args.cam_scale,
        use_perspective=args.use_perspective_cam,
        fov=args.cam_fov,
        distance=args.cam_distance,
    )
    
    # Build scene dict
    if args.multiple_scattering:
        integrator_dict = {
            'type': ('gfields_prb_stochastic' if args.stochastic_sampling
                     else 'gfields_prb'),
            'max_depth': args.max_depth,
            'kernel_normalized': True,
            'kernel_type': 'gabor_gaussian_mixture',
            'pyramid_level_count': steerable_info.num_frequency_levels,
            'asset_encoding': 'steerable',
            'num_orientations': steerable_info.num_orientations,
            'stochastic_selection_criteria': args.stochastic_selection_criteria,
            'stochastic_selection_criteria_nee': args.stochastic_selection_criteria_nee,
            'power_law_beta': args.power_law_beta,
            'power_law_beta_nee': args.power_law_beta_nee,
            'orientation_selection': args.orientation_selection,
            'orientation_selection_nee': args.orientation_selection_nee,
            'orientation_threshold': args.orientation_threshold,
            'orientation_decay_rate': args.orientation_decay_rate,
            'importance_mix_weight': args.importance_mix_weight,
            'solver_type': args.solver_type,
            'solver_max_iterations': args.solver_max_iterations,
            'max_overlaps': args.max_overlaps,
            'rr_depth_primitive': args.rr_depth_primitive,
            'max_depth_primitive': args.max_depth_primitive,
            'use_nee': True,
        }
        if args.stochastic_sampling:
            integrator_dict['nee_transmittance'] = args.nee_transmittance
            if args.use_anyhit_sampling:
                integrator_dict['use_anyhit_sampling'] = True
            if args.no_anyhit:
                integrator_dict['use_anyhit'] = False
            elif args.use_anyhit:
                integrator_dict['use_anyhit'] = True
            if args.residual_weight_clamp is not None:
                integrator_dict['residual_weight_clamp'] = args.residual_weight_clamp
        if args.rr_depth > -1:
            integrator_dict['rr_depth'] = args.rr_depth
        if args.rr_nee_depth > -1:
            integrator_dict['rr_nee_depth'] = args.rr_nee_depth
        if args.bin_rotation is not None:
            integrator_dict['bin_rotation'] = mi.ScalarQuaternion4f(*args.bin_rotation)
        
        # Environment
        if args.envmap and os.path.exists(args.envmap):
            environment = {
                'type': 'envmap',
                'filename': args.envmap,
                'scale': args.envmap_scale,
                'to_world': envmap_to_world_transform(
                    0.0,
                    args.envmap_rotation_x,
                    args.envmap_rotation_y,
                    args.envmap_rotation_z,
                ),
            }
        else:
            environment = {'type': 'constant'}
    else:
        integrator_dict = {
            'type': 'gfields_tomography',
            'asset_encoding': 'steerable',
            'max_depth': args.max_depth,
            'kernel_normalized': True,
            'kernel_full_range': True,
            'kernel_type': 'gabor_gaussian_mixture',
            'render_marginalization': True,
            'pyramid_level_count': steerable_info.num_frequency_levels,
            'num_orientations': steerable_info.num_orientations,
            'stochastic_selection_criteria': args.stochastic_selection_criteria,
            'power_law_beta': args.power_law_beta,
            'orientation_selection': args.orientation_selection,
            'orientation_threshold': args.orientation_threshold,
            'orientation_decay_rate': args.orientation_decay_rate,
            'importance_mix_weight': args.importance_mix_weight,
        }
        if args.bin_rotation is not None:
            integrator_dict['bin_rotation'] = mi.ScalarQuaternion4f(*args.bin_rotation)
        if args.use_anyhit:
            integrator_dict['use_anyhit'] = True
        if args.use_kernel_freeze:
            integrator_dict['use_kernel_freeze'] = True

        environment = {'type': 'constant'}
    
    scene_dict = {
        'type': 'scene',
        'integrator': integrator_dict,
        'environment': environment,
    }
    
    # Add all primitives
    for subband_key, prim_dict in steerable_info.primitives_dict.items():
        scene_dict[f'primitives_{subband_key}'] = prim_dict

    # Load and render
    print(f"\nLoading scene...")
    scene = mi.load_dict(scene_dict)

    # Render based on mode
    if args.render_all_cams:
        for cam_idx, cam in enumerate(cameras):
            print(f"Rendering camera {cam_idx}...")
            img = render_single_frame(scene, cam, args.spp)
            output_path = output_config.get_frame_path(cam_idx=cam_idx)
            mi.util.write_bitmap(output_path, img)
            print(f"  Saved: {os.path.basename(output_path)}")
    else:
        print(f"Rendering camera {args.cam_index}...")
        img = render_single_frame(scene, cameras[args.cam_index], args.spp)
        output_path = output_config.get_frame_path(cam_idx=args.cam_index)
        mi.util.write_bitmap(output_path, img)
        print(f"Saved: {os.path.basename(output_path)}")
    
    print(f"\nRendering complete. Output saved to: {output_config.experiment_dir}")
    
    gc.collect()
    dr.flush_kernel_cache()


def render_with_benchmark(args, scene_dict: Dict, cameras: List, output_config: OutputConfig):
    """Render with benchmarking enabled."""
    
    df = []
    
    @vp.benchmark.wrap_function(
        label='Render',
        nb_runs=args.benchmark_run_count,
        nb_dry_runs=args.benchmark_dry_runs,
        dataframes=df,
        log_level=2
    )
    def render_asset(spp, sensor, scene, label=None):
        return mi.render(scene, sensor=sensor, spp=spp, seed=0)
    
    scene = mi.load_dict(scene_dict)
    
    if args.render_all_cams:
        for cam_idx, cam in enumerate(cameras):
            img = render_asset(spp=args.spp, sensor=cam, scene=scene, label=f'cam_{cam_idx}')
            output_path = output_config.get_frame_path(cam_idx=cam_idx, suffix="benchmark")
            mi.util.write_bitmap(output_path, img)
            print(f"Saved: {os.path.basename(output_path)}")
    else:
        img = render_asset(
            spp=args.spp, sensor=cameras[args.cam_index], scene=scene, label='single_cam'
        )
        output_path = output_config.get_frame_path(cam_idx=args.cam_index, suffix="benchmark")
        mi.util.write_bitmap(output_path, img)
        print(f"Saved: {os.path.basename(output_path)}")


if __name__ == '__main__':
    main()
