# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mitsuba as mi
import drjit as dr
import numpy as np

# Set the Mitsuba variant (adjust to 'cuda_rgb' or 'llvm_rgb' for faster rendering)
mi.set_variant('cuda_ad_rgb')
import argparse
import contextlib
import gfields
import gc
from pathlib import Path
from mitsuba import ScalarTransform4f as T

from gfields.cameras import create_cameras
from gfields.training.configs import GenericConfig, get_volume_config
from gfields.references import render_absorption_references


def prepare_cameras(args):
    """Create the same configurable turntable cameras used by render.py."""
    return create_cameras(
        args.cam_count, args.cam_res_x, args.cam_res_y,
        cam_scale=args.cam_scale,
        use_perspective=args.use_perspective_cam,
        fov=args.cam_fov,
        distance=args.cam_distance,
        elevation=args.cam_elevation,
        azimuth_span=args.cam_azimuth_span,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Render a Python asset')
    parser.add_argument('asset',                                type=str,                  help='Path to the Python asset')
    parser.add_argument('--albedo',                             type=float, default=0.0, help='Scaling factor for the sigmat value')
    parser.add_argument('--cam_index',                          type=int,   default=9,     help='Index of the cameras to use for rendering')
    parser.add_argument('--cam_scale',                          type=float, default=0.6,   help='Scaling factor for the camera resolution')
    parser.add_argument('--marginalization_cam',                action="store_true",  help='Turn rendering into marginalization camera rendering')
    parser.add_argument('--spp',                                type=int,   default=32,     help='Number of samples per pixel used for rendering')
    parser.add_argument('--cam_count',                          type=int,   default=16,     help='Number of cameras to use for rendering')
    parser.add_argument('--cam_res_x',                          type=int,   default=800,   help='Width of the camera resolution')
    parser.add_argument('--cam_res_y',                          type=int,   default=800,   help='Height of the camera resolution')
    parser.add_argument('--cam_distance',                       type=float, default=8.0,   help='Camera distance from the origin')
    parser.add_argument('--cam_fov',                            type=float, default=40.0,  help='Perspective camera field of view in degrees')
    parser.add_argument('--cam_elevation',                      type=float, default=None,  help='Camera elevation in degrees; positive values look down from +Y')
    parser.add_argument('--cam_azimuth_span',                   type=float, default=180.0, help='Total azimuth span covered by the cameras in degrees')
    parser.add_argument('--use_perspective_cam',                action='store_true',      help='Use perspective cameras instead of orthographic cameras')
    parser.add_argument('--render_all_cams',                    action='store_true',      help='Render every generated camera')
    parser.add_argument('--output',                             type=str,   default='./test_voxelized_renders/', help='Path to the output image')
    parser.add_argument('--multiple_scattering',                action="store_true",  help='render multiple scattering of the asset')
    parser.add_argument('--envmap',                             type=str,   default='./data/envmaps/stierberg_sunrise_2k.exr', help='Path to the environment map')
    parser.add_argument('--envmap_scale',                       type=float, default=0.8, help='Scaling factor for the environment map')
    parser.add_argument('--max_depth',                          type=int,   default=-1, help='Maximum depth for the volume rendering')
    parser.add_argument('--rr_depth',                           type=int,   default=10000, help='Start using RR at this depth (-1 no rr)')
    parser.add_argument('--majorant_res_factor',                type=int, default=1, help='Majorant resolution factor')
    parser.add_argument('--majorant_factor',                    type=float, default=1.01, help='Majorant factor')
    parser.add_argument('--density_scale',                      type=float, default=1.0, help='Scaling factor for the density')
    parser.add_argument('--asset_name',                         type=str, default='volume', help='Asset label used for named presets and output filenames; unknown names use the generic cube transform')
    parser.add_argument('--vol_size_scale',                     type=float, default=2.0, help='Scaling factor for the volume size')
    parser.add_argument('--samples_per_voxel',                  type=int, default=0, help='Number of samples per voxel to add to the file name.')
    parser.add_argument('--method',                             type=str, default='', help='Which method was used for sampling the MM to add to the file name.')
    return parser.parse_args()


def get_grid_transform(asset_name: str, vol_size_scale: float) -> mi.ScalarTransform4f:
    """Per-asset ``to_world`` transform, delegating to the canonical
    :class:`~gfields.training.configs.VolumeConfig` table (single source of
    truth for asset transforms; this used to duplicate it with a second,
    inconsistent set of per-asset scales/rotations)."""
    try:
        config = get_volume_config(asset_name)
    except ValueError:
        # Raw user-provided grids normally have no paper-asset preset.  In
        # that case ``asset_name`` remains a harmless output label and the
        # volume uses the same isotropic, origin-centred cube as generic
        # training runs.
        print(f"No named VolumeConfig for {asset_name!r}; using GenericConfig.")
        config = GenericConfig()
    to_world = config.to_world
    if vol_size_scale != 2.0:
        to_world = T().scale([vol_size_scale / 2.0] * 3) @ to_world
    return to_world

def _render_volume(
        density_data: np.ndarray, to_world: mi.ScalarTransform4f,
        camera: dict, spp: int,
        majorant_res_factor: int, majorant_factor: float,
        albedo: float, density_scale: float,
        multiple_scattering: bool,
        envmap: str, envmap_scale: float,
        max_depth: int = -1, rr_depth: int = 10000,
        verbose: bool = False
        ) -> np.ndarray:
    return _render_volumes(
        density_data=density_data,
        to_world=to_world,
        cameras=[camera],
        spp=spp,
        majorant_res_factor=majorant_res_factor,
        majorant_factor=majorant_factor,
        albedo=albedo,
        density_scale=density_scale,
        multiple_scattering=multiple_scattering,
        envmap=envmap,
        envmap_scale=envmap_scale,
        max_depth=max_depth,
        rr_depth=rr_depth,
        verbose=verbose,
    )[0]


def _render_volumes(
        density_data: np.ndarray, to_world: mi.ScalarTransform4f,
        cameras: list, spp: int,
        majorant_res_factor: int, majorant_factor: float,
        albedo: float, density_scale: float,
        multiple_scattering: bool,
        envmap: str, envmap_scale: float,
        max_depth: int = -1, rr_depth: int = 10000,
        verbose: bool = False
        ) -> list[np.ndarray]:
    """Render one or more cameras while uploading the dense grid only once."""
    # Non-scattering case is pure
    # absorption/transmittance, so reuse the RRT + supervoxel-majorant
    # reference integrator (gfields.references) instead of full PT
    if not multiple_scattering and albedo == 0.0:
        with (gfields.benchmark.single_run('Rendering') if verbose else contextlib.nullcontext()):
            img = render_absorption_references(
                density_data * density_scale, to_world, cameras,
                spp=spp, background=1.0,
                supervoxel_factor=max(1, 8 // max(int(majorant_res_factor), 1)),
            )
        gc.collect()
        dr.flush_kernel_cache()
        return img

    medium_dict = {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'data': mi.TensorXf(density_data),
                'to_world': to_world,
            },
            'albedo': albedo,
            'scale': density_scale,
        }
    integrator_dict = {
        'type': 'prbvolpath',
        'max_depth': max_depth,
        'rr_depth': rr_depth,
    }
    scene_ref = mi.load_dict({
        'type': 'scene',
        'integrator': integrator_dict,
        'object': {
            'type': 'cube',
            'bsdf': { 'type': 'null' },
            'interior': medium_dict,
            'to_world': T().scale([1.5, 1.5, 1.5]),
        },
        'environment': {'type': 'constant'} if not multiple_scattering else {'type': 'envmap', 'filename': envmap, 'scale': envmap_scale},
    })

    images = []
    render_context = (gfields.benchmark.single_run('Rendering')
                      if verbose else contextlib.nullcontext())
    with render_context:
        for camera in cameras:
            images.append(mi.render(scene_ref, sensor=camera, spp=spp))
    del scene_ref
    gc.collect()
    dr.flush_kernel_cache()
    return images

def render_volume(
        density_data: np.ndarray, 
        camera: dict, spp: int = 1,
        to_world: mi.ScalarTransform4f = T().scale([2.0, 2.0, 2.0]).translate(-1.0),
        majorant_res_factor: int = 1, majorant_factor: float = 1.01, 
        albedo: float = 0.5, density_scale: float = 1.0,
        multiple_scattering: bool = False,
        envmap: str = './data/envmaps/stierberg_sunrise_2k.exr',
        envmap_scale: float = 0.8,
        max_depth: int = -1,
        rr_depth: int = 10000,
        verbose: bool = False
        ) -> np.ndarray:
    # Render reference images ----------------------------------------
    img = _render_volume(
        density_data=density_data,
        to_world=to_world,
        camera=camera,
        spp=spp,
        multiple_scattering=multiple_scattering,
        majorant_res_factor=majorant_res_factor,
        majorant_factor=majorant_factor,
        albedo=albedo,
        density_scale=density_scale,
        envmap=envmap, 
        envmap_scale=envmap_scale,
        max_depth=max_depth,
        rr_depth=rr_depth,
        verbose=verbose
    )
    return img


if __name__ == '__main__':
    args = parse_args()
    # Create the Mitsuba volume grid
    import drjit as dr
    density_data = np.load(args.asset)

    # Scene definition
    np.random.seed(0)
    cameras = prepare_cameras(args)

    to_world = get_grid_transform(args.asset_name, args.vol_size_scale)
    
    if args.render_all_cams:
        selected_cameras = cameras
        selected_indices = list(range(len(cameras)))
    else:
        if not 0 <= args.cam_index < len(cameras):
            raise ValueError(
                f'--cam_index must be in [0, {len(cameras) - 1}], got {args.cam_index}')
        selected_cameras = [cameras[args.cam_index]]
        selected_indices = [args.cam_index]

    images = _render_volumes(
        density_data=density_data,
        to_world=to_world,
        cameras=selected_cameras,
        spp=args.spp,
        multiple_scattering=args.multiple_scattering,
        majorant_res_factor=args.majorant_res_factor,
        majorant_factor=args.majorant_factor,
        albedo=args.albedo,
        density_scale=args.density_scale,
        envmap=args.envmap,
        envmap_scale=args.envmap_scale,
        max_depth=args.max_depth,
        rr_depth=args.rr_depth,
        verbose=True
    )

    # Write image(s) to disk. Keep the legacy single-camera filename, and add
    # an explicit camera suffix when rendering the complete camera set.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{args.asset_name}_{args.method}_{args.samples_per_voxel}'
    for image, camera_index in zip(images, selected_indices):
        suffix = f'_cam{camera_index:02d}' if args.render_all_cams else ''
        output_path = output_dir / f'{stem}{suffix}.exr'
        print(f'Writing rendered image to {output_path}')
        mi.util.write_bitmap(str(output_path), image)
    print("Done!")
