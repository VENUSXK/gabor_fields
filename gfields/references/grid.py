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
from mitsuba import ScalarTransform4f as T

from gfields.cameras import create_cameras
from gfields.training.configs import get_volume_config
from gfields.references import render_absorption_references


def prepare_cameras(args):
    """Orthographic turntable ring at distance 8 (matches the retired
    hierarchical helper; ``create_cameras`` is the single source of truth)."""
    return create_cameras(
        args.cam_count, args.cam_res_x, args.cam_res_y,
        cam_scale=args.cam_scale, distance=8.0)


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
    parser.add_argument('--output',                             type=str,   default='./test_voxelized_renders/', help='Path to the output image')
    parser.add_argument('--multiple_scattering',                action="store_true",  help='render multiple scattering of the asset')
    parser.add_argument('--envmap',                             type=str,   default='./data/envmaps/stierberg_sunrise_2k.exr', help='Path to the environment map')
    parser.add_argument('--envmap_scale',                       type=float, default=0.8, help='Scaling factor for the environment map')
    parser.add_argument('--max_depth',                          type=int,   default=-1, help='Maximum depth for the volume rendering')
    parser.add_argument('--rr_depth',                           type=int,   default=10000, help='Start using RR at this depth (-1 no rr)')
    parser.add_argument('--majorant_res_factor',                type=int, default=1, help='Majorant resolution factor')
    parser.add_argument('--majorant_factor',                    type=float, default=1.01, help='Majorant factor')
    parser.add_argument('--density_scale',                      type=float, default=1.0, help='Scaling factor for the density')
    parser.add_argument('--asset_name',                         type=str, default=1.0, help='Scaling factor for the volume size')
    parser.add_argument('--vol_size_scale',                     type=float, default=2.0, help='Scaling factor for the volume size')
    parser.add_argument('--samples_per_voxel',                  type=int, default=0, help='Number of samples per voxel to add to the file name.')
    parser.add_argument('--method',                             type=str, default='', help='Which method was used for sampling the MM to add to the file name.')
    return parser.parse_args()


def get_grid_transform(asset_name: str, vol_size_scale: float) -> mi.ScalarTransform4f:
    """Per-asset ``to_world`` transform, delegating to the canonical
    :class:`~gfields.training.configs.VolumeConfig` table (single source of
    truth for asset transforms; this used to duplicate it with a second,
    inconsistent set of per-asset scales/rotations)."""
    to_world = get_volume_config(asset_name).to_world
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
        verbose: bool = False
        ) -> np.ndarray:
    # Non-scattering case is pure
    # absorption/transmittance, so reuse the RRT + supervoxel-majorant
    # reference integrator (gfields.references) instead of full PT
    if not multiple_scattering and albedo == 0.0:
        with (gfields.benchmark.single_run('Rendering') if verbose else contextlib.nullcontext()):
            img = render_absorption_references(
                density_data * density_scale, to_world, [camera],
                spp=spp, background=1.0,
                supervoxel_factor=max(1, 8 // max(int(majorant_res_factor), 1)),
            )[0]
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
    scene_ref = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'prbvolpath',
        },
        'object': {
            'type': 'cube',
            'bsdf': { 'type': 'null' },
            'interior': medium_dict,
            'to_world': T().scale([1.5, 1.5, 1.5]),
        },
        'environment': {'type': 'constant'} if not multiple_scattering else {'type': 'envmap', 'filename': envmap, 'scale': envmap_scale},
    })

    if verbose:
        with gfields.benchmark.single_run('Rendering'):
            img = mi.render(scene_ref, sensor=camera, spp=spp)
    else:
        img = mi.render(scene_ref, sensor=camera, spp=spp)
    del scene_ref
    gc.collect()
    dr.flush_kernel_cache()
    return img

def render_volume(
        density_data: np.ndarray, 
        camera: dict, spp: int = 1,
        to_world: mi.ScalarTransform4f = T().scale([2.0, 2.0, 2.0]).translate(-1.0),
        majorant_res_factor: int = 1, majorant_factor: float = 1.01, 
        albedo: float = 0.5, density_scale: float = 1.0,
        multiple_scattering: bool = False,
        envmap: str = './data/envmaps/stierberg_sunrise_2k.exr',
        envmap_scale: float = 0.8,
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
    
    img = _render_volume(
        density_data=density_data,
        to_world=to_world,
        camera=cameras[args.cam_index],
        spp=args.spp,
        multiple_scattering=args.multiple_scattering,
        majorant_res_factor=args.majorant_res_factor,
        majorant_factor=args.majorant_factor,
        albedo=args.albedo,
        density_scale=args.density_scale,
        envmap=args.envmap,
        envmap_scale=args.envmap_scale,
        verbose=True
    )

    # Write image to file on disk
    print(f'Writing rendered image to {args.output}')
    import os
    output_path = args.output + f'{args.asset_name}_{args.method}_{args.samples_per_voxel}.exr'
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    mi.util.write_bitmap(output_path, img)
    print("Done!")