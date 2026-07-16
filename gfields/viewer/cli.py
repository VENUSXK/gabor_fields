# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Interactive viewer for volumetric-primitive assets (polyscope).

Progressive accumulation, continuous frequency-domain LOD slider, integrator
performance controls (tomography / prb, stochastic LOD criteria, depths,
spp), environment-map rotation, EXR/PNG frame export and full Mitsuba
XML+PLY scene export.

Usage:
    gfields viewer <asset_dir> [--envmap path.exr] [--integrator prb]
"""

from __future__ import annotations

import argparse

import mitsuba as mi

if mi.variant() is None:
    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

from gfields._paths import REPO_ROOT
from gfields.rendering.pipeline import MS_ALBEDO_DEFAULT

# Shipped defaults so `gfields viewer` just works for the release demo.
DEFAULT_ASSET = REPO_ROOT / 'data' / 'examples' / 'wdas8'
DEFAULT_ENVMAP = (REPO_ROOT / 'resources' / 'environment_maps'
                  / 'rosendal_park_sunset_puresky_2k.exr')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('asset', nargs='?', default=str(DEFAULT_ASSET),
                        help='Optimized asset directory (optimized_asset_pyrN '
                        'layout). Defaults to the bundled wdas8 (Disney '
                        'cloud) example asset.')
    parser.add_argument('--asset_pyr', type=int, default=None,
                        help='Pyramid variant to load (optimized_asset_pyr<N>)')
    parser.add_argument('--envmap', default=None,
                        help='Environment map (EXR/HDR). Omit for a constant '
                        'emitter (matches training).')
    parser.add_argument('--integrator',
                        choices=['tomography', 'prb', 'prb_stochastic'],
                        default='tomography',
                        help='Tomography matches training; prb enables '
                        'multiple scattering with an env map; prb_stochastic '
                        'uses decomposition-tracking distance sampling '
                        '(docs/prb_stochastic.md).')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Full render resolution when the camera is still')
    parser.add_argument('--preview_resolution', type=int, default=128,
                        help='Reduced resolution while the camera is moving')
    parser.add_argument('--ortho_scale', type=float, default=None,
                        help='Orthographic half-extent in world units '
                        '(default: from asset bounds / training cam_scale)')
    parser.add_argument('--max_spp', type=int, default=512,
                        help='Accumulation budget per view')
    parser.add_argument('--output', default='results/viewer',
                        help='Directory for exported frames/scenes')
    parser.add_argument('--opacity_scale', type=float, default=1.0)
    parser.add_argument('--albedo_scale', type=float, default=MS_ALBEDO_DEFAULT,
                        help='Uniform primitive albedo for path tracing (MS mode)')
    parser.add_argument('--kernel_extent', type=float, default=3.0)
    parser.add_argument('--world_scale', type=float, default=1.0)
    parser.add_argument('--ignore_asset_albedo', action='store_true')
    parser.add_argument('--gaussian_only', action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument('--freeze', action='store_true',
                        help="Enable the tomography integrator's internal "
                             '@dr.freeze (use_kernel_freeze): ~2.4x faster '
                             'frozen replay per frame, records cleanly on '
                             'Mitsuba 3.9 (unlike the old outer dr.freeze). '
                             'Sliders still propagate. Tomography only.')
    parser.add_argument('--no-freeze', '--no_freeze', dest='freeze',
                        action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(freeze=False)
    parser.add_argument('--use_anyhit', action='store_true',
                        help='Tomography: render the primal pass with a '
                             'single OptiX any-hit traversal instead of the '
                             'sequential walk (adjoint pass unaffected). '
                             'Also toggleable live in the Integrator panel.')
    return parser.parse_args()


if __name__ == '__main__':
    from gfields.viewer import run_viewer
    run_viewer(parse_args())
