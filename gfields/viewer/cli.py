# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Interactive viewer for volumetric-primitive assets (polyscope).

Progressive accumulation, continuous frequency-domain LOD slider, integrator
performance controls (tomography / prb / prb stochastic, stochastic LOD
criteria, any-hit fast paths, depths, spp), environment-map rotation, EXR/PNG
frame export and full Mitsuba XML+PLY scene export.

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
DEFAULT_ENVMAP = REPO_ROOT / 'data' / 'envmaps' / 'qwantani_dusk_2_1k.exr'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('asset', nargs='?', default=str(DEFAULT_ASSET),
                        help='Optimized asset directory (optimized_asset_pyrN '
                        'layout). Defaults to the bundled wdas8 (Disney '
                        'cloud) example asset.')
    parser.add_argument('--asset_pyr', type=int, default=None,
                        help='Pyramid variant to load (optimized_asset_pyr<N>)')
    parser.add_argument('--envmap', default=None,
                        help='Environment map (EXR/HDR). Default is mode-'
                        'dependent: tomography uses the constant emitter '
                        '(radiance 1.0, matches training) and the multiple-'
                        'scattering modes (prb / prb_stochastic) use the '
                        'bundled env map -- also when switching modes live. '
                        'Pass a path to force that env map in every mode, or '
                        'an empty string to force the constant emitter.')
    parser.add_argument('--integrator',
                        choices=['tomography', 'prb', 'prb_stochastic'],
                        default='tomography',
                        help='Tomography matches training; prb enables '
                        'multiple scattering with an env map; prb_stochastic '
                        'uses decomposition-tracking distance sampling '
                        '(docs/prb_stochastic.md). prb_stochastic is '
                        'Gaussian-only: on assets containing Gabor kernels '
                        'it falls back to prb with a warning.')
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
    parser.add_argument('--use_anyhit', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Any-hit fast paths, ON by default for every '
                             'integrator that supports one: tomography renders '
                             'the primal pass with a single OptiX any-hit '
                             'traversal instead of the sequential walk '
                             '(adjoint pass unaffected; falls back to the hit '
                             'walk on unsupported configs), and prb_stochastic '
                             'resolves NEE shadow rays with its any-hit query. '
                             'Each is also toggleable live in the Integrator '
                             'panel.')
    parser.add_argument('--no_anyhit', dest='use_anyhit', action='store_false',
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    # --envmap unset -> mode-dependent default: the bundled env map is carried
    # along but only *activates* in the multiple-scattering modes (the viewer
    # checks ``envmap_explicit``); tomography launches with the constant
    # emitter, matching training.
    args.envmap_explicit = args.envmap is not None
    if not args.envmap_explicit:
        args.envmap = str(DEFAULT_ENVMAP)
    return args


if __name__ == '__main__':
    from gfields.viewer import run_viewer
    run_viewer(parse_args())
