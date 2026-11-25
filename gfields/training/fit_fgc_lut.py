"""Fit local nonnegative XZ and Y textures directly to a scalar density grid."""

import argparse
import json
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from gfields.io import load_ellipsoid_ply
from gfields.optimizers import BoundedAdam
from gfields.rendering.fgc_lut import fgc_factor_support, pack_fgc_luts, sample_fgc_lut_factors
from gfields.rendering.pipeline import resolve_asset_path
from gfields.training.configs import get_volume_config
from gfields.training.regression import sample_density_batch


class FgcLutFit:
    def __init__(self, lower, upper, xz_resolution, y_resolution, init_density, learning_rate):
        # initialize both factors uniformly so every texel can receive a gradient.
        self.groups = len(lower)
        self.xz_shape = (self.groups, xz_resolution, xz_resolution, 1)
        self.y_shape = (self.groups, y_resolution, 1)
        self.lower, self.upper = lower, upper
        self.lower_gpu = mi.Point3f(np.ascontiguousarray(lower.T, dtype=np.float32))
        self.upper_gpu = mi.Point3f(np.ascontiguousarray(upper.T, dtype=np.float32))
        self.opt = BoundedAdam(lr=learning_rate, noise_init=0)
        for name, shape in (('xz', self.xz_shape), ('y', self.y_shape)):
            self.opt[name] = dr.full(mi.Float, np.sqrt(init_density), int(np.prod(shape)))
            self.opt.set_bounds(name, lower=0.0)

    def eval_density(self, points):
        # keep only point and group pairs inside the fixed local bounds.
        point_count = dr.width(points)
        pairs = dr.arange(mi.UInt32, point_count * self.groups)
        group = pairs % self.groups
        point = dr.gather(mi.Point3f, points, pairs // self.groups)
        lower = dr.gather(mi.Point3f, self.lower_gpu, group)
        upper = dr.gather(mi.Point3f, self.upper_gpu, group)
        indices = dr.compress(dr.all((point >= lower) & (point <= upper)))
        density = dr.zeros(mi.Float, point_count)
        if dr.width(indices) == 0:
            return density

        # interpolate live fp32 textures with differentiable drjit lookups.
        uvw = dr.gather(mi.Vector3f, mi.Vector3f((point - lower) / (upper - lower)), indices)
        layer = (mi.Float(indices % self.groups) + 0.5) / self.groups
        xz = mi.Texture3f(mi.TensorXf(self.opt['xz'], shape=self.xz_shape), use_accel=False)
        y = mi.Texture2f(mi.TensorXf(self.opt['y'], shape=self.y_shape), use_accel=False)
        values = sample_fgc_lut_factors(xz, y, uvw, layer)
        dr.scatter_reduce(dr.ReduceOp.Add, density, values, indices // self.groups)
        return density

    def export(self, albedo):
        # pack exactly the same fp16 texture format as the gaussian baking path.
        xz = self.opt['xz'].numpy().reshape(self.xz_shape[:-1])
        y = self.opt['y'].numpy().reshape(self.y_shape[:-1])
        return pack_fgc_luts(xz, y, self.lower, self.upper, np.ones(self.groups), albedo)


def main():
    # configure direct density fitting independently of the gaussian training loop.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume_grid', required=True, help='Input NPY density grid in Z,Y,X order')
    parser.add_argument('--layout', required=True, type=Path, help='FGC PLY or asset directory supplying group bounds')
    parser.add_argument('--output', required=True, type=Path, help='Output .npz texture table archive')
    parser.add_argument('--opacity_scale', type=float, default=None, help='Density scale used by the reference volume')
    parser.add_argument('--xz_res', type=int, default=64, help='Texture resolution on each XZ axis')
    parser.add_argument('--y_res', type=int, default=128, help='Texture resolution on Y')
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--density_batch_size', type=int, default=4096)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--init_density', type=float, default=0.1, help='Initial constant extinction per group')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--seed_offset', type=int, default=0)
    args = parser.parse_args()
    if args.output.suffix.lower() != '.npz' or min(args.xz_res, args.y_res) < 2:
        parser.error('Use an .npz output and texture resolutions of at least two')
    if min(args.iterations, args.density_batch_size, args.log_every) < 1:
        parser.error('Iterations, batch size, and log interval must be positive')
    if not all(np.isfinite(value) and value > 0 for value in (args.learning_rate, args.init_density)):
        parser.error('Learning rate and initial density must be positive and finite')

    # use the existing fgc bounds and density scaling without initializing from gaussians.
    ply = args.layout if args.layout.suffix.lower() == '.ply' else resolve_asset_path(args.layout) / 'data' / 'root.primitives_pyr0.ply'
    factors = load_ellipsoid_ply(ply, use_factorized_columns_representation=True)
    lower, upper, *_ = fgc_factor_support(factors)
    volume = np.load(args.volume_grid, mmap_mode='r', allow_pickle=False)
    if volume.ndim != 3 or min(volume.shape) == 0:
        parser.error('--volume_grid must be a nonempty (Z,Y,X) density array')
    config = get_volume_config(args.volume_grid, opacity_scale=args.opacity_scale)
    model = FgcLutFit(lower, upper, args.xz_res, args.y_res, args.init_density, args.learning_rate)
    print(f'Fitting {model.groups} groups directly from density ({args.xz_res}x{args.xz_res} XZ, {args.y_res} Y)')

    # train on mixed uniform and foreground samples and log a fixed evaluation batch.
    rng = np.random.default_rng(args.seed_offset % (2**32))
    eval_points, eval_density = sample_density_batch(volume, config, args.density_batch_size, np.random.default_rng([0, 1]))
    metrics = []
    for iteration in range(args.iterations + 1):
        if iteration > 0:
            points, target = sample_density_batch(volume, config, args.density_batch_size, rng)
            loss = dr.mean(dr.square(model.eval_density(points) - target))
            if dr.grad_enabled(loss):
                dr.backward(loss)
                model.opt.step()
        if iteration % args.log_every == 0 or iteration == args.iterations:
            with dr.suspend_grad():
                mse = float(dr.mean(dr.square(model.eval_density(eval_points) - eval_density))[0])
            metrics.append(dict(iteration=iteration, density_mse=mse))
            print(f'Iteration {iteration}/{args.iterations} | density_mse {mse:.6g}', flush=True)

    # write render-ready textures and the density metrics sidecar.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **model.export(factors['albedo']))
    args.output.with_name(f'{args.output.stem}_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
