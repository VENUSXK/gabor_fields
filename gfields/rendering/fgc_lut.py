"""Bake FGC factors into local textures and march their extinction on the GPU."""

import hashlib
import tempfile
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from gfields.io import load_ellipsoid_ply


def bake_fgc_luts(factors, xz_resolution, y_resolution):
    # collect each group's geometry and use the ply extent as the factor cutoff.
    groups, j, k = (factors[name] for name in ('group_count', 'xz_gaussians_per_group', 'y_gaussians_per_group'))
    extent = factors['extent']
    xz_centers = factors['column_center_xz'].reshape(groups, j, 2).astype(np.float64)
    xz_scales = factors['column_scale_xz'].reshape(groups, j, 2).astype(np.float64)
    angles = factors['column_quaternions'].reshape(groups, j)
    y_centers = factors['primitive_center_y'].reshape(groups, k).astype(np.float64)
    y_scales = factors['primitive_scale_y'].reshape(groups, k).astype(np.float64)

    # bound the rotated xz ellipses and y intervals in asset coordinates.
    cosine, sine = np.cos(angles), np.sin(angles)
    sx, sz = xz_scales[..., 0], xz_scales[..., 1]
    xz_radius = extent * np.stack((np.hypot(cosine * sx, sine * sz), np.hypot(sine * sx, cosine * sz)), axis=-1)
    xz_min, xz_max = xz_centers - xz_radius, xz_centers + xz_radius
    y_min, y_max = y_centers - extent * y_scales, y_centers + extent * y_scales
    lower = np.column_stack((xz_min[..., 0].min(1), y_min.min(1), xz_min[..., 1].min(1)))
    upper = np.column_stack((xz_max[..., 0].max(1), y_max.max(1), xz_max[..., 1].max(1)))

    # splat each normalized 2d gaussian into its bounding rectangle in the xz texture.
    xz_lut = np.zeros((groups, xz_resolution, xz_resolution), dtype=np.float32)
    y_lut = np.zeros((groups, y_resolution), dtype=np.float32)
    xz_weights = factors['xz_weight'].reshape(groups, j)
    y_weights = factors['y_weight'].reshape(groups, k)
    xz_positions = (np.arange(xz_resolution) + 0.5) / xz_resolution
    y_positions = (np.arange(y_resolution) + 0.5) / y_resolution
    for group in range(groups):
        x, _, z = lower[group, :, None] + (upper - lower)[group, :, None] * xz_positions
        for index in range(j):
            x0, z0 = xz_min[group, index]
            x1, z1 = xz_max[group, index]
            xs = slice(x.searchsorted(x0), x.searchsorted(x1, side='right'))
            zs = slice(z.searchsorted(z0), z.searchsorted(z1, side='right'))
            dx = x[xs][None, :] - xz_centers[group, index, 0]
            dz = z[zs][:, None] - xz_centers[group, index, 1]
            c, s = cosine[group, index], sine[group, index]
            q = ((c * dx - s * dz) / sx[group, index]) ** 2 + ((s * dx + c * dz) / sz[group, index]) ** 2
            weight = xz_weights[group, index] / (2 * np.pi * sx[group, index] * sz[group, index])
            xz_lut[group, zs, xs] += np.where(q <= extent**2, weight * np.exp(-0.5 * q), 0.0)

        # splat each normalized 1d gaussian into its interval in the y texture.
        y = lower[group, 1] + (upper - lower)[group, 1] * y_positions
        for index in range(k):
            ys = slice(y.searchsorted(y_min[group, index]), y.searchsorted(y_max[group, index], side='right'))
            scale = y_scales[group, index]
            distance = (y[ys] - y_centers[group, index]) / scale
            y_lut[group, ys] += y_weights[group, index] * np.exp(-0.5 * distance**2) / (np.sqrt(2 * np.pi) * scale)

    # normalize texture values before fp16 conversion and preserve their amplitudes.
    xz_peak, y_peak = xz_lut.max(axis=(1, 2)), y_lut.max(axis=1)
    xz_lut /= np.where(xz_peak > 0, xz_peak, 1)[:, None, None]
    y_lut /= np.where(y_peak > 0, y_peak, 1)[:, None]
    weights = (factors['group_opacity'].astype(np.float64) * xz_peak * y_peak).astype(np.float32)
    return dict(xz=xz_lut.astype(np.float16), y=y_lut.astype(np.float16),
                lower=lower.astype(np.float32), upper=upper.astype(np.float32), weights=weights)


def load_fgc_luts(ply_path, cache_dir, xz_resolution, y_resolution):
    # key the cache on the source asset and texture dimensions.
    digest = hashlib.sha256(Path(ply_path).read_bytes()).hexdigest()[:20]
    cache_path = Path(cache_dir) / f'fgc_lut_v1_{digest}_{xz_resolution}_{y_resolution}.npz'
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            return dict(cached)

    # publish the baked tables atomically for reuse by other render runs.
    factors = load_ellipsoid_ply(ply_path, use_factorized_columns_representation=True)
    tables = bake_fgc_luts(factors, xz_resolution, y_resolution)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.npz', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez(temporary_path, **tables)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return tables


class FgcLutVolume:
    def __init__(self, tables, world_scale=1.0, opacity_scale=1.0):
        # store xz and y textures in separate group layers at their texel centers.
        self.group_count = len(tables['weights'])
        self.xz = mi.Texture3f16(dr.cuda.ad.TensorXf16(tables['xz'][..., None]))
        self.y = mi.Texture2f16(dr.cuda.ad.TensorXf16(tables['y'][..., None]))

        # follow the existing asset scaling of centers and normalized gaussian mass.
        self.lower = mi.Point3f(np.ascontiguousarray(tables['lower'].T * world_scale))
        self.upper = mi.Point3f(np.ascontiguousarray(tables['upper'].T * world_scale))
        self.weights = mi.Float(tables['weights'] * (opacity_scale / world_scale**3))

    def _density(self, point, lower, upper, group, weight, active):
        # look up both factors in the selected group and restore their density amplitude.
        uvw = (point - lower) / (upper - lower)
        layer = (mi.Float(group) + 0.5) / self.group_count
        xz = self.xz.eval(mi.Point3f(uvw[0], uvw[2], layer), active)[0]
        y = self.y.eval(mi.Point2f(uvw[1], layer), active)[0]
        return weight * mi.Float(xz) * mi.Float(y)

    def optical_depth(self, ray, step_size):
        # find the overlapping ray and group bounds before evaluating any density.
        ray_count = dr.width(ray.o)
        pairs = dr.arange(mi.UInt32, ray_count * self.group_count)
        group, ray_index = pairs % self.group_count, pairs // self.group_count
        origin = dr.gather(mi.Point3f, ray.o, ray_index)
        direction = dr.gather(mi.Vector3f, ray.d, ray_index)
        lower = dr.gather(mi.Point3f, self.lower, group)
        upper = dr.gather(mi.Point3f, self.upper, group)
        weight = dr.gather(mi.Float, self.weights, group)
        parallel = direction == 0
        safe_direction = dr.select(parallel, 1.0, direction)
        t0, t1 = (lower - origin) / safe_direction, (upper - origin) / safe_direction
        near = dr.maximum(0.0, dr.max(dr.select(parallel, -dr.inf, dr.minimum(t0, t1))))
        far = dr.minimum(dr.gather(mi.Float, ray.maxt, ray_index), dr.min(dr.select(parallel, dr.inf, dr.maximum(t0, t1))))
        hit = (near < far) & (weight > 0) & ~dr.any(parallel & ((origin < lower) | (origin > upper)))

        # compact misses and integrate each surviving group with midpoint steps.
        indices = dr.compress(hit)
        if dr.width(indices) == 0:
            return dr.zeros(mi.Float, ray_count)
        origin, direction, lower, upper, group, weight, near, far = (
            dr.gather(type(value), value, indices) for value in (origin, direction, lower, upper, group, weight, near, far))
        count = mi.UInt32(dr.ceil((far - near) / step_size))
        spacing = (far - near) / count

        def march(index, integrated):
            active = index < count
            point = origin + direction * (near + (mi.Float(index) + 0.5) * spacing)
            density = self._density(point, lower, upper, group, weight, active)
            return index + 1, integrated + dr.select(active, density * spacing, 0.0)

        _, integrated = dr.while_loop((mi.UInt32(0), mi.Float(0)),
                                     lambda index, _: index < count, march, label='fgc LUT transmittance')

        # sum optical depths of all additive groups before applying exponential attenuation.
        depth = dr.zeros(mi.Float, ray_count)
        dr.scatter_reduce(dr.ReduceOp.Add, depth, integrated, indices // self.group_count)
        return depth

    def render(self, sensor, spp, step_size, batch_pixels):
        # reuse the mitsuba camera and sample each pixel with independent subpixel offsets.
        width, height = map(int, sensor.film().crop_size())
        image = dr.zeros(mi.Float, width * height)
        for start in range(0, width * height, batch_pixels):
            size = min(batch_pixels, width * height - start)
            samples = dr.arange(mi.UInt32, size * spp)
            pixels = samples // spp + start
            rng = mi.PCG32(size * spp, initseq=mi.UInt64(samples) + start * spp)
            offset = mi.Point2f(0.5) if spp == 1 else mi.Point2f(rng.next_float32(), rng.next_float32())
            uv = (mi.Point2f(pixels % width, pixels // width) + offset) / mi.Vector2f(width, height)
            ray, _ = sensor.sample_ray(0.0, 0.5, uv, mi.Point2f(0.5))

            # average transmittance over samples and keep working memory bounded per tile.
            transmission = dr.exp(-self.optical_depth(ray, step_size))
            averaged = dr.mean(mi.TensorXf(transmission, shape=(size, spp)), axis=1)
            dr.scatter(image, averaged.array, dr.arange(mi.UInt32, size) + start)
            dr.eval(image)
        return mi.TensorXf(image, shape=(height, width, 1))
