"""Check compact FGC texture baking and transmittance integration."""

import tempfile
import unittest

import drjit as dr
import mitsuba as mi
import numpy as np

import gfields
from gfields.io import load_ellipsoid_ply
from gfields.rendering.fgc_lut import FgcLutVolume, load_fgc_luts
from gfields.tests.test_factorized_column_ply import save_factors


@unittest.skipUnless(mi.variant().startswith('cuda'), 'FGC LUT rendering requires CUDA textures')
class FgcLutTests(unittest.TestCase):
    def test_cached_textures_match_factor_values(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, path = save_factors(directory)
            factors = load_ellipsoid_ply(path, use_factorized_columns_representation=True)
            tables = load_fgc_luts(path, directory, 128, 128)
            cached = load_fgc_luts(path, directory, 128, 128)
        for name in tables:
            np.testing.assert_array_equal(tables[name], cached[name])
        self.assertEqual(tables['xz'].dtype, np.float16)
        self.assertEqual(tables['y'].dtype, np.float16)

        # compare paired texture lookups against a direct, separately truncated gaussian sum.
        volume = FgcLutVolume(tables)
        groups = dr.arange(mi.UInt32, 2)
        points = np.array([[0.23, 0.185, 0.14], [0.38, 0.285, -0.06]])
        actual = volume._density(mi.Point3f(np.ascontiguousarray(points.T)), volume.lower, volume.upper, groups, volume.weights, True)
        expected = []
        for group, (x, y, z) in enumerate(points):
            xz_rows, y_rows = slice(2 * group, 2 * group + 2), slice(3 * group, 3 * group + 3)
            dx, dz = (np.array([x, z]) - factors['column_center_xz'][xz_rows]).T
            sx, sz = factors['column_scale_xz'][xz_rows].T
            theta = factors['column_quaternions'][xz_rows]
            c, s = np.cos(theta), np.sin(theta)
            q = ((c * dx - s * dz) / sx) ** 2 + ((s * dx + c * dz) / sz) ** 2
            xz = np.sum(np.where(q <= factors['extent']**2, factors['xz_weight'][xz_rows] * np.exp(-0.5 * q) / (2 * np.pi * sx * sz), 0))
            sy = factors['primitive_scale_y'][y_rows]
            q = ((y - factors['primitive_center_y'][y_rows]) / sy) ** 2
            axial = np.sum(np.where(q <= factors['extent']**2, factors['y_weight'][y_rows] * np.exp(-0.5 * q) / (np.sqrt(2 * np.pi) * sy), 0))
            expected.append(factors['group_opacity'][group] * xz * axial)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=0.004)

    def test_transmittance_adds_overlapping_segments(self):
        # check an overlapping ray, a parallel miss, and a ray starting inside both groups.
        tables = dict(xz=np.ones((2, 4, 4), np.float16), y=np.ones((2, 4), np.float16),
                      lower=np.array([[0, 0, 0], [0, 0.25, 0]], np.float32),
                      upper=np.array([[1, 1, 1], [1, 0.75, 1]], np.float32), weights=np.array([1, 2], np.float32))
        volume = FgcLutVolume(tables, world_scale=2, opacity_scale=4)
        rays = mi.Ray3f(mi.Point3f([1, 4, 1], [-1, -1, 1], 1), mi.Vector3f(0, 1, 0))
        transmission = dr.exp(-volume.optical_depth(rays, 0.08))
        np.testing.assert_allclose(transmission.numpy(), np.exp([-2, 0, -1]), rtol=1e-6)

    def test_free_flight_samples_summed_extinction(self):
        # overlapping homogeneous groups have an analytic piecewise exponential cdf.
        tables = dict(xz=np.ones((2, 4, 4), np.float16), y=np.ones((2, 4), np.float16),
                      lower=np.array([[0, 0, 0], [0, 0.25, 0]], np.float32),
                      upper=np.array([[1, 1, 1], [1, 0.75, 1]], np.float32), weights=np.array([1, 2], np.float32))
        volume = FgcLutVolume(tables)
        count = 16_384
        rays = mi.Ray3f(mi.Point3f(0.5, dr.full(mi.Float, -1, count), 0.5), mi.Vector3f(0, 1, 0))
        seed = mi.PCG32(count, initseq=dr.arange(mi.UInt64, count)).next_uint32()
        distance = volume.sample_free_flight(rays, 0.17, seed).numpy()
        self.assertTrue(np.all(distance >= 1.0))
        for t, depth in [(1.125, 0.125), (1.5, 1.0), (2.0, 2.0)]:
            self.assertAlmostEqual(np.mean(distance >= t), np.exp(-depth), delta=0.012)

    def test_multiple_bounces_restore_energy_in_conservative_medium(self):
        # compare direct transmission and scattering under constant colored illumination.
        tables = dict(xz=np.ones((1, 4, 4), np.float16), y=np.ones((1, 4), np.float16),
                      lower=np.zeros((1, 3), np.float32), upper=np.ones((1, 3), np.float32), weights=np.ones(1, np.float32))
        volume = FgcLutVolume(tables)
        count = 8192
        rays = mi.Ray3f(mi.Point3f(0.5, dr.full(mi.Float, -1, count), 0.5), mi.Vector3f(0, 1, 0))
        color = np.array([0.25, 0.5, 0.75])
        environment = mi.load_dict({'type': 'constant', 'radiance': {'type': 'rgb', 'value': color.tolist()}})
        results = []
        for max_depth in (0, 1, 12):
            rng = mi.PCG32(count, initseq=dr.arange(mi.UInt64, count))
            image = volume._scatter(rays, rng, 0.17, environment, mi.Color3f(1), max_depth, -1)
            results.append(image.numpy().mean(axis=1))
        np.testing.assert_allclose(results[0], color * np.exp(-1), atol=0.012)
        self.assertTrue(np.all(results[1] > results[0]))
        self.assertTrue(np.all(results[2] > results[1]))
        np.testing.assert_allclose(results[2], color, atol=0.006)
