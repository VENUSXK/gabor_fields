"""Check slab-aware FGC initialization, local XZ sampling, and density budgets."""
import unittest
from types import SimpleNamespace
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.training.regression import generate_random_columns, generate_random_factorized_columns


class FactorizedColumnInitializationTests(unittest.TestCase):
    def setUp(self):
        # use a noncubic volume with zero padding around the occupied Y range.
        self.volume = np.zeros((19, 10, 27), dtype=np.float32)
        self.volume[3:16, 2:8, 4:23] = 1.0
        self.config = SimpleNamespace(opacity_scale=10.0,
                                      to_world=mi.ScalarTransform4f().scale([2.7, 1.0, 1.9]).translate(-0.5))

    def generate(self, j=4, k=3, slab_depth=2, seed=7):
        return generate_random_factorized_columns(
            12, j, k, 0.002, 1.0, 0.6, volume=self.volume, volume_config=self.config,
            column_slab_depth=slab_depth, seed_offset=seed,
        )

    def test_layout_weights_and_slabs(self):
        # retain one Y mixture per group and equal opacity for every expanded pair.
        factors = self.generate()
        parameters = factors.parameters
        self.assertEqual(dr.width(parameters['column_center_xz']), 48)
        self.assertEqual(dr.width(parameters['primitive_center_y']), 36)
        self.assertNotIn('primitive_opacity', parameters)
        self.assertIsNone(factors.column_optimizer)
        self.assertEqual(factors.init_albedo, 0.6)
        np.testing.assert_allclose(parameters['xz_logits'].numpy(), 0)
        np.testing.assert_allclose(parameters['y_logits'].numpy(), 0)
        np.testing.assert_allclose(parameters['group_opacity'].numpy(), 0.024)
        np.testing.assert_allclose(factors.expand()[3].numpy(), 0.002, rtol=1e-6)
        self.assertAlmostEqual(float(dr.sum(factors.expand()[3])[0]), 12 * 4 * 3 * 0.002, places=6)

        # distribute groups over the three occupied slabs, excluding zero padding.
        centers_y = parameters['primitive_center_y'].numpy().reshape(12, 3)
        expected = (2 + (np.arange(12) % 3)[:, None] * 2 + (np.arange(3) + 0.5) * 2 / 3) / 10 - 0.5
        np.testing.assert_allclose(centers_y, expected, atol=1e-7)
        scales_y = parameters['primitive_scale_y'].numpy()
        self.assertTrue(np.all(scales_y >= factors.scale_lower_bounds['primitive_scale_y']))

    def test_local_density_sampling_and_reproducibility(self):
        # keep all secondary footprints within the anchor's local sampling window.
        factors = self.generate()
        centers = factors.parameters['column_center_xz'].numpy().reshape(2, 12, 4)
        group_scale = 0.5 / np.sqrt(12 / 3)
        radii = np.ceil(group_scale / 0.1) * 0.1
        self.assertTrue(np.all(np.abs(centers - centers[:, :, :1]) <= radii + 1e-6))
        pixel_x = np.rint((centers[0] / 2.7 + 0.5) * 27 - 0.5).astype(int)
        pixel_z = np.rint((centers[1] / 1.9 + 0.5) * 19 - 0.5).astype(int)
        self.assertTrue(np.all(self.volume[pixel_z, 2, pixel_x] > 0))
        repeated = self.generate()
        for name in factors.parameter_names:
            np.testing.assert_array_equal(factors.parameters[name].numpy(), repeated.parameters[name].numpy())
        self.assertFalse(np.array_equal(centers, self.generate(seed=8).parameters['column_center_xz'].numpy().reshape(2, 12, 4)))

    def test_single_xz_factor_matches_existing_columns(self):
        # preserve the exact existing layout, scales, rotations, and density when J is one.
        factors = self.generate(j=1)
        columns = generate_random_columns(12, 3, 0.002, 1.0, 0.6, volume=self.volume,
                                          volume_config=self.config, column_slab_depth=2, seed_offset=7)
        for expected, actual in zip(columns.expand(), factors.expand()):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
        self.assertEqual(factors.scale_lower_bounds, columns.scale_lower_bounds)

    def test_full_range_and_empty_volume(self):
        # slab zero spans the occupied interval; empty density falls back to uniform sampling.
        factors = self.generate(slab_depth=0)
        expected = (2 + (np.arange(3) + 0.5) * 6 / 3) / 10 - 0.5
        np.testing.assert_allclose(factors.parameters['primitive_center_y'].numpy().reshape(12, 3),
                                   np.broadcast_to(expected, (12, 3)), atol=1e-7)
        self.volume.fill(0)
        centers, _, _, weights = self.generate(slab_depth=0).expand()
        self.assertTrue(np.isfinite(centers.numpy()).all())
        np.testing.assert_allclose(weights.numpy(), 0.002, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
