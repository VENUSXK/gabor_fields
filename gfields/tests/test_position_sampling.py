"""Check occupied voxel centers map from ZYX arrays into world XYZ coordinates."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mitsuba as mi
from gfields.training.regression import importance_sample_positions, generate_random_kernels


class PositionSamplingTests(unittest.TestCase):
    def setUp(self):
        # isolate one occupied voxel so sample positions are deterministic.
        self.volume = np.zeros((2, 3, 5), dtype=np.float32)
        self.volume[1, 0, 4] = 1.0
        self.config = SimpleNamespace(to_world=mi.ScalarTransform4f.scale([10, 6, 4]).translate(-0.5))

    def test_noncubic_axes_and_voxel_centers(self):
        # repeat occupied voxels when the requested count exceeds their number.
        points = importance_sample_positions(self.volume, self.config, n_gmm=3)
        np.testing.assert_allclose(points, [[4, -2, 1]] * 3, atol=1e-6)
        self.assertEqual(points.dtype, np.float32)
        self.assertTrue(points.flags.c_contiguous)

    def test_rotation_and_translation(self):
        # apply the full reference-volume transform rather than scale alone.
        self.config.to_world = mi.ScalarTransform4f.translate([1, 2, 3]).rotate([0, 1, 0], 90).scale([10, 6, 4]).translate(-0.5)
        points = importance_sample_positions(self.volume, self.config, n_gmm=1)
        np.testing.assert_allclose(points, [[2, 0, -1]], atol=1e-6)

    def test_gaussian_initialization_keeps_world_xyz_order(self):
        # do not reverse coordinates a second time after occupied-voxel sampling.
        with patch('gfields.training.regression.importance_sample_positions', return_value=np.array([[4, -2, 1]], dtype=np.float32)):
            kernels = generate_random_kernels(1, 1e-4, 0.0, 1.0, 0.0, 0, importance_sample_means=True,
                                              volume=self.volume, volume_config=self.config, target_level=0)
        np.testing.assert_allclose(kernels['centers'].numpy(), [[4, -2, 1]])

    def test_empty_volume_and_invalid_sample_count(self):
        # fail clearly when occupied-voxel sampling has no valid input.
        with self.assertRaises(ValueError):
            importance_sample_positions(np.zeros_like(self.volume), self.config, n_gmm=1)
        with self.assertRaises(ValueError):
            importance_sample_positions(self.volume, self.config, n_gmm=0)


if __name__ == '__main__':
    unittest.main()
