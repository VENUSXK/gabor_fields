"""Check matched FGC initialization, scene-to-optimizer recovery and live training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.tests.test_factorized_gaussian_column import make_parameters
from gfields.integrators.common import FactorizedGaussianColumn
from gfields.training.callbacks import PlotCallback
from gfields.training.configs import ThinLayerConfig
from gfields.training.regression import (
    generate_random_factorized_columns, generate_random_kernels, create_scene_dict_from_primitives,
    write_scene_params_to_optimizer, write_optimizer_params_to_scene, configure_optimizer,
    create_lr_dict, opacity_scale_regularization,
)


class FactorizedColumnTrainingTests(unittest.TestCase):
    def test_matched_independent_initialization(self):
        # create exactly the same Gaussian geometry for both optimization representations.
        volume = np.ones((12, 6, 10), dtype=np.float32)
        config = ThinLayerConfig(volume.shape)
        kwargs = dict(volume=volume, volume_config=config, column_slab_depth=0, seed_offset=7)
        factors = generate_random_factorized_columns(4, 2, 3, 0.002, 1.0, 0.4, **kwargs)
        kernels = generate_random_kernels(24, 0.002, 0, 1.0, 0.4, level=0,
                                         init_from_factorized_columns=True, xz_gaussians_per_group=2,
                                         y_gaussians_per_group=3, **kwargs)
        for name, values in zip(('centers', 'scales', 'quaternions', 'opacities'), factors.expand()):
            expected = values.numpy().T if name != 'opacities' else values.numpy().reshape(-1, 1)
            np.testing.assert_array_equal(np.asarray(kernels[name]), expected)

    def test_scene_recovery_and_live_gradients(self):
        # recover nonuniform factors from a real expanded scene before creating optimizer state.
        parameters = make_parameters()
        parameters['column_center_xz'] = mi.Point2f([-0.3, 0.2, -0.1, 0.4], [-0.2, 0.1, 0.3, -0.1])
        factors = FactorizedGaussianColumn(parameters, 2, 3)
        expected = [value.numpy().copy() for value in factors.expand()]
        scene = mi.load_dict(create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=factors))
        scene_params = mi.traverse(scene)
        opt = gfields.optimizers.BoundedAdam(noise_init=0)
        args = SimpleNamespace(use_columns_representation=False, use_factorized_columns_representation=True,
                               centers_lr=0.1, scales_lr=0.03, quats_lr=0.1, opacity_lr=0.001,
                               mixture_lr=1.0, xz_gaussians_per_group=2, y_gaussians_per_group=3,
                               lambda_opacity=0.0001, lambda_scale=0.000001)
        write_scene_params_to_optimizer(scene_params, opt, 0, use_columns_representation=True, columns=factors)
        rates = create_lr_dict(args, 0.003, 0)
        configure_optimizer(opt, args, rates, 0, columns=factors)
        self.assertEqual(set(rates), set(factors.parameter_names))
        self.assertAlmostEqual(rates['group_opacity'], 0.003 * 0.001 * 6)
        self.assertEqual(opt.bounds['group_opacity'], (None, 0.0))
        for original, recovered in zip(expected, factors.expand()):
            np.testing.assert_allclose(recovered.numpy(), original, rtol=1e-6, atol=1e-7)

        # propagate the expanded scene gradient to every optimized factor family.
        write_optimizer_params_to_scene(opt, scene_params, 0, use_columns_representation=True, columns=factors)
        ellipsoids = gfields.integrators.common.Ellipsoid.unravel(scene_params['primitives_pyr0.data'])
        coefficients = mi.Float(np.arange(12))
        opacity_penalty, scale_penalty = opacity_scale_regularization(opt, args, 0)
        loss = dr.sum(scene_params['primitives_pyr0.opacities'] * coefficients)
        loss += dr.sum(ellipsoids.center[0]) + dr.sum(ellipsoids.center[1]) + opacity_penalty + scale_penalty
        dr.backward(loss)
        for name in ('xz_logits', 'y_logits', 'group_opacity', 'column_center_xz', 'primitive_center_y'):
            gradient = dr.grad(opt[name]).numpy()
            self.assertTrue(np.isfinite(gradient).all(), name)
            self.assertTrue(np.any(gradient != 0), name)
        opt.step()
        write_optimizer_params_to_scene(opt, scene_params, 0, use_columns_representation=True, columns=factors)
        self.assertTrue(np.isfinite(scene_params['primitives_pyr0.opacities'].numpy()).all())

    def test_factor_histograms(self):
        # uniform or nearly uniform factor weights must still produce valid histograms.
        factors = FactorizedGaussianColumn(make_parameters(), 2, 3)
        callback = PlotCallback(factors)
        before, after = callback._column_histogram_data()
        self.assertIn('xz_density', before)
        self.assertIn('y_density', before)
        self.assertEqual(len(after['density']), 12)
        with tempfile.TemporaryDirectory() as directory:
            callback.args = SimpleNamespace(output=Path(directory))
            callback.level = 0
            callback._make_histogram(1, 'xz_weight', np.array([1e-4, np.nextafter(np.float32(1e-4), np.float32(1))], np.float32), bins=100)
            self.assertTrue(list(Path(directory).rglob('*.png')))


if __name__ == '__main__':
    unittest.main()
