"""Check factorized Gaussian weights, layout, gradients, and scene writing."""
import unittest
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import FactorizedGaussianColumn, GaussianColumn


def make_parameters(j=2, k=3):
    # use two distinct groups to detect accidental cross-group sharing.
    return {
        'column_center_xz': mi.Point2f(np.arange(2 * j), np.arange(2 * j) + 10),
        'column_scale_xz': mi.Vector2f(0.1, 0.2) + dr.zeros(mi.Vector2f, 2 * j),
        'column_quaternions': dr.zeros(mi.Float, 2 * j),
        'xz_logits': mi.Float(np.tile(np.arange(j), 2)),
        'primitive_center_y': mi.Float(np.arange(2 * k) * 0.1),
        'primitive_scale_y': dr.full(mi.Float, 0.05, 2 * k),
        'y_logits': mi.Float(np.tile(np.arange(k), 2)),
        'group_opacity': mi.Float([0.4, 0.8])
    }


class FactorizedGaussianColumnTests(unittest.TestCase):
    def test_layout_and_mass(self):
        # compare Cartesian products against independently normalized weights.
        parameters = make_parameters()
        columns = FactorizedGaussianColumn(parameters, 2, 3)
        centers, scales, quaternions, opacity = columns.expand()
        a = np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum()
        b = np.exp([0.0, 1.0, 2.0]) / np.exp([0.0, 1.0, 2.0]).sum()
        expected = np.array([0.4, 0.8])[:, None, None] * a[None, :, None] * b[None, None, :]
        np.testing.assert_allclose(opacity.numpy().reshape(2, 2, 3), expected, rtol=1e-6)
        np.testing.assert_allclose(opacity.numpy().reshape(2, -1).sum(axis=1), [0.4, 0.8], rtol=1e-6)
        np.testing.assert_allclose(centers[0].numpy(), np.repeat(np.arange(4), 3))
        np.testing.assert_allclose(centers[1].numpy(), np.tile(np.arange(3) * 0.1, 4) + np.repeat([0, 0.3], 6), atol=1e-7)
        self.assertEqual(dr.width(scales), 12)
        self.assertEqual(dr.width(quaternions), 12)

    def test_stable_softmax_and_zero_mass(self):
        # large finite logits must remain normalized without overflow.
        parameters = make_parameters()
        parameters['xz_logits'] += 10000
        parameters['y_logits'] -= 10000
        parameters['group_opacity'] = mi.Float([0, 0.8])
        opacity = FactorizedGaussianColumn(parameters, 2, 3).expand()[3].numpy()
        self.assertTrue(np.isfinite(opacity).all())
        np.testing.assert_allclose(opacity[:6], 0)
        self.assertAlmostEqual(float(opacity[6:].sum()), 0.8, places=6)

    def test_gradients(self):
        # compare normalized weight gradients with finite differences.
        parameters = make_parameters()
        for name in ('xz_logits', 'y_logits', 'group_opacity', 'column_center_xz', 'primitive_center_y'):
            dr.enable_grad(parameters[name])
        columns = FactorizedGaussianColumn(parameters, 2, 3)
        centers, _, _, opacity = columns.expand()
        coefficients = mi.Float(np.arange(12))
        dr.backward(dr.sum(opacity * coefficients) + dr.sum(centers[0]) + dr.sum(centers[1]))
        np.testing.assert_allclose(dr.grad(parameters['column_center_xz'])[0].numpy(), 3)
        np.testing.assert_allclose(dr.grad(parameters['primitive_center_y']).numpy(), 2)
        for name in ('xz_logits', 'y_logits', 'group_opacity'):
            actual = dr.grad(parameters[name]).numpy()
            for index in range(len(actual)):
                results = []
                for delta in (-0.001, 0.001):
                    changed = make_parameters()
                    values = changed[name].numpy().copy()
                    values[index] += delta
                    changed[name] = mi.Float(values)
                    results.append(float(dr.sum(FactorizedGaussianColumn(changed, 2, 3).expand()[3] * coefficients)[0]))
                self.assertAlmostEqual(float(actual[index]), (results[1] - results[0]) / 0.002, delta=0.002)

    def test_column_equivalence_and_scene_write(self):
        from gfields.training.regression import write_optimizer_params_to_scene

        # one XZ factor per group reproduces an ordinary column exactly.
        parameters = make_parameters(j=1)
        factors = FactorizedGaussianColumn(parameters, 1, 3)
        independent = {name: parameters[name] for name in GaussianColumn.parameter_names if name != 'primitive_opacity'}
        independent['primitive_opacity'] = factors.expand()[3]
        columns = GaussianColumn(independent, 3)
        for expected, actual in zip(columns.expand(), factors.expand()):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)
        scene_params = {}
        write_optimizer_params_to_scene(None, scene_params, 0, use_columns_representation=True, columns=factors)
        np.testing.assert_allclose(scene_params['primitives_pyr0.opacities'].numpy(), independent['primitive_opacity'].numpy())
        self.assertEqual(dr.width(scene_params['primitives_pyr0.albedo']), 18)

    def test_optimizer_binding(self):
        # preserve live optimizer updates and constrain total group weights.
        opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
        for name, value in make_parameters().items():
            opt[name] = value
        factors = FactorizedGaussianColumn(opt, 2, 3)
        self.assertIs(factors.parameters, opt)
        self.assertEqual(opt.bounds['group_opacity'][1], 0.0)
        opt['group_opacity'] = mi.Float([1, 2])
        np.testing.assert_allclose(factors.expand()[3].numpy().reshape(2, -1).sum(axis=1), [1, 2], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
