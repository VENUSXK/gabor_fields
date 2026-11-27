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

    def test_density_matches_the_expanded_gaussians_and_backpropagates(self):
        # use rotated, anisotropic xz kernels and sample off center in both groups.
        parameters = make_parameters()
        parameters['column_quaternions'] = mi.Float([0.4, -0.3, 0.6, -0.5])
        for value in parameters.values():
            dr.enable_grad(value)
        columns = FactorizedGaussianColumn(parameters, 2, 3)
        points = mi.Point3f([0.03, 2.08], [0.08, 0.35], [10.06, 11.91])
        centers, scales, quaternions, weights = columns.expand()
        rotation = dr.quat_to_matrix(mi.Matrix3f, quaternions)

        # compare against the normalized sum over the expanded 3d kernels.
        expected = []
        for x, y, z in points.numpy().T.tolist():
            local = (rotation.T * (mi.Point3f(x, y, z) - centers)) / scales
            pdf = dr.exp(-0.5 * dr.squared_norm(local)) / ((2.0 * np.pi) ** 1.5 * dr.prod(scales))
            expected.append(float(dr.sum(weights * pdf)[0]))
        actual = columns.eval_density(points)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6)

        # every parameter family should receive a finite density gradient.
        dr.backward(dr.sum(dr.square(actual - 1.0)))
        for name, value in parameters.items():
            gradient = dr.grad(value).numpy()
            self.assertTrue(np.isfinite(gradient).all(), name)
            self.assertTrue(np.any(gradient != 0), name)

    def test_group_tilt_starts_upright_and_has_gradients(self):
        # check identity parity while keeping both tilt parameters connected.
        parameters = make_parameters()
        upright = FactorizedGaussianColumn(parameters.copy(), 2, 3)
        for name in FactorizedGaussianColumn.tilt_parameter_names:
            parameters[name] = dr.zeros(mi.Float, 2)
            dr.enable_grad(parameters[name])
        tilted = FactorizedGaussianColumn(parameters, 2, 3)
        for actual, expected in zip(tilted.expand(), upright.expand()):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        self.assertEqual(tilted.parameter_stats()['trainable_parameter_count'],
                         upright.parameter_stats()['trainable_parameter_count'] + 4)
        points = mi.Point3f([0.03, 2.08], [0.08, 0.35], [10.06, 11.91])
        density = tilted.eval_density(points)
        np.testing.assert_array_equal(density.numpy(), upright.eval_density(points).numpy())
        dr.backward(dr.sum(density))
        for name in tilted.tilt_parameter_names:
            gradient = dr.grad(parameters[name]).numpy()
            self.assertTrue(np.isfinite(gradient).all(), name)
            self.assertTrue(np.all(gradient != 0), name)

    def test_tilted_density_matches_expansion(self):
        # combine distinct group tilts over rotated anisotropic xz factors.
        parameters = make_parameters()
        parameters['column_quaternions'] = mi.Float([0.4, -0.3, 0.6, -0.5])
        parameters['group_tilt_x'] = mi.Float([0.1, -0.2])
        parameters['group_tilt_z'] = mi.Float([-0.05, 0.15])
        columns = FactorizedGaussianColumn(parameters, 2, 3)
        centers, scales, quaternions, weights = columns.expand()
        rotation = dr.quat_to_matrix(mi.Matrix3f, quaternions)
        points = mi.Point3f([0.03, 2.08], [0.08, 0.35], [10.06, 11.91])

        # independently evaluate all expanded kernels at each world-space point.
        expected = []
        for x, y, z in points.numpy().T.tolist():
            local = (rotation.T * (mi.Point3f(x, y, z) - centers)) / scales
            pdf = dr.exp(-0.5 * dr.squared_norm(local)) / ((2.0 * np.pi) ** 1.5 * dr.prod(scales))
            expected.append(float(dr.sum(weights * pdf)[0]))
        np.testing.assert_allclose(columns.eval_density(points).numpy(), expected, rtol=2e-5)

    def test_per_xz_height_weights(self):
        # duplicate nonuniform shared weights before allowing each xz factor to vary.
        parameters = make_parameters()
        shared = FactorizedGaussianColumn(parameters.copy(), 2, 3)
        y_logits = np.repeat(parameters['y_logits'].numpy().reshape(2, 1, 3), 2, axis=1)
        parameters['y_logits'] = mi.Float(y_logits.ravel())
        columns = FactorizedGaussianColumn(parameters, 2, 3, per_xz_y_weights=True)
        for actual, expected in zip(columns.expand(), shared.expand()):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        points = mi.Point3f([0.03, 2.08], [0.08, 0.35], [10.06, 11.91])
        np.testing.assert_allclose(columns.eval_density(points).numpy(), shared.eval_density(points).numpy(), rtol=1e-6)

        # change only one horizontal factor and compare the density against expanded gaussians.
        y_logits[0, 0] = [2.0, -0.5, 0.1]
        parameters['y_logits'] = mi.Float(y_logits.ravel())
        dr.enable_grad(parameters['y_logits'])
        centers, scales, quaternions, weights = columns.expand()
        np.testing.assert_array_equal(weights.numpy()[3:], shared.expand()[3].numpy()[3:])
        expected = []
        rotation = dr.quat_to_matrix(mi.Matrix3f, quaternions)
        for x, y, z in points.numpy().T.tolist():
            local = (rotation.T * (mi.Point3f(x, y, z) - centers)) / scales
            pdf = dr.exp(-0.5 * dr.squared_norm(local)) / ((2.0 * np.pi) ** 1.5 * dr.prod(scales))
            expected.append(float(dr.sum(weights * pdf)[0]))
        density = columns.eval_density(points)
        np.testing.assert_allclose(density.numpy(), expected, rtol=2e-6)
        dr.backward(dr.sum(density))
        self.assertTrue(np.all(np.isfinite(dr.grad(parameters['y_logits']).numpy())))
        self.assertTrue(np.all(dr.grad(parameters['y_logits']).numpy().reshape(4, 3)[0] != 0))

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
