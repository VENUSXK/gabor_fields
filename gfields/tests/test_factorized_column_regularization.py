"""Check FGC factor regularization and rejection of independent primitive adaptation."""
import unittest
from types import SimpleNamespace
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import FactorizedGaussianColumn
from gfields.tests.test_factorized_gaussian_column import make_parameters
from gfields.training.regression import (
    opacity_scale_regularization, accumulate_column_y_gradients, split_column_primitives_y,
    prune_column_primitives, adapt_column_primitives,
)


class FactorizedColumnRegularizationTests(unittest.TestCase):
    def setUp(self):
        # bind live factors with distinct geometry and group masses.
        self.opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
        for name, value in make_parameters().items():
            self.opt[name] = value
        self.factors = FactorizedGaussianColumn(self.opt, 2, 3)
        self.args = SimpleNamespace(use_columns_representation=False, use_factorized_columns_representation=True,
                                    lambda_opacity=0.5, lambda_scale=0.2)

    def test_scalar_penalties_and_gradients(self):
        # count physical group mass and stored factor scales without Cartesian duplication.
        opacity, scale = opacity_scale_regularization(self.opt, self.args, 0)
        self.assertEqual(dr.width(opacity), 1)
        self.assertEqual(dr.width(scale), 1)
        self.assertAlmostEqual(float(opacity[0]), 0.6, places=6)
        self.assertAlmostEqual(float(scale[0]), 0.3, places=6)
        expanded_mass = dr.sum(self.factors.expand()[3]) * self.args.lambda_opacity
        np.testing.assert_allclose(opacity.numpy(), expanded_mass.numpy(), rtol=1e-6)
        dr.backward(opacity + scale)
        np.testing.assert_allclose(dr.grad(self.opt['group_opacity']).numpy(), 0.5)
        for name in ('column_scale_xz', 'primitive_scale_y'):
            np.testing.assert_allclose(dr.grad(self.opt[name]).numpy(), 0.2)
        for name in ('xz_logits', 'y_logits'):
            np.testing.assert_allclose(dr.grad(self.opt[name]).numpy(), 0)

    def test_logit_independence_and_zero_coefficients(self):
        # redistribution within either normalized mixture must not change either penalty.
        expected = opacity_scale_regularization(self.opt, self.args, 0)
        self.opt['xz_logits'] = mi.Float([100, -100, -10, 20])
        self.opt['y_logits'] = mi.Float([20, -30, 0, 10, -10, 1])
        for before, after in zip(expected, opacity_scale_regularization(self.opt, self.args, 0)):
            np.testing.assert_allclose(after.numpy(), before.numpy())
        self.args.lambda_opacity = self.args.lambda_scale = 0
        for penalty in opacity_scale_regularization(self.opt, self.args, 0):
            np.testing.assert_allclose(penalty.numpy(), 0)

    def test_reject_adaptive_operations(self):
        # fail explicitly before any observation history or parameter array is changed.
        operations = (
            lambda: accumulate_column_y_gradients(self.factors),
            lambda: split_column_primitives_y(self.factors, gradient_threshold=1e-4,
                                               max_primitives_per_column=16, max_primitives=100, max_splits=10),
            lambda: prune_column_primitives(self.factors, relative_weight_threshold=0.01, peak_density_threshold=0.001),
            lambda: adapt_column_primitives(self.factors, SimpleNamespace(use_adaptive_columns=True), 0),
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'FGC does not support'):
                operation()
        self.assertFalse(hasattr(self.factors, 'y_gradient_sum'))
        self.assertEqual(adapt_column_primitives(self.factors, SimpleNamespace(use_adaptive_columns=False), 0),
                         {'pruned': 0, 'split': 0})


if __name__ == '__main__':
    unittest.main()
