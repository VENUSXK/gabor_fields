"""Check budgeted Y splitting, moment conservation, and optimizer state."""
import unittest
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import GaussianColumn
from gfields.training.regression import accumulate_column_y_gradients, split_column_primitives_y


def make_columns():
    # keep two distinct footprints and four independent axial primitives.
    opt = gfields.optimizers.BoundedAdam(lr=1e-4, noise_init=0.0)
    opt['column_center_xz'] = mi.Point2f([-0.2, 0.3], [0.1, -0.4])
    opt['column_scale_xz'] = mi.Vector2f([0.1, 0.2], [0.3, 0.4])
    opt['column_quaternions'] = mi.Float([0.0, 0.4])
    opt['primitive_center_y'] = mi.Float([-0.4, 0.3, -0.2, 0.4])
    opt['primitive_scale_y'] = mi.Float([0.2, 0.1, 0.3, 0.05])
    opt['primitive_opacity'] = mi.Float([0.2, 0.4, 0.3, 0.1])
    opt.set_bounds('primitive_center_y', lower=-1.0, upper=1.0)
    opt.set_bounds('primitive_scale_y', lower=0.02, upper=1.0)
    opt.set_bounds('primitive_opacity', lower=0.001, upper=1.0)
    return GaussianColumn(opt, 2, {'column_scale_xz': [1e-5, 1e-5], 'primitive_scale_y': 0.02})


def observe(columns, gradients, steps=4):
    # emulate gradients from several backward passes before optimizer updates.
    for step in range(steps):
        dr.set_grad(columns.parameters['primitive_center_y'], mi.Float(gradients) * (-1 if step % 2 else 1))
        accumulate_column_y_gradients(columns)


def split(columns, **overrides):
    # use explicit budgets so every test makes the growth limit visible.
    settings = dict(gradient_threshold=1.0, max_primitives_per_column=3, max_primitives=8, max_splits=4)
    settings.update(overrides)
    return split_column_primitives_y(columns, **settings)


class ColumnYSplitTests(unittest.TestCase):
    def test_split_conserves_column_moments(self):
        # record per-column mass and first two raw moments before replacement.
        columns = make_columns()
        before = columns.parameters
        centers, scales, weights = (before[name].numpy().astype(np.float64) for name in ('primitive_center_y', 'primitive_scale_y', 'primitive_opacity'))
        parents = columns.primitive_indices.numpy()
        expected = [np.bincount(parents, weights=term) for term in (weights, weights * centers, weights * (centers ** 2 + scales ** 2))]
        shared = {name: before[name].numpy().copy() for name in columns.parameter_names[:3]}
        observe(columns, [4.0, 0.2, 3.0, 0.1])
        self.assertEqual(split(columns), 2)

        # ensure two children replace each parent without altering shared geometry.
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [3, 3])
        np.testing.assert_array_equal(columns.column_offsets.numpy(), [0, 3, 6])
        np.testing.assert_allclose(before['primitive_center_y'].numpy(), [-0.5, -0.3, 0.3, -0.35, -0.05, 0.4], atol=1e-7)
        for name, values in shared.items():
            np.testing.assert_array_equal(before[name].numpy(), values)
        centers, scales, weights = (before[name].numpy().astype(np.float64) for name in ('primitive_center_y', 'primitive_scale_y', 'primitive_opacity'))
        for actual, original in zip((weights, weights * centers, weights * (centers ** 2 + scales ** 2)), expected):
            np.testing.assert_allclose(np.bincount(columns.primitive_indices.numpy(), weights=actual), original, rtol=1e-6)
        self.assertEqual(dr.width(columns.expand()[0]), 6)
        np.testing.assert_array_equal(columns.y_gradient_samples.numpy(), np.zeros(6))

    def test_gradient_ranking_and_budgets(self):
        # select the largest score globally and respect each separate column cap.
        columns = make_columns()
        observe(columns, [2.0, 8.0, 6.0, 4.0])
        self.assertEqual(split(columns, max_primitives=5), 1)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [3, 2])
        np.testing.assert_allclose(columns.parameters['primitive_center_y'].numpy(), [-0.4, 0.25, 0.35, -0.2, 0.4], atol=1e-7)
        columns = make_columns()
        observe(columns, [2.0, 8.0, 6.0, 4.0])
        self.assertEqual(split(columns, max_splits=1), 1)
        columns = make_columns()
        observe(columns, [2.0, 8.0, 6.0, 4.0])
        self.assertEqual(split(columns, max_primitives_per_column=2), 0)

    def test_scale_center_and_weight_bounds(self):
        # reject children below the scale floor, outside bounds, or below the opacity floor.
        columns = make_columns()
        opt = columns.parameters
        opt['primitive_scale_y'] = mi.Float([0.021, 0.1, 0.3, 0.05])
        opt['primitive_center_y'] = mi.Float([-0.4, 0.99, -0.2, 0.4])
        opt['primitive_opacity'] = mi.Float([0.2, 0.4, 0.001, 0.1])
        observe(columns, [5.0, 4.0, 3.0, 0.0])
        self.assertEqual(split(columns), 0)
        self.assertEqual(dr.width(opt['primitive_center_y']), 4)

    def test_observation_window_and_nonfinite_gradients(self):
        # require enough finite observations and ignore NaN or infinite gradients.
        columns = make_columns()
        self.assertEqual(split(columns), 0)
        observe(columns, [4.0, np.nan, np.inf, 0.0], steps=3)
        np.testing.assert_array_equal(columns.y_gradient_samples.numpy(), [3, 0, 0, 3])
        self.assertEqual(split(columns), 0)
        observe(columns, [4.0, np.nan, np.inf, 0.0])
        self.assertEqual(split(columns), 1)

    def test_preserves_optimizer_state_and_learning_rates(self):
        # seed nonzero moments to distinguish retained primitives from new children.
        columns = make_columns()
        opt = columns.parameters
        opt.set_learning_rate({'primitive_center_y': 0.007, 'primitive_scale_y': 0.008, 'primitive_opacity': 0.009})
        for name in columns.parameter_names[3:]:
            value, promoted, learning_rate, extra = opt.state[name]
            step, first, second, *rest = extra
            rest = [type(first)([9, 10, 11, 12])] if rest else []
            opt.state[name] = value, promoted, learning_rate, (type(step)(5), type(first)([1, 2, 3, 4]), type(second)([5, 6, 7, 8]), *rest)
        shared_state = {name: opt.state[name] for name in columns.parameter_names[:3]}
        observe(columns, [4.0, 0.0, 3.0, 0.0])
        self.assertEqual(split(columns), 2)

        # preserve old lanes and zero all child moments, including optional maximum moments.
        for name in columns.parameter_names[3:]:
            _, _, learning_rate, (step, first, second, *rest) = opt.state[name]
            self.assertEqual(learning_rate, {'primitive_center_y': 0.007, 'primitive_scale_y': 0.008, 'primitive_opacity': 0.009}[name])
            np.testing.assert_array_equal(first.numpy(), [0, 0, 2, 0, 0, 4])
            np.testing.assert_array_equal(second.numpy(), [0, 0, 6, 0, 0, 8])
            np.testing.assert_array_equal(step.numpy(), [5])
            if rest:
                np.testing.assert_array_equal(rest[0].numpy(), [0, 0, 10, 0, 0, 12])
        for name, original in shared_state.items():
            self.assertIs(opt.state[name], original)

        # verify another backward pass and optimizer step work on the resized arrays.
        loss = dr.sum(dr.square(columns.expand()[0][1]))
        dr.backward(loss)
        opt.step()
        self.assertTrue(np.all(np.isfinite(opt['primitive_center_y'].numpy())))

if __name__ == '__main__':
    unittest.main()
