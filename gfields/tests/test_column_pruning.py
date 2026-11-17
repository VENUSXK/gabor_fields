"""Check conservative pruning, split protection, and surviving optimizer state."""
import unittest
import numpy as np
import drjit as dr
import mitsuba as mi
from gfields.tests.test_column_y_split import make_columns, observe, split
from gfields.training.regression import prune_column_primitives


def weak_columns(two_weak=True):
    # make low-weight primitives distinguishable from their stronger column neighbors.
    columns = make_columns()
    columns.parameters.set_bounds('primitive_opacity', lower=1e-8, upper=1.0)
    columns.parameters['primitive_opacity'] = mi.Float([1e-5, 0.4, 2e-5 if two_weak else 0.3, 0.1])
    return columns


def prune(columns, **overrides):
    # use a relative mass threshold and an absolute normalized peak-density threshold.
    settings = dict(relative_weight_threshold=0.01, peak_density_threshold=0.01,
                    min_low_contribution_checks=3, protection_checks=2)
    settings.update(overrides)
    return prune_column_primitives(columns, **settings)


class ColumnPruningTests(unittest.TestCase):
    def test_requires_consecutive_observations(self):
        # delete only after three low-contribution checks and preserve contiguous column layout.
        columns = weak_columns()
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 2)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [1, 1])
        np.testing.assert_array_equal(columns.column_offsets.numpy(), [0, 1, 2])
        np.testing.assert_array_equal(columns.primitive_indices.numpy(), [0, 1])
        np.testing.assert_allclose(columns.parameters['primitive_center_y'].numpy(), [0.3, 0.4])
        self.assertEqual(dr.width(columns.expand()[0]), 2)

    def test_recovery_resets_the_streak(self):
        # one recovered observation must invalidate the previous low-contribution streak.
        columns = weak_columns(two_weak=False)
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 0)
        columns.parameters['primitive_opacity'] = mi.Float([0.2, 0.4, 0.3, 0.1])
        self.assertEqual(prune(columns), 0)
        np.testing.assert_array_equal(columns.primitive_low_contribution_checks.numpy(), [0, 0, 0, 0])
        columns.parameters['primitive_opacity'] = mi.Float([1e-5, 0.4, 0.3, 0.1])
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 1)

    def test_weight_and_density_are_both_required(self):
        # small mass alone must not delete a sharply concentrated high-density primitive.
        columns = weak_columns(two_weak=False)
        columns.parameters['primitive_scale_y'] = mi.Float([1e-8, 0.1, 0.3, 0.05])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 0)

        # low absolute density alone must not delete a large relative share of column mass.
        columns = make_columns()
        columns.parameters['primitive_opacity'] = mi.Float([1e-5, 2e-5, 1e-5, 2e-5])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 0)

    def test_keep_one_primitive_per_column(self):
        # even entirely empty columns retain one primitive and their shared parameters.
        columns = make_columns()
        columns.parameters['primitive_opacity'] = mi.Float([0, 0, 0, 0])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 2)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [1, 1])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 0)

        # when all candidates qualify, retain the one with the largest integrated weight.
        columns = make_columns()
        columns.parameters['primitive_opacity'] = mi.Float([1e-5, 3e-5, 4e-5, 2e-5])
        self.assertEqual(prune(columns, relative_weight_threshold=1.0, min_low_contribution_checks=1), 2)
        np.testing.assert_allclose(columns.parameters['primitive_center_y'].numpy(), [0.3, -0.2])

    def test_split_children_have_a_complete_protection_period(self):
        # a weak parent may split, but its children do not inherit its pending deletion.
        columns = weak_columns(two_weak=False)
        prune(columns)
        prune(columns)
        observe(columns, [4.0, 0.0, 0.0, 0.0])
        self.assertEqual(split(columns), 1)
        np.testing.assert_array_equal(columns.primitive_low_contribution_checks.numpy(), [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(columns.primitive_prune_check_count.numpy()[:2], [0, 0])
        for _ in range(2):
            self.assertEqual(prune(columns), 0)
            np.testing.assert_array_equal(columns.primitive_low_contribution_checks.numpy()[:2], [0, 0])
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 0)
        self.assertEqual(prune(columns), 2)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [1, 2])

    def test_optimizer_state_and_gradients_follow_survivors(self):
        # seed nonzero moments and per-lane steps to expose incorrect state compaction.
        columns = weak_columns()
        opt = columns.parameters
        opt.set_learning_rate({'primitive_center_y': 0.007, 'primitive_scale_y': 0.008, 'primitive_opacity': 0.009})
        for name in columns.parameter_names[3:]:
            value, promoted, learning_rate, extra = opt.state[name]
            step, first, second, *rest = extra
            rest = [type(first)([9, 10, 11, 12])] if rest else []
            opt.state[name] = value, promoted, learning_rate, (type(step)([1, 2, 3, 4]), type(first)([1, 2, 3, 4]), type(second)([5, 6, 7, 8]), *rest)
        shared_state = {name: opt.state[name] for name in columns.parameter_names[:3]}
        observe(columns, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 2)
        np.testing.assert_allclose(columns.y_gradient_sum.numpy(), [8, 16])
        np.testing.assert_array_equal(columns.y_gradient_samples.numpy(), [4, 4])

        # retain learning rates, steps, optional maximum moments, and shared XZ state.
        for name in columns.parameter_names[3:]:
            _, _, learning_rate, (step, first, second, *rest) = opt.state[name]
            self.assertEqual(learning_rate, {'primitive_center_y': 0.007, 'primitive_scale_y': 0.008, 'primitive_opacity': 0.009}[name])
            np.testing.assert_array_equal(step.numpy(), [2, 4])
            np.testing.assert_array_equal(first.numpy(), [2, 4])
            np.testing.assert_array_equal(second.numpy(), [6, 8])
            if rest:
                np.testing.assert_array_equal(rest[0].numpy(), [10, 12])
        for name, state in shared_state.items():
            self.assertIs(opt.state[name], state)

        # preserved gradient windows remain usable by splitting after pruning.
        self.assertEqual(split(columns), 2)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [2, 2])
        dr.backward(dr.sum(dr.square(columns.expand()[0][1])))
        opt.step()
        self.assertTrue(np.all(np.isfinite(opt['primitive_center_y'].numpy())))

    def test_invalid_density_is_not_treated_as_low_contribution(self):
        # non-finite or invalid geometry must not silently qualify for low-density pruning.
        columns = weak_columns()
        columns.parameters['primitive_scale_y'] = mi.Float([0.0, 0.1, np.nan, 0.05])
        self.assertEqual(prune(columns, min_low_contribution_checks=1), 0)

if __name__ == '__main__':
    unittest.main()
