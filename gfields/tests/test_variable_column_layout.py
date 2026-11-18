"""Check variable column layouts, parameter updates, and shared gradients."""
import unittest
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import GaussianColumn


def make_parameters(primitive_count):
    # create distinct footprints and independent axial parameters.
    return {
        'column_center_xz': mi.Point2f([1.0, 2.0], [3.0, 4.0]),
        'column_scale_xz': mi.Vector2f([0.1, 0.2], [0.3, 0.4]),
        'column_quaternions': mi.Float([0.0, 0.5]),
        'primitive_center_y': mi.Float(np.linspace(-0.5, 0.5, primitive_count)),
        'primitive_scale_y': mi.Float(np.full(primitive_count, 0.1)),
        'primitive_opacity': mi.Float(np.full(primitive_count, 0.2)),
    }


class VariableColumnLayoutTests(unittest.TestCase):
    def test_fixed_layout(self):
        # preserve existing uniform initialization and expansion.
        columns = GaussianColumn(make_parameters(4), primitives_per_column=2)
        np.testing.assert_array_equal(columns.primitive_count_per_column.numpy(), [2, 2])
        np.testing.assert_array_equal(columns.column_offsets.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(columns.primitive_indices.numpy(), [0, 0, 1, 1])
        np.testing.assert_allclose(columns.expand()[0][0].numpy(), [1, 1, 2, 2])

    def test_variable_layout_and_gradients(self):
        # expand unequal groups and accumulate gradients into each footprint.
        parameters = make_parameters(7)
        dr.enable_grad(parameters['column_center_xz'])
        columns = GaussianColumn(parameters, 2, primitive_count_per_column=[2, 5])
        centers, scales, quaternions, opacity = columns.expand()
        np.testing.assert_array_equal(columns.column_offsets.numpy(), [0, 2, 7])
        np.testing.assert_array_equal(columns.primitive_indices.numpy(), [0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(centers[0].numpy(), [1, 1, 2, 2, 2, 2, 2])
        np.testing.assert_allclose(centers[2].numpy(), [3, 3, 4, 4, 4, 4, 4])
        np.testing.assert_allclose(scales[0].numpy(), [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(centers[1].numpy(), parameters['primitive_center_y'].numpy())
        np.testing.assert_allclose(opacity.numpy(), parameters['primitive_opacity'].numpy())
        self.assertEqual(dr.width(quaternions), 7)
        dr.backward(dr.sum(centers[0]))
        np.testing.assert_allclose(dr.grad(parameters['column_center_xz'])[0].numpy(), [2, 5])

    def test_update_layout(self):
        # refresh indices after independent parameter arrays change length.
        parameters = make_parameters(4)
        columns = GaussianColumn(parameters, 2)
        updated_parameters = make_parameters(5)
        for name in ('primitive_center_y', 'primitive_scale_y', 'primitive_opacity'):
            parameters[name] = updated_parameters[name]
        with self.assertRaises(ValueError):
            columns.expand()
        columns.set_primitive_count_per_column([2, 3])
        np.testing.assert_array_equal(columns.column_offsets.numpy(), [0, 2, 5])
        np.testing.assert_allclose(columns.expand()[0][0].numpy(), [1, 1, 2, 2, 2])

    def test_scene_parameter_round_trip(self):
        from gfields.training.regression import write_optimizer_params_to_scene, write_scene_params_to_optimizer

        # reuse scene writing and recover shared footprints at column offsets.
        columns = GaussianColumn(make_parameters(7), 2, primitive_count_per_column=[2, 5])
        expected_centers = columns.expand()[0].numpy().copy()
        scene_params = {}
        write_optimizer_params_to_scene(None, scene_params, 0, use_columns_representation=True, columns=columns)
        opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
        write_scene_params_to_optimizer(scene_params, opt, 0, use_columns_representation=True, columns=columns)
        np.testing.assert_allclose(opt['column_center_xz'].numpy(), [[1, 2], [3, 4]])
        np.testing.assert_allclose(columns.expand()[0].numpy(), expected_centers)

    def test_variable_layout_histogram(self):
        from gfields.training.callbacks import PlotCallback

        # use the actual parent footprint for every primitive density.
        columns = GaussianColumn(make_parameters(7), 2, primitive_count_per_column=[2, 5])
        before, after = PlotCallback(columns)._column_histogram_data()
        np.testing.assert_allclose(before['density'], after['density'], rtol=1e-6)
        self.assertEqual(len(before['density']), 7)

    def test_invalid_counts(self):
        # reject invalid counts before they can produce out-of-bounds gathers.
        for counts in ([1], [0, 4], [-1, 5], [2.5, 1.5], [True, True], [2, 3], [[2, 2]]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                GaussianColumn(make_parameters(4), 2, primitive_count_per_column=counts)


if __name__ == '__main__':
    unittest.main()
