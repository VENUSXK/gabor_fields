"""Exercise adaptive column steps with real rendering and model export."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import drjit as dr
import mitsuba as mi
from gfields.tests.test_column_pruning import weak_columns
from gfields.training.regression import (create_scene_dict_from_primitives, accumulate_column_y_gradients,
    adapt_column_primitives, write_optimizer_params_to_scene, save_model)
from gfields.io import load_ellipsoid_ply


class AdaptiveColumnTrainingTests(unittest.TestCase):
    def test_resize_render_and_export(self):
        # build a real primitive mesh scene with nonzero constant albedo.
        columns = weak_columns()
        columns.init_albedo = 0.37
        opt = columns.parameters
        opt.set_learning_rate({'primitive_opacity': 0.0})
        scene_dict = create_scene_dict_from_primitives(-1, 0, use_anyhit=True, use_columns_representation=True, columns=columns)
        scene_dict['sensor'] = {'type': 'orthographic', 'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, -4], target=[0, 0, 0], up=[0, 1, 0]),
                                'film': {'type': 'hdrfilm', 'width': 8, 'height': 8, 'rfilter': {'type': 'box'}},
                                'sampler': {'type': 'independent', 'sample_count': 2}}
        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)
        write_optimizer_params_to_scene(opt, params, 0, use_columns_representation=True, columns=columns)
        args = SimpleNamespace(use_adaptive_columns=True, column_adapt_from=4, column_adapt_until=8,
                               column_adapt_every=4, column_prune_weight_threshold=0.01,
                               column_prune_density_threshold=0.01, column_prune_checks=1,
                               column_protection_checks=2, column_split_grad_threshold=1.0,
                               column_max_primitives_per_column=4, column_max_primitives=8, column_max_splits=2)

        # render and backpropagate after each resize; use controlled Y gradients for deterministic decisions.
        results = []
        for iteration in range(1, 10):
            dr.clear_grad(opt.variables)
            image = mi.render(scene, params, spp=2, seed=iteration)
            dr.backward(dr.mean(dr.square(image.array)))
            self.assertTrue(np.all(np.isfinite(dr.grad(opt['primitive_opacity']).numpy())))
            if iteration <= args.column_adapt_until:
                dr.set_grad(opt['primitive_center_y'], dr.full(mi.Float, 4.0, dr.width(opt['primitive_center_y'])))
                accumulate_column_y_gradients(columns)
            opt.step()
            shared_state = {name: opt.state[name] for name in columns.parameter_names[:3]}
            results.append(adapt_column_primitives(columns, args, iteration))
            for name, state in shared_state.items():
                self.assertIs(opt.state[name], state)
            write_optimizer_params_to_scene(opt, params, 0, use_columns_representation=True, columns=columns)

            # all scene arrays must match the live primitive count before the next render.
            count = dr.width(opt['primitive_center_y'])
            self.assertEqual(dr.width(params['primitives_pyr0.data']), 10 * count)
            self.assertEqual(dr.width(params['primitives_pyr0.opacities']), count)
            self.assertEqual(dr.width(params['primitives_pyr0.omega']), count)
            self.assertEqual(dr.width(params['primitives_pyr0.albedo']), 3 * count)
            np.testing.assert_allclose(params['primitives_pyr0.albedo'].numpy(), 0.37)
        self.assertEqual(results[3], {'pruned': 2, 'split': 2})
        self.assertEqual(results[7], {'pruned': 0, 'split': 2})
        self.assertEqual(count, 6)

        # export changed layouts as compact columns and preserve per-column primitive counts.
        with tempfile.TemporaryDirectory() as directory:
            save_model(Path(directory), scene, 0, use_columns_representation=True, columns=columns)
            files = list(Path(directory).rglob('*.ply'))
            self.assertEqual(len(files), 1)
            saved = load_ellipsoid_ply(files[0], use_columns_representation=True)
            self.assertEqual(len(saved['primitive_opacity']), count)
            np.testing.assert_array_equal(saved['primitive_count_per_column'], columns.primitive_count_per_column.numpy())

    def test_disabled_adaptation_is_a_noop(self):
        # ordinary training must not require any column-specific settings or parameters.
        self.assertEqual(adapt_column_primitives(None, SimpleNamespace(use_adaptive_columns=False), 1), {'pruned': 0, 'split': 0})


if __name__ == '__main__':
    unittest.main()
