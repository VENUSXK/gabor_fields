"""Check grouped analytic traversal against the same independent Gaussians."""
import unittest
import tempfile
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import gfields

from gfields.integrators.common import GaussianColumn
from gfields.training.regression import create_scene_dict_from_primitives, write_scene_params_to_optimizer, write_optimizer_params_to_scene


def make_columns():
    # use separated axial supports and rotated lateral footprints.
    parameters = {'column_center_xz': mi.Point2f(mi.Float([-0.35, 0.35]), mi.Float([0.0, 0.0])),
                  'column_scale_xz': mi.Vector2f(dr.full(mi.Float, 0.12, 2), dr.full(mi.Float, 0.16, 2)),
                  'column_quaternions': mi.Float([0.2, -0.3]), 'primitive_center_y': mi.Float([-0.45, 0.45, -0.45, 0.45]),
                  'primitive_scale_y': mi.Float([0.09, 0.10, 0.09, 0.10]), 'primitive_opacity': mi.Float([0.01, 0.02, 0.015, 0.025])}
    columns = GaussianColumn(parameters, primitives_per_column=2)
    columns.init_albedo = 0.0
    return columns


def make_scene(grouped, anyhit):
    # compare analytic geometry with identical finite support in both cases.
    columns = make_columns()
    scene_dict = create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns, use_column_bvh=grouped, use_anyhit=anyhit)
    primitive = scene_dict['primitives_pyr0']
    primitive['type'] = 'ellipsoids'
    primitive.pop('shell', None)
    scene_dict['sensor'] = {'type': 'orthographic', 'to_world': mi.ScalarTransform4f.look_at(origin=[0, 0, -3], target=[0, 0, 0], up=[0, 1, 0]), 'film': {'type': 'hdrfilm', 'width': 16, 'height': 16, 'rfilter': {'type': 'box'}}, 'sampler': {'type': 'independent', 'sample_count': 2}}

    # bind shared optimizer parameters through the existing scene interface.
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)
    opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
    write_scene_params_to_optimizer(params, opt, 0, use_columns_representation=True, columns=columns)
    write_optimizer_params_to_scene(opt, params, 0, use_columns_representation=True, columns=columns)
    return scene, params, opt, columns


class ColumnBvhConfigurationTests(unittest.TestCase):
    def test_factory_preserves_parameters_and_selects_grouped_geometry(self):
        # keep the mesh default unchanged and group only on explicit request.
        columns = make_columns()
        mesh = create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns)
        grouped = create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns, use_column_bvh=True)
        self.assertEqual(mesh['primitives_pyr0']['type'], 'ellipsoidsmesh')
        self.assertEqual(grouped['primitives_pyr0']['type'], 'ellipsoids')
        self.assertEqual(grouped['primitives_pyr0']['primitives_per_column'], 2)
        self.assertNotIn('shell', grouped['primitives_pyr0'])
        for name in ('centers', 'scales', 'quaternions', 'opacities', 'albedo', 'omega'):
            np.testing.assert_array_equal(mesh['primitives_pyr0'][name].numpy(), grouped['primitives_pyr0'][name].numpy())

    def test_factory_rejects_missing_column_layout(self):
        # grouped ids must never be inferred from independent primitive arrays.
        with self.assertRaisesRegex(ValueError, 'shared column parameters'):
            create_scene_dict_from_primitives(-1, 0, use_column_bvh=True)


@unittest.skipUnless(mi.variant().startswith('cuda'), 'Grouped column acceleration requires the CUDA backend')
class ColumnBvhCudaTests(unittest.TestCase):
    def test_small_scale_supports_at_training_primitive_count(self):
        from gfields.training.configs import GenericConfig
        from gfields.training.regression import generate_random_columns

        # reproduce small supports where inaccurate quadratic shifts overcount.
        volume = np.zeros((64, 64, 64), dtype=np.float32)
        volume[:, 28:36, :] = 0.4
        columns = generate_random_columns(6144, 4, 0.00006, 1.0, 0.0, volume=volume, volume_config=GenericConfig(), column_slab_depth=8)
        scenes = []
        for grouped in (False, True):
            scene_dict = create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns, use_column_bvh=grouped)
            scene_dict['primitives_pyr0']['type'] = 'ellipsoids'
            scene_dict['primitives_pyr0'].pop('shell', None)
            scenes.append(mi.load_dict(scene_dict))

        # compare identical random rays across full and grouped acceleration.
        rng = np.random.default_rng(17)
        ray = mi.Ray3f(mi.Point3f(mi.Float(rng.uniform(-1, 1, 2048)), mi.Float(rng.uniform(-0.2, 0.2, 2048)), -3), mi.Vector3f(0, 0, 1))
        flags = int(mi.RayFlags.TransmittanceEllipsoids)
        values = [scene.ray_transmittance_ellipsoids(ray, mi.UInt32(0), flags, mi.Bool(True)).numpy() for scene in scenes]
        np.testing.assert_allclose(values[0], values[1], rtol=2e-5, atol=2e-6)
        self.assertEqual(scenes[0].shapes()[0].primitive_count(), 24576)
        self.assertEqual(scenes[1].shapes()[0].primitive_count(), 6144)

    def test_hit_ids_inside_origins_and_finite_segments(self):
        # rays cover each child, a miss, an interior origin, and a short segment.
        baseline = make_scene(False, False)[0]
        grouped = make_scene(True, False)[0]
        self.assertEqual(baseline.shapes()[0].primitive_count(), 4)
        self.assertEqual(grouped.shapes()[0].primitive_count(), 2)
        ray = mi.Ray3f(mi.Point3f(mi.Float([-0.35, -0.35, 0.35, 0.35, 2, -0.35, -0.35, -0.35]), mi.Float([-0.45, 0.45, -0.45, 0.45, 0, -0.45, -0.45, 0]), mi.Float([-3, -3, -3, -3, -3, 0, -3, -3])), mi.Vector3f(0, 0, 1))
        ray.maxt = mi.Float([10, 10, 10, 10, 10, 10, 1, 10])
        expected = baseline.ray_intersect(ray)
        actual = grouped.ray_intersect(ray)
        np.testing.assert_array_equal(actual.is_valid().numpy(), expected.is_valid().numpy())
        valid = expected.is_valid().numpy()
        np.testing.assert_array_equal(actual.prim_index.numpy()[valid], expected.prim_index.numpy()[valid])
        np.testing.assert_allclose(actual.t.numpy()[valid], expected.t.numpy()[valid], rtol=2e-5, atol=1e-6)
        np.testing.assert_array_equal(actual.prim_index.numpy()[:4], [0, 1, 2, 3])
        np.testing.assert_array_equal(grouped.ray_test(ray).numpy(), baseline.ray_test(ray).numpy())
        self.assertFalse(bool(actual.is_valid()[-1]))

        # dedicated traversal must preserve the same clipping and interior rules.
        flags = int(mi.RayFlags.TransmittanceEllipsoids)
        expected_tau = baseline.ray_transmittance_ellipsoids(ray, mi.UInt32(0), flags, mi.Bool(True))
        actual_tau = grouped.ray_transmittance_ellipsoids(ray, mi.UInt32(0), flags, mi.Bool(True))
        np.testing.assert_allclose(actual_tau.numpy(), expected_tau.numpy(), rtol=2e-5, atol=1e-6)
        self.assertEqual(float(actual_tau[-1]), 0.0)

        # the same column must remain discoverable after its first child hit.
        ray = mi.Ray3f(mi.Point3f(-0.35, -2, 0), mi.Vector3f(0, 1, 0))
        for primitive_index in (0, 1):
            hit = grouped.ray_intersect(ray)
            self.assertTrue(bool(hit.is_valid()[0]))
            self.assertEqual(int(hit.prim_index[0]), primitive_index)
            ray.o = hit.p + ray.d * 5e-6

    def test_stochastic_transmittance_and_free_flight_queries(self):
        # stateless seeds must identify the same child irrespective of grouping.
        baseline = make_scene(False, True)[0]
        grouped = make_scene(True, True)[0]
        seeds = dr.arange(mi.UInt32, 128) + 17
        ray = mi.Ray3f(mi.Point3f(dr.full(mi.Float, -0.35, 128), dr.full(mi.Float, -2.0, 128), 0), mi.Vector3f(0, 1, 0))
        flags = int(mi.RayFlags.StochasticEllipsoids)
        expected = baseline.ray_transmittance_ellipsoids(ray, seeds, flags)
        actual = grouped.ray_transmittance_ellipsoids(ray, seeds, flags)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        self.assertEqual(set(np.unique(actual.numpy())), {0.0, 1.0})

        # cover regular, restarted, interior, clipped, and empty free flights.
        for origin, maxt, restart in (([-0.35, -2, 0], 10, 0), ([-0.35, -2, 0], 10, 1.65), ([-0.35, -0.45, 0], 10, 0), ([-0.35, -2, 0], 0.1, 0), ([2, -2, 0], 10, 0)):
            ray = mi.Ray3f(mi.Point3f(*origin), mi.Vector3f(0, 1, 0))
            ray.maxt = maxt
            expected = baseline.ray_sample_ellipsoids(ray, seeds, mi.Float(restart))
            actual = grouped.ray_sample_ellipsoids(ray, seeds, mi.Float(restart))
            expected_t, expected_shape, expected_primitive, expected_surface = expected
            actual_t, actual_shape, actual_primitive, actual_surface = actual
            np.testing.assert_allclose(actual_t.numpy(), expected_t.numpy(), rtol=2e-5, atol=2e-6)
            valid = np.isfinite(expected_t.numpy())
            np.testing.assert_array_equal(actual_primitive.numpy()[valid], expected_primitive.numpy()[valid])
            np.testing.assert_array_equal(actual_surface.is_valid().numpy(), expected_surface.is_valid().numpy())
            self.assertTrue(np.all(actual_shape.numpy()[valid] != 0))
            if origin[0] == 2 or maxt == 0.1:
                self.assertFalse(valid.any())

    def test_covering_queries_preserve_overlapping_children(self):
        # overlap the two children so collection must append both ids once.
        scenes = [make_scene(False, True), make_scene(True, True)]
        for scene, params, opt, columns in scenes:
            opt['primitive_center_y'] = mi.Float([0.0, 0.1, 0.0, 0.1])
            opt['primitive_scale_y'] = dr.full(mi.Float, 0.1, 4)
            write_optimizer_params_to_scene(opt, params, 0, use_columns_representation=True, columns=columns)
        results = []
        for scene, params, opt, columns in scenes:
            capacity, lanes = 4, 4
            buffer = dr.empty(mi.UInt32, 2 * capacity * lanes)
            dr.eval(buffer)
            slots = dr.arange(mi.UInt32, lanes)
            addresses = mi.UInt64(mi.device_address(buffer)) + mi.UInt64(slots) * (2 * capacity * 4)

            # materialize the query before reading device-written slots.
            ray = mi.Ray3f(mi.Point3f(mi.Float([-0.35, 0.35, 2, -0.35]), mi.Float([0.05, 0.05, 0, 0.05]), 0), mi.Vector3f(0, 1, 0))
            ray.maxt = 1.0
            count = scene.ray_collect_ellipsoids(ray, addresses, capacity, mi.Bool([True, True, True, False]))
            dr.eval(count)
            dr.sync_thread()
            offsets = slots * (2 * capacity) + (count & mi.UInt32(0))
            ids = np.stack([dr.gather(mi.UInt32, buffer, offsets + 2 * k + 1).numpy() for k in range(capacity)], axis=1)
            counts = count.numpy()
            results.append([set(ids[lane, :counts[lane]].tolist()) for lane in range(lanes)])
            np.testing.assert_array_equal(counts, [2, 2, 0, 0])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], [{0, 1}, {2, 3}, set(), set()])

    def test_checkpoint_keeps_every_expanded_gaussian(self):
        # grouped acceleration must not truncate the saved primitive arrays.
        from gfields.training.regression import save_model
        scene, params, opt, columns = make_scene(True, True)
        root = Path(__file__).resolve().parents[2] / 'results' / 'column_bvh_checks'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            save_model(Path(directory), scene, 0)
            path = Path(directory) / 'optimized_asset_pyr0' / 'data' / 'root.primitives_pyr0.ply'
            arrays = gfields.io.load_ellipsoid_ply(path)
            self.assertEqual(len(arrays['centers']), 4)
            self.assertEqual(len(arrays['opacities']), 4)
            np.testing.assert_allclose(arrays['centers'], columns.expand()[0].numpy().T, atol=1e-6)
            np.testing.assert_allclose(np.asarray(arrays['opacities']).reshape(-1), opt['primitive_opacity'].numpy(), atol=1e-6)

    def test_render_and_adjoint_match_before_and_after_optimizer_updates(self):
        for anyhit in (False, True):
            with self.subTest(anyhit=anyhit):
                baseline = make_scene(False, anyhit)
                grouped = make_scene(True, anyhit)
                for iteration in range(2):
                    images, gradients = [], []
                    for scene, params, opt, columns in (baseline, grouped):
                        # compare both primal images and all shared parameter gradients.
                        dr.clear_grad(opt.variables)
                        image = mi.render(scene, params, spp=2, seed=31 + iteration)
                        weights = mi.Float(np.linspace(0.5, 1.5, dr.width(image.array), dtype=np.float32))
                        dr.backward(dr.mean(dr.square(image.array) * weights))
                        images.append(image.numpy())
                        gradients.append({name: dr.grad(opt[name]).numpy() for name in columns.parameter_names})
                    np.testing.assert_allclose(images[0], images[1], rtol=5e-5, atol=2e-6)
                    for name in baseline[3].parameter_names:
                        self.assertTrue(np.isfinite(gradients[1][name]).all())
                        np.testing.assert_allclose(gradients[0][name], gradients[1][name], rtol=2e-3, atol=2e-5, err_msg=name)
                    self.assertGreater(float(np.abs(gradients[1]['primitive_opacity']).sum()), 0.0)

                    # move the footprint beyond its old bound and rebuild after a step.
                    for scene, params, opt, columns in (baseline, grouped):
                        opt.step()
                        opt['column_center_xz'] = mi.Point2f(opt['column_center_xz'] + mi.Vector2f(1.2, 0.0))
                        write_optimizer_params_to_scene(opt, params, 0, use_columns_representation=True, columns=columns)
                    center_x = grouped[2]['column_center_xz'][0][0]
                    center_y = grouped[2]['primitive_center_y'][0]
                    ray = mi.Ray3f(mi.Point3f(center_x, center_y, -3), mi.Vector3f(0, 0, 1))
                    expected = baseline[0].ray_intersect(ray)
                    actual = grouped[0].ray_intersect(ray)
                    self.assertTrue(bool(actual.is_valid()[0]))
                    self.assertEqual(int(actual.prim_index[0]), int(expected.prim_index[0]))


if __name__ == '__main__':
    unittest.main()
