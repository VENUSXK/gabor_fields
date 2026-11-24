"""Check compact FGC assets, legacy formats, and per-Gaussian rendering."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mitsuba as mi
import plyfile
import gfields
from gfields.integrators.common import FactorizedGaussianColumn
from gfields.tests.test_factorized_gaussian_column import make_parameters
from gfields.training.regression import create_scene_dict_from_primitives, save_model
from gfields.io import load_ellipsoid_ply, ellipsoid_dict_to_ply
from gfields.rendering.pipeline import detect_asset_info, load_pyramid_level


def save_factors(directory, zero_weights=False):
    # use nonuniform factors with nonzero rotations and compact world-space geometry.
    parameters = make_parameters()
    parameters['column_center_xz'] = mi.Point2f([-0.3, 0.2, -0.1, 0.4], [-0.2, 0.1, 0.3, -0.1])
    parameters['column_quaternions'] = mi.Float([0.2, -0.5, 0.7, 0.4])
    parameters['primitive_center_y'] = mi.Float([-0.2, 0, 0.2, -0.1, 0.1, 0.3])
    if zero_weights:
        parameters['xz_logits'] = mi.Float([0, -np.inf, -np.inf, 0])
        parameters['y_logits'] = mi.Float([0, -np.inf, -np.inf, -np.inf, 0, -np.inf])
        parameters['group_opacity'] = mi.Float([0, 0.8])
    factors = FactorizedGaussianColumn(parameters, 2, 3)
    factors.init_albedo = 0.4
    scene_dict = create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=factors, extent=2.5)
    save_model(Path(directory), mi.load_dict(scene_dict), 0, columns=factors, use_factorized_columns_representation=True)
    asset = Path(directory) / 'optimized_asset_pyr0'
    return factors, scene_dict, asset, asset / 'data' / 'root.primitives_pyr0.ply'


class FactorizedColumnPlyTests(unittest.TestCase):
    def test_compact_tables_and_render_round_trip(self):
        # store R + R*J + R*K factors, with no expanded geometry or logits in the file.
        with tempfile.TemporaryDirectory() as directory:
            factors, _, asset, path = save_factors(directory)
            ply = plyfile.PlyData.read(path, mmap=False)
            self.assertEqual([element.name for element in ply.elements], ['metadata', 'group', 'xz', 'y'])
            self.assertEqual([len(ply[name].data) for name in ('metadata', 'group', 'xz', 'y')], [1, 2, 4, 6])
            self.assertEqual(int(ply['metadata'].data['format_version'][0]), 3)
            self.assertFalse(list(asset.rglob('*.npy')))
            loaded = load_ellipsoid_ply(path, use_factorized_columns_representation=True)
            np.testing.assert_allclose(loaded['xz_weight'].reshape(2, 2).sum(axis=1), 1)
            np.testing.assert_allclose(loaded['y_weight'].reshape(2, 3).sum(axis=1), 1)
            np.testing.assert_allclose(loaded['group_opacity'], [0.4, 0.8])

            # preserve factor dimensions while counting expanded render primitives.
            info = detect_asset_info(asset, use_factorized_columns_representation=True)
            self.assertEqual(info.factorized_group_shapes[0], (2, 2, 3))
            self.assertEqual(info.num_primitives[0], 12)
            geometry = load_pyramid_level(info, 0, extent=info.extents[0], use_asset_albedo=True)
            self.assertEqual(geometry['type'], 'ellipsoidsmesh')
            self.assertNotIn('primitives_per_column', geometry)
            self.assertEqual(geometry['extent'], 2.5)
            centers, scales, quaternions, weights = factors.expand()
            for name, expected in (('centers', centers.numpy().T), ('scales', scales.numpy().T),
                                   ('quaternions', quaternions.numpy().T), ('opacities', weights.numpy().reshape(-1, 1))):
                np.testing.assert_allclose(np.asarray(geometry[name]), expected, rtol=2e-6, atol=1e-7)
            np.testing.assert_allclose(np.asarray(geometry['albedo']), 0.4)
            scene = mi.load_dict({'type': 'scene', 'primitives': geometry})
            self.assertEqual(mi.traverse(scene)['primitives.opacities'].shape, (12,))

    def test_common_transforms_and_arbitrary_subset(self):
        # apply existing world, opacity, albedo and subset processing after expansion.
        with tempfile.TemporaryDirectory() as directory:
            factors, _, asset, _ = save_factors(directory)
            info = detect_asset_info(asset, use_factorized_columns_representation=True)
            geometry = load_pyramid_level(info, 0, primitive_subset=slice(1, 7), world_scale=2,
                                           opacity_scale=3, albedo_rgb=[0.2, 0.4, 0.6], albedo_scale=0.5)
            centers, scales, _, weights = factors.expand()
            np.testing.assert_allclose(np.asarray(geometry['centers']), centers.numpy().T[1:7] * 2)
            np.testing.assert_allclose(np.asarray(geometry['scales']), scales.numpy().T[1:7] * 2)
            np.testing.assert_allclose(np.asarray(geometry['opacities']).ravel(), weights.numpy()[1:7] * 3, rtol=2e-6)
            np.testing.assert_allclose(np.asarray(geometry['albedo']), np.tile([0.1, 0.2, 0.3], (6, 1)))

    @unittest.skipUnless(mi.variant().startswith('cuda'), 'FGC BVH requires CUDA')
    def test_grouped_render_preserves_factor_order_and_transmission(self):
        # compare an fgc asset through the default and grouped render paths.
        with tempfile.TemporaryDirectory() as directory:
            factors, original, asset, _ = save_factors(directory)
            default = detect_asset_info(asset, use_factorized_columns_representation=True)
            grouped = detect_asset_info(asset, use_factorized_columns_representation=True, use_fgc_bvh=True)
            original['sensor'] = {'type': 'orthographic', 'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, -3], target=[0, 0, 0], up=[0, 1, 0]),
                'film': {'type': 'hdrfilm', 'width': 8, 'height': 8, 'rfilter': {'type': 'box'}}}

            # check the compact buffers and the expanded integration attributes.
            baseline = load_pyramid_level(default, 0, extent=2.5, world_scale=2)
            geometry = load_pyramid_level(grouped, 0, extent=2.5, world_scale=2)
            self.assertEqual((geometry['type'], geometry['primitives_per_column']), ('ellipsoids', 6))
            self.assertEqual((geometry['fgc_xz_count'], geometry['fgc_y_count']), (2, 3))
            self.assertEqual((geometry['fgc_xz'].shape, geometry['fgc_y'].shape), ((4, 5), (6, 3)))
            y_factors = np.asarray(geometry['fgc_y'])
            np.testing.assert_allclose(y_factors[:, 0], factors.parameters['primitive_center_y'].numpy() * 2)
            np.testing.assert_allclose(y_factors[:, 1], factors.parameters['primitive_scale_y'].numpy() * 2)
            y_logits = factors.parameters['y_logits'].numpy().reshape(2, 3)
            weights = np.exp(y_logits)
            np.testing.assert_allclose(y_factors[:, 2].reshape(2, 3), weights / weights.sum(axis=1, keepdims=True))
            np.testing.assert_allclose(np.asarray(geometry['opacities']), np.asarray(baseline['opacities']))
            second_group = load_pyramid_level(grouped, 0, world_scale=2, primitive_subset=slice(6, 12))
            np.testing.assert_allclose(np.asarray(second_group['fgc_xz']), np.asarray(geometry['fgc_xz'])[2:])
            np.testing.assert_allclose(np.asarray(second_group['fgc_y']), y_factors[3:])

            # compare the same analytic support with one bound per primitive or group.
            baseline['type'] = 'ellipsoids'
            baseline.pop('shell')
            scenes = [mi.load_dict(dict(original, primitives_pyr0=entry)) for entry in (baseline, geometry)]
            self.assertEqual([scene.shapes()[0].primitive_count() for scene in scenes], [12, 2])
            images = [np.asarray(mi.render(scene, seed=0, spp=1)) for scene in scenes]
            np.testing.assert_allclose(images[1], images[0], rtol=4e-5, atol=4e-6)

    @unittest.skipUnless(mi.variant().startswith('cuda'), 'FGC BVH requires CUDA')
    def test_grouped_free_flight_matches_expanded_gaussians(self):
        # compare identical expanded primitive ids with one analytic bound per primitive or group.
        with tempfile.TemporaryDirectory() as directory:
            _, _, asset, _ = save_factors(directory)
            options = {'use_factorized_columns_representation': True}
            info = detect_asset_info(asset, **options)
            grouped = detect_asset_info(asset, use_fgc_bvh=True, **options)
            extents = np.linspace(2.4, 3.6, info.num_primitives[0], dtype=np.float32)
            baseline, optimized = (load_pyramid_level(layout, 0, extent=extents) for layout in (info, grouped))
            baseline['type'] = 'ellipsoids'
            baseline.pop('shell')
            scenes = [mi.load_dict({'type': 'scene', 'primitives': geometry}) for geometry in (baseline, optimized)]

            # cover vertical, oblique, side, interior, restarted and clipped rays.
            seeds = mi.UInt32(np.arange(256, dtype=np.uint32) + 17)
            cases = (
                ((-0.3, -2, -0.2), (0, 1, 0), 10., 0., True),
                ((-0.48, -2, -0.11), (0.10, 1, -0.05), 10., 0., True),
                ((0.2, 0.2, -2), (0, 0, 1), 10., 0., True),
                ((-0.3, -0.2, -0.2), (0, 1, 0), 10., 0., True),
                ((-0.3, -2, -0.2), (0, 1, 0), 10., 2.02, True),
                ((-0.3, -2, -0.2), (0, 1, 0), 0.5, 0., False),
                ((2, -2, 0), (0, 1, 0), 10., 0., False),
            )
            for origin, direction, maxt, restart, has_events in cases:
                with self.subTest(origin=origin, direction=direction, maxt=maxt, restart=restart):
                    direction = np.asarray(direction, dtype=np.float32) / np.linalg.norm(direction)
                    ray = mi.Ray3f(mi.Point3f(*origin), mi.Vector3f(*direction.tolist()))
                    ray.maxt = maxt
                    expected, actual = (scene.ray_sample_ellipsoids(ray, seeds, mi.Float(restart)) for scene in scenes)
                    valid = np.isfinite(expected[0].numpy())
                    self.assertEqual(bool(valid.any()), has_events)
                    np.testing.assert_allclose(actual[0].numpy(), expected[0].numpy(), rtol=4e-5, atol=4e-6)
                    np.testing.assert_array_equal(actual[2].numpy()[valid], expected[2].numpy()[valid])

    def test_zero_weights(self):
        # represent exact zero factors with negative infinity logits when loading.
        with tempfile.TemporaryDirectory() as directory:
            factors, _, asset, _ = save_factors(directory, zero_weights=True)
            info = detect_asset_info(asset, use_factorized_columns_representation=True)
            geometry = load_pyramid_level(info, 0)
            weights = np.asarray(geometry['opacities']).ravel()
            self.assertTrue(np.isfinite(weights).all())
            np.testing.assert_allclose(weights, factors.expand()[3].numpy())

    def test_render_matches_expanded_scene(self):
        # compare actual transmission renders of the original and reloaded Gaussian mixtures.
        with tempfile.TemporaryDirectory() as directory:
            _, original, asset, _ = save_factors(directory)
            info = detect_asset_info(asset, use_factorized_columns_representation=True)
            restored = dict(original)
            restored['primitives_pyr0'] = load_pyramid_level(info, 0, extent=2.5, use_asset_albedo=True)
            sensor = {'type': 'orthographic', 'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, -3], target=[0, 0, 0], up=[0, 1, 0]),
                'film': {'type': 'hdrfilm', 'width': 8, 'height': 8, 'rfilter': {'type': 'box'}},
                'sampler': {'type': 'independent', 'sample_count': 1}}
            original['sensor'] = restored['sensor'] = sensor
            images = [np.asarray(mi.render(mi.load_dict(scene), seed=0, spp=1)) for scene in (original, restored)]
            self.assertGreater(float(np.ptp(images[0])), 0.01)
            np.testing.assert_allclose(images[1], images[0], rtol=2e-5, atol=2e-6)

    def test_corrupt_weights_and_wrong_flags(self):
        # reject invalid physical factors and mismatched representation requests.
        with tempfile.TemporaryDirectory() as directory:
            _, _, asset, path = save_factors(directory)
            with self.assertRaisesRegex(ValueError, 'vertex'):
                load_ellipsoid_ply(path)
            with self.assertRaisesRegex(ValueError, 'Choose either'):
                detect_asset_info(asset, use_columns_representation=True, use_factorized_columns_representation=True)
            ply = plyfile.PlyData.read(path, mmap=False)
            ply['xz'].data['weight'][0] = 4
            ply.write(path)
            with self.assertRaisesRegex(ValueError, 'normalized'):
                load_ellipsoid_ply(path, use_factorized_columns_representation=True)

    def test_ordinary_gaussian_format(self):
        # keep the original vertex writer, decoder and geometry selection unchanged.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'data'
            folder.mkdir()
            path = folder / 'root.primitives_pyr0.ply'
            data = {'centers': np.zeros((1, 3)), 'scales': np.full((1, 3), 0.2),
                    'quaternions': np.array([[0, 0, 0, 1]]), 'opacities': np.array([[0.3]])}
            ellipsoid_dict_to_ply(data, ['opacities'], path)
            loaded = load_ellipsoid_ply(path)
            np.testing.assert_allclose(loaded['scales'], 0.2)
            geometry = load_pyramid_level(detect_asset_info(directory), 0)
            self.assertEqual(geometry['type'], 'ellipsoidsmesh')
            np.testing.assert_allclose(np.asarray(geometry['opacities']), 0.3)

    def test_render_cli_flag(self):
        from gfields.render import parse_args

        # expose compact FGC loading through the standard rendering entry point.
        with patch('sys.argv', ['gfields.render', 'example', '--use_factorized_columns_representation']):
            self.assertTrue(parse_args().use_factorized_columns_representation)


if __name__ == '__main__':
    unittest.main()
