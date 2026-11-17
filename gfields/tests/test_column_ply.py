"""Check compact variable-count PLY, legacy compatibility, and render geometry selection."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mitsuba as mi
import plyfile
from gfields.integrators.common import GaussianColumn
from gfields.tests.test_variable_column_layout import make_parameters
from gfields.training.regression import create_scene_dict_from_primitives, save_model
from gfields.io import load_ellipsoid_ply
from gfields.rendering.pipeline import detect_asset_info, load_pyramid_level


def save_columns(directory, counts):
    # save real scene parameters through the common asset exporter.
    columns = GaussianColumn(make_parameters(sum(counts)), 2, primitive_count_per_column=counts)
    columns.init_albedo = 0.4
    scene = mi.load_dict(create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns))
    save_model(Path(directory), scene, 0, use_columns_representation=True, columns=columns)
    asset = Path(directory) / 'optimized_asset_pyr0'
    return columns, asset, next(asset.rglob('*.ply'))


class ColumnPlyTests(unittest.TestCase):
    def test_variable_counts_and_parameter_round_trip(self):
        # preserve compact fields, ordering, and every original physical parameter.
        with tempfile.TemporaryDirectory() as directory:
            columns, asset, path = save_columns(directory, [2, 5])
            ply = plyfile.PlyData.read(path, mmap=False)
            self.assertEqual(int(ply['metadata'].data['format_version'][0]), 2)
            np.testing.assert_array_equal(ply['column'].data['primitive_count'], [2, 5])
            self.assertEqual(len(ply['primitive'].data), 7)
            loaded = load_ellipsoid_ply(path, use_columns_representation=True)
            np.testing.assert_array_equal(loaded['primitive_count_per_column'], [2, 5])
            parameters = {name: mi.Float(loaded[name]) for name in columns.parameter_names[2:]}
            parameters['column_center_xz'] = mi.Point2f(loaded['column_center_xz'].T)
            parameters['column_scale_xz'] = mi.Vector2f(loaded['column_scale_xz'].T)
            restored = GaussianColumn(parameters, loaded['primitives_per_column'], primitive_count_per_column=loaded['primitive_count_per_column'])
            for original, recovered in zip(columns.expand(), restored.expand()):
                np.testing.assert_array_equal(original.numpy(), recovered.numpy())

            # variable groups must load through the normal per-primitive geometry path.
            info = detect_asset_info(asset, use_columns_representation=True)
            np.testing.assert_array_equal(info.primitive_count_per_column[0], [2, 5])
            geometry = load_pyramid_level(info, 0, use_asset_albedo=True)
            self.assertEqual(geometry['type'], 'ellipsoidsmesh')
            self.assertNotIn('primitives_per_column', geometry)
            self.assertEqual(geometry['centers'].shape, (7, 3))
            scene = mi.load_dict({'type': 'scene', 'primitives': geometry})
            self.assertEqual(mi.traverse(scene)['primitives.opacities'].shape, (7,))
            subset = load_pyramid_level(info, 0, primitive_subset=slice(1, 4))
            self.assertEqual(subset['centers'].shape, (3, 3))

    def test_legacy_fixed_count_file(self):
        # recreate the previous v1 layout with no per-column count field.
        with tempfile.TemporaryDirectory() as directory:
            columns, asset, path = save_columns(directory, [2, 2])
            ply = plyfile.PlyData.read(path, mmap=False)
            metadata = ply['metadata'].data.copy()
            metadata['format_version'] = 1
            names = [name for name in ply['column'].data.dtype.names if name != 'primitive_count']
            records = np.empty(2, dtype=[(name, 'f4') for name in names])
            for name in names:
                records[name] = ply['column'].data[name]
            plyfile.PlyData([plyfile.PlyElement.describe(metadata, 'metadata'),
                             plyfile.PlyElement.describe(records, 'column'), ply['primitive']]).write(path)
            loaded = load_ellipsoid_ply(path, use_columns_representation=True)
            np.testing.assert_array_equal(loaded['primitive_count_per_column'], [2, 2])
            info = detect_asset_info(asset, use_columns_representation=True)
            geometry = load_pyramid_level(info, 0)
            self.assertEqual(geometry['type'], 'ellipsoids')
            self.assertEqual(geometry['primitives_per_column'], 2)
            self.assertEqual(geometry['centers'].shape, (4, 3))

    def test_uniform_counts_can_differ_from_initial_k(self):
        # grouped rendering must use current counts rather than the initialization metadata.
        with tempfile.TemporaryDirectory() as directory:
            _, asset, _ = save_columns(directory, [3, 3])
            info = detect_asset_info(asset, use_columns_representation=True)
            geometry = load_pyramid_level(info, 0)
            self.assertEqual(geometry['primitives_per_column'], 3)
            self.assertEqual(geometry['centers'].shape, (6, 3))

    def test_reject_inconsistent_file_counts(self):
        # reject corrupted tables before creating GPU gather indices.
        with tempfile.TemporaryDirectory() as directory:
            _, _, path = save_columns(directory, [2, 5])
            ply = plyfile.PlyData.read(path, mmap=False)
            ply['column'].data['primitive_count'][1] = 4
            ply.write(path)
            with self.assertRaisesRegex(ValueError, 'sum'):
                load_ellipsoid_ply(path, use_columns_representation=True)

    def test_training_grouped_bvh_rejects_variable_counts(self):
        # fixed-size native grouping cannot interpret variable-length primitive ranges.
        columns = GaussianColumn(make_parameters(7), 2, primitive_count_per_column=[2, 5])
        with self.assertRaisesRegex(ValueError, 'fixed group size'):
            create_scene_dict_from_primitives(-1, 0, use_columns_representation=True, columns=columns, use_column_bvh=True)


if __name__ == '__main__':
    unittest.main()
