"""Check CPU-only FGC visualization, group trees, filtered views, and slice connections."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from examples.visualize_result import (
    load_result_geometry, minimum_spanning_edges, factorized_geometry_edges,
    factorized_geometry_segments, projection_geometry, factorized_colors, visualize_result,
)


def write_factors(filename, j=3, k=2):
    # store two separated groups with nonuniform weights and distinct Y levels.
    metadata = np.array([(3, 2, j, k, 3.0)], dtype=[('format_version', 'u4'), ('group_count', 'u4'),
                        ('xz_gaussians_per_group', 'u4'), ('y_gaussians_per_group', 'u4'), ('extent', 'f4')])
    groups = np.array([(0.4,), (0.8,)], dtype=[('opacity', 'f4')])
    xz = np.zeros(2 * j, dtype=[(name, 'f4') for name in ('center_x', 'center_z', 'scale_x', 'scale_z', 'rotation_angle', 'weight')])
    xz['center_x'] = np.tile(np.arange(j), 2) + np.repeat([0, 10], j)
    xz['center_z'] = np.tile(np.arange(j) ** 2, 2)
    xz['scale_x'], xz['scale_z'], xz['rotation_angle'] = 0.1, 0.2, 0.4
    xz['weight'] = np.tile(np.arange(1, j + 1) / np.arange(1, j + 1).sum(), 2)
    y = np.zeros(2 * k, dtype=[('center_y', 'f4'), ('scale_y', 'f4'), ('weight', 'f4')])
    y['center_y'] = np.tile(np.linspace(-0.2, 0.3, k), 2)
    y['scale_y'] = 0.05
    y['weight'] = np.tile(np.arange(1, k + 1) / np.arange(1, k + 1).sum(), 2)
    filename.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, name) for name, data in
             (('metadata', metadata), ('group', groups), ('xz', xz), ('y', y))]).write(str(filename))


class FactorizedVisualizationTests(unittest.TestCase):
    def test_physical_geometry_and_factor_layout(self):
        # retain group, XZ and Y ordering with exactly the saved world-space scales.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            self.assertEqual(geometry['factorized_shape'], (2, 3, 2))
            np.testing.assert_array_equal(geometry['group_ids'], np.repeat([0, 1], 6))
            np.testing.assert_allclose(geometry['centers'][:, 0], np.repeat([0, 1, 2, 10, 11, 12], 2))
            np.testing.assert_allclose(geometry['centers'][:, 1], np.tile([-0.2, 0.3], 6))
            np.testing.assert_allclose(geometry['weights'].reshape(2, -1).sum(axis=1), [0.4, 0.8], rtol=1e-6)
            np.testing.assert_allclose(np.linalg.eigvalsh(geometry['covariances']), np.tile([0.05**2, 0.1**2, 0.2**2], (12, 1)), rtol=1e-6)

    def test_minimum_spanning_tree_and_coincident_points(self):
        # connect every center with minimal length and retain zero-length edges.
        points = np.array([[0, 0], [1, 0], [1, 2]])
        edges = minimum_spanning_edges(points)
        np.testing.assert_array_equal(edges, [[0, 1], [1, 2]])
        self.assertAlmostEqual(float(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1).sum()), 3)
        np.testing.assert_array_equal(minimum_spanning_edges(np.zeros((3, 2))), [[0, 1], [0, 2]])
        self.assertEqual(minimum_spanning_edges(np.zeros((1, 2))).shape, (0, 2))

    def test_connections_preserve_groups_and_factors(self):
        # horizontal edges stay on one Y layer and axial edges keep one XZ center.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            axial, transverse = factorized_geometry_edges(geometry)
            self.assertEqual(axial.shape, (6, 2))
            self.assertEqual(transverse.shape, (8, 2))
            for edges in (axial, transverse):
                np.testing.assert_array_equal(geometry['group_ids'][edges[:, 0]], geometry['group_ids'][edges[:, 1]])
            centers = geometry['centers']
            np.testing.assert_array_equal(centers[axial[:, 0]][:, (0, 2)], centers[axial[:, 1]][:, (0, 2)])
            np.testing.assert_array_equal(centers[transverse[:, 0], 1], centers[transverse[:, 1], 1])

    def test_top_projection_and_slices_keep_original_tree(self):
        # deduplicate XZ projections and do not reconnect across hidden tree vertices.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            geometry['factorized_edges'] = factorized_geometry_edges(geometry)
            selected, segments, groups = projection_geometry(geometry, np.arange(12), (0, 2), 3, True)
            self.assertEqual(len(selected), 6)
            self.assertEqual(len(segments[0]), 0)
            self.assertEqual(len(segments[1]), 4)
            np.testing.assert_array_equal(groups[1], [0, 0, 1, 1])
            segments, groups = factorized_geometry_segments(geometry, np.array([0, 4]), project_xz=True)
            self.assertEqual(len(segments[1]), 0)
            selected, segments, _ = projection_geometry(geometry, np.arange(12), (0, 2), 3, False)
            self.assertEqual(len(selected), 6)
            self.assertIsNone(segments)

    def test_group_filter_and_output_images(self):
        # preserve original group ids and colors while exporting only the requested group.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_factors(root / 'optimized_asset_pyr0/data/root.primitives_pyr0.ply')
            from examples import visualize_result as module
            with patch.object(module, 'save_projection', wraps=module.save_projection) as save:
                files = visualize_result(root, dpi=40, alpha=0.05, center_size=12, group_ids=[1])
            self.assertEqual(len(files), 4)
            self.assertTrue(all(file.parent.name == 'groups_1' for file in files))
            for call in save.call_args_list:
                np.testing.assert_array_equal(call.kwargs['group_ids'], 1)
                self.assertTrue(np.all(call.args[0][:, 0] >= 10))
            with files[-1].open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(rows[0]['group_id'], '1')
            self.assertEqual(rows[0]['expanded_gaussians'], '6')
            for file in files[:-1]:
                with Image.open(file) as image:
                    self.assertGreater(np.ptp(np.asarray(image)[:, :, :3]), 0)
            np.testing.assert_array_equal(factorized_colors([1, 1])[0], factorized_colors([1])[0])
            with self.assertRaisesRegex(ValueError, 'zero-based'):
                visualize_result(root, dpi=40, group_ids=[2])

    def test_slice_pairs_and_degenerate_factor_counts(self):
        # generate paired views and retain correct behavior for a single factor on either axis.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'root.primitives_pyr0.ply'
            write_factors(path)
            files = visualize_result(path, dpi=40, slice_axis='y', slice_width=0.2, slice_3d=True, group_ids=[0])
            summary = next(file for file in files if file.name == 'slice_distribution.txt')
            with summary.open() as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(sum(int(row['primitive_count']) for row in rows), 6)
            for row in rows:
                self.assertTrue((summary.parent / row['filename']).exists())
                self.assertTrue((summary.parent / row['filename_3d']).exists())
            for j, k in ((1, 3), (3, 1), (1, 1)):
                write_factors(path, j=j, k=k)
                geometry = load_result_geometry(path)
                axial, transverse = factorized_geometry_edges(geometry)
                self.assertEqual(len(axial), 2 * j * (k - 1))
                self.assertEqual(len(transverse), 2 * (j - 1) * k)


if __name__ == '__main__':
    unittest.main()
