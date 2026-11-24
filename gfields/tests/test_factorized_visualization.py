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
    factorized_envelope_segments, include_segment_limits, oblique_camera,
    geometry_group_centers, geometry_slices, factorized_center_markers,
)


def write_factors(filename, j=3, k=2):
    # store two separated groups with nonuniform weights and distinct y levels.
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
        # retain group, xz and y ordering with exactly the saved world-space scales.
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
        # horizontal edges stay on one y layer and axial edges keep one xz center.
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
        # deduplicate xz projections and do not reconnect across hidden tree vertices.
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
            path = root / 'optimized_asset_pyr0/data/root.primitives_pyr0.ply'
            write_factors(path)
            weights = load_result_geometry(path)['weights'][6:]
            from examples import visualize_result as module
            with patch.object(module, 'save_projection', wraps=module.save_projection) as save:
                files = visualize_result(root, dpi=40, alpha=0.05, center_size=12, group_ids=[1])
            self.assertEqual(len(files), 4)
            self.assertTrue(all(file.parent.name == 'groups_1' for file in files))
            for call in save.call_args_list:
                np.testing.assert_array_equal(call.kwargs['group_ids'], 1)
                self.assertTrue(np.all(call.args[0][:, 0] >= 10))
                # use primitive opacity for side views and summed opacity for xz footprints.
                expected = weights.reshape(3, 2).sum(axis=1) if call.args[5] == (0, 2) else weights
                np.testing.assert_allclose(call.args[3], np.log10(expected))
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

    def test_envelope_contains_full_ellipses_and_axial_support(self):
        # surround the complete ellipse supports rather than taking a hull of centers.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            selected = np.arange(6)
            edges, groups = factorized_envelope_segments(geometry, selected, 3)
            top_edges, top_groups = factorized_envelope_segments(geometry, selected, 3, project_xz=True)
            np.testing.assert_array_equal(groups, 0)
            np.testing.assert_array_equal(top_groups, 0)
            self.assertEqual(len(edges), 2 * len(top_edges) + 8)
            self.assertAlmostEqual(float(edges[:, :, 1].min()), -0.35, places=6)
            self.assertAlmostEqual(float(edges[:, :, 1].max()), 0.45, places=6)
            vertical = edges[np.abs(edges[:, 1, 1] - edges[:, 0, 1]) > 1e-8]
            np.testing.assert_array_equal(vertical[:, 0][:, (0, 2)], vertical[:, 1][:, (0, 2)])

            # test dense true ellipse samples against every counterclockwise boundary edge.
            angle = np.arange(1024) * (2 * np.pi / 1024)
            circle = np.stack((np.cos(angle), np.sin(angle)))
            covariance = geometry['covariances'][[0, 2, 4]][:, (0, 2), :][:, :, (0, 2)]
            values, vectors = np.linalg.eigh(covariance)
            transform = vectors * np.sqrt(values)[:, None, :] * 3
            points = np.einsum('nij,jp->npi', transform, circle) + geometry['centers'][[0, 2, 4]][:, None, (0, 2)]
            lines = top_edges[:, :, (0, 2)]
            direction = lines[:, 1] - lines[:, 0]
            delta = points.reshape(1, -1, 2) - lines[:, :1]
            cross = direction[:, None, 0] * delta[:, :, 1] - direction[:, None, 1] * delta[:, :, 0]
            self.assertGreaterEqual(float(cross.min()), -1e-10)

    def test_envelope_slices_and_group_isolation(self):
        # exclude hidden footprints and y layers while keeping group outlines independent.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            edges, groups = factorized_envelope_segments(geometry, np.arange(12), 3)
            self.assertLess(float(edges[groups == 0, :, 0].max()), 3)
            self.assertGreater(float(edges[groups == 1, :, 0].min()), 9)
            sliced, _ = factorized_envelope_segments(geometry, np.array([0, 2, 4]), 3)
            self.assertAlmostEqual(float(sliced[:, :, 1].max()), -0.05, places=6)
            footprint, _ = factorized_envelope_segments(geometry, np.array([0, 1]), 3)
            self.assertLess(float(footprint[:, :, 0].max()), 1)

    def test_single_group_axis_and_factor_markers(self):
        # replace repeated primitive axes with one support-length line and k axial markers.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            geometry['factorized_edges'] = factorized_geometry_edges(geometry)
            selected = np.arange(6)
            (axial, transverse), groups = factorized_geometry_segments(geometry, selected)
            self.assertEqual(axial.shape, (1, 2, 3))
            self.assertEqual(len(transverse), 2)
            expected_center = np.array([4 / 3, 2 / 15, 7 / 3])
            np.testing.assert_allclose(geometry_group_centers(geometry)[0], expected_center, atol=1e-7)
            np.testing.assert_allclose(axial[0][:, (0, 2)], np.tile(expected_center[[0, 2]], (2, 1)), atol=1e-7)
            np.testing.assert_allclose(axial[0, :, 1], [-0.35, 0.45], atol=1e-7)
            np.testing.assert_allclose(transverse[:, :, 1], expected_center[1], atol=1e-7)
            markers, parents = factorized_center_markers(geometry, selected)
            self.assertEqual(len(markers), 5)
            np.testing.assert_allclose(markers[-2:, 1], [-0.2, 0.3], atol=1e-7)
            np.testing.assert_allclose(markers[-2:][:, (0, 2)], np.tile(expected_center[[0, 2]], (2, 1)), atol=1e-7)
            np.testing.assert_array_equal(parents, 0)
            self.assertEqual(len(factorized_center_markers(geometry, selected, project_xz=True)[0]), 3)

    def test_slices_select_complete_groups_by_density_center(self):
        # every group appears once with all j*k components, including centers outside the slab.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            slices = list(geometry_slices(geometry, np.arange(12), axis=0, width=1))
            np.testing.assert_array_equal(np.sort(np.concatenate([indices for _, _, indices in slices])), np.arange(12))
            occupied = [(lower, upper, indices) for lower, upper, indices in slices if len(indices)]
            self.assertEqual(len(occupied), 2)
            for lower, upper, indices in occupied:
                self.assertEqual(len(indices), 6)
                group = geometry['group_ids'][indices[0]]
                np.testing.assert_array_equal(geometry['group_ids'][indices], group)
                center = geometry_group_centers(geometry)[group, 0]
                self.assertTrue(lower <= center <= upper)
                self.assertTrue(np.any((geometry['centers'][indices, 0] < lower) | (geometry['centers'][indices, 0] > upper)))
            y_slices = list(geometry_slices(geometry, np.arange(12), axis=1, width=0.02))
            self.assertEqual(len(y_slices), 1)
            self.assertEqual(len(y_slices[0][2]), 12)

    def test_zero_weight_group_center_falls_back_to_geometry(self):
        # invisible groups still have a finite and useful visualization center.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path)
            geometry = load_result_geometry(path)
            geometry['weights'][:6] = 0
            np.testing.assert_allclose(geometry_group_centers(geometry)[0], geometry['centers'][:6].mean(axis=0), atol=1e-7)

    def test_legacy_columns_also_select_complete_groups(self):
        from gfields.tests.test_visualize_result import write_columns

        # keep a legacy column intact even when its primitive centers span several slabs.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_columns(path, [2, 3])
            geometry = load_result_geometry(path)
            occupied = [selected for _, _, selected in geometry_slices(geometry, np.arange(5), 1, 0.02) if len(selected)]
            np.testing.assert_array_equal(occupied[0], [0, 1])
            np.testing.assert_array_equal(occupied[1], [2, 3, 4])

    def test_oblique_bounds_include_envelope_corners(self):
        # cylinder corners can exceed projected ellipsoid limits and must not be clipped.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'root.primitives_pyr0.ply'
            write_factors(path, j=1, k=1)
            geometry = load_result_geometry(path)
            edges, _ = factorized_envelope_segments(geometry, np.array([0]), 3)
            camera = oblique_camera()
            projected = edges @ camera.T
            bounds = include_segment_limits(np.array([[-0.01, 0.01], [-0.01, 0.01]]), (projected,), (0, 1))
            for axis in (0, 1):
                self.assertLessEqual(bounds[axis, 0], projected[:, :, axis].min())
                self.assertGreaterEqual(bounds[axis, 1], projected[:, :, axis].max())
            empty, groups = factorized_envelope_segments(geometry, np.array([], dtype=int), 3)
            self.assertEqual(empty.shape, (0, 2, 3))
            self.assertEqual(len(groups), 0)

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
