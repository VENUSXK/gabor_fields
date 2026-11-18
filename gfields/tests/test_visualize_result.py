"""Check standalone result visualization for compact and independent Gaussian PLY files."""
import tempfile
import csv
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from examples.visualize_result import find_result_ply, load_result_geometry, gaussian_covariances, projected_ellipses, primitive_slices, oblique_geometry, column_geometry_segments, visualize_result


def write_columns(filename, counts, version=2):
    # create small files whose grouping and center order can be checked exactly.
    metadata = np.array([(version, counts[0], 3.0)], dtype=[('format_version', 'u4'), ('primitives_per_column', 'u4'), ('extent', 'f4')])
    fields = [('center_x', 'f4'), ('center_z', 'f4'), ('scale_x', 'f4'), ('scale_z', 'f4'), ('rotation_angle', 'f4')] + ([('primitive_count', 'u4')] if version == 2 else [])
    columns = np.zeros(len(counts), dtype=fields)
    columns['center_x'] = np.arange(len(counts)) * 0.2
    columns['center_z'] = np.arange(len(counts)) * 0.3
    columns['scale_x'], columns['scale_z'] = 0.05, 0.1
    columns['rotation_angle'] = 0.35
    if version == 2:
        columns['primitive_count'] = counts
    primitives = np.zeros(sum(counts), dtype=[('center_y', 'f4'), ('scale_y', 'f4'), ('opacity', 'f4')])
    primitives['center_y'] = np.linspace(-0.1, 0.1, sum(counts))
    primitives['opacity'] = 1e-4
    primitives['scale_y'] = 0.02
    filename.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(metadata, 'metadata'), PlyElement.describe(columns, 'column'),
             PlyElement.describe(primitives, 'primitive')]).write(str(filename))


class ResultVisualizationTests(unittest.TestCase):
    def test_variable_columns_and_separate_output_files(self):
        # verify projections, exact txt counts, and cleanup of obsolete generated images.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = root / 'optimized_asset_pyr0/data/root.primitives_pyr0.ply'
            write_columns(filename, [2, 3])
            for source in (root, filename.parent.parent, filename.parent, filename):
                self.assertEqual(find_result_ply(source), filename)
            geometry = load_result_geometry(filename)
            centers, counts = geometry['centers'], geometry['counts']
            np.testing.assert_allclose(centers[:, 0], [0, 0, 0.2, 0.2, 0.2])
            np.testing.assert_array_equal(counts, [2, 3])
            output = root / 'visualizations/pyr0'
            output.mkdir(parents=True)
            obsolete = output / 'columns_k2.png'
            obsolete.write_bytes(b'old generated image')
            unrelated = output / 'columns_k2_notes.png'
            unrelated.write_bytes(b'keep user file')
            files = visualize_result(root, dpi=40)
            self.assertEqual({file.name for file in files}, {'front_xy.png', 'side_zy.png', 'top_xz.png', 'column_k_distribution.txt'})
            self.assertFalse(obsolete.exists())
            self.assertTrue(unrelated.exists())
            for file in files:
                self.assertGreater(file.stat().st_size, 0)
            rows = np.loadtxt(files[-1], skiprows=1, delimiter='\t', dtype=int)
            np.testing.assert_array_equal(rows, [[2, 1, 2], [3, 1, 3]])
            self.assertIn(b'\r\n', files[-1].read_bytes())

    def test_legacy_fixed_column_layout(self):
        # recover fixed group sizes from v1 metadata without a primitive_count field.
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'root.primitives_pyr0.ply'
            write_columns(filename, [2, 2], version=1)
            geometry = load_result_geometry(filename)
            centers, counts = geometry['centers'], geometry['counts']
            np.testing.assert_array_equal(counts, [2, 2])
            np.testing.assert_allclose(centers[:, 0], [0, 0, 0.2, 0.2])

    def test_independent_gaussian_layout(self):
        # draw ordinary gaussian centers without assigning nonexistent columns.
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'root.primitives_pyr0.ply'
            fields = [(name, 'f4') for name in ('x', 'y', 'z', 'opacities_0', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3')]
            vertices = np.zeros(2, dtype=fields)
            vertices['x'], vertices['y'], vertices['z'] = [0, 0.3], [0.1, 0.4], [0.2, 0.5]
            vertices['opacities_0'] = [1e-4, 2e-4]
            vertices['rot_0'] = 1
            for axis in range(3):
                vertices[f'scale_{axis}'] = np.log(0.05)
            PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(filename))
            files = visualize_result(filename, dpi=40)
            self.assertEqual(len(files), 4)
            self.assertIn('no column grouping', files[-1].read_text())

    def test_projected_ellipse_uses_full_covariance(self):
        # tilted 3d ellipsoids project differently from central plane cross-sections.
        angle = np.pi / 4
        rotation = [[np.sin(angle / 2), 0, 0, np.cos(angle / 2)]]
        covariance = gaussian_covariances(np.array([[1, 2, 3]]), rotation)
        widths, heights, angles = projected_ellipses(covariance, (0, 1), 3.0)
        np.testing.assert_allclose(widths, [6 * np.sqrt(6.5)])
        np.testing.assert_allclose(heights, [6])
        np.testing.assert_allclose(np.mod(angles, 180), [90])

    def test_column_rotation_and_extent(self):
        # preserve y-axis rotation, projected orientation, and per-primitive extent factors.
        angle = np.pi / 4
        rotations = [[0, np.sin(angle / 2), 0, np.cos(angle / 2)]] * 2
        covariance = gaussian_covariances(np.array([[2, 1, 1], [2, 1, 1]]), rotations)
        np.testing.assert_allclose(covariance[:, 0, 2], [-1.5, -1.5])
        widths, heights, angles = projected_ellipses(covariance, (0, 2), np.array([1.0, 3.0]))
        np.testing.assert_allclose(widths, [4, 12])
        np.testing.assert_allclose(heights, [2, 6])
        np.testing.assert_allclose(np.mod(angles, 180), [135, 135])

    def test_slice_partition_has_no_duplicate_boundary_centers(self):
        # shared boundaries and the final maximum must not lose or duplicate primitives.
        centers = np.column_stack(([0.0, 0.1, 0.2, 0.3], np.zeros(4), np.zeros(4)))
        slices = list(primitive_slices(centers, 0, 0.1))
        indices = np.concatenate([selected for _, _, selected in slices])
        np.testing.assert_array_equal(np.sort(indices), np.arange(4))
        self.assertEqual(len(np.unique(indices)), 4)
        self.assertIn(1, slices[1][2])
        self.assertIn(3, slices[-1][2])
        for lower, upper, selected in slices:
            self.assertAlmostEqual(upper - lower, 0.1)
            self.assertTrue(np.all(centers[selected, 0] >= lower - 1e-12))
            self.assertTrue(np.all(centers[selected, 0] <= upper + 1e-12))

    def test_slice_images_manifest_and_visible_centers(self):
        # retain empty slabs, draw visible center markers, and account for every primitive.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_columns(root / 'optimized_asset_pyr0/data/root.primitives_pyr0.ply', [2, 3])
            files = visualize_result(root, dpi=60, slice_axis='x', slice_width=0.1, center_size=30)
            summary = next(file for file in files if file.name == 'slice_distribution.txt')
            with summary.open(newline='') as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(sum(int(row['primitive_count']) for row in rows), 5)
            self.assertTrue(all(row['axis'] == 'x' for row in rows))
            self.assertNotIn('front_xy.png', [file.name for file in files])
            empty = next(row for row in rows if int(row['primitive_count']) == 0)
            with Image.open(summary.parent / empty['filename']) as image:
                self.assertTrue(np.all(np.asarray(image)[:, :, :3] == 255))
            occupied = next(row for row in rows if int(row['primitive_count']) > 0)
            with Image.open(summary.parent / occupied['filename']) as image:
                pixels = np.asarray(image)[:, :, :3].astype(int)
                self.assertTrue(np.any((pixels[:, :, 0] < 120) & (pixels[:, :, 2] > pixels[:, :, 0] + 10)))

    def test_oblique_camera_matches_axis_aligned_views(self):
        # a front-facing camera reproduces the existing yz covariance projection.
        centers = np.array([[1.0, 2.0, 3.0]])
        covariance = np.array([[[4.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 2.0]]])
        view_centers, view_covariance = oblique_geometry(centers, covariance, elevation=0, azimuth=0)
        np.testing.assert_allclose(view_centers[:, :2], centers[:, (2, 1)], atol=1e-12)
        np.testing.assert_allclose(view_covariance[:, :2, :2], covariance[:, (2, 1), :][:, :, (2, 1)], atol=1e-12)

        # looking down y reproduces xz and camera rotation preserves physical covariance eigenvalues.
        view_centers, view_covariance = oblique_geometry(centers, covariance, elevation=90, azimuth=-90)
        np.testing.assert_allclose(view_centers[:, :2], centers[:, (0, 2)], atol=1e-12)
        np.testing.assert_allclose(np.linalg.eigvalsh(view_covariance), np.linalg.eigvalsh(covariance), atol=1e-12)

    def test_y_slices_have_xz_and_oblique_images(self):
        # export matching slice pairs and name both images in the same interval manifest.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_columns(root / 'optimized_asset_pyr0/data/root.primitives_pyr0.ply', [2, 3])
            files = visualize_result(root, dpi=40, slice_axis='y', slice_width=0.1, slice_3d=True)
            summary = next(file for file in files if file.name == 'slice_distribution.txt')
            with summary.open(newline='') as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(sum(int(row['primitive_count']) for row in rows), 5)
            for row in rows:
                self.assertEqual(row['axis'], 'y')
                self.assertTrue((summary.parent / row['filename']).exists())
                self.assertTrue((summary.parent / row['filename_3d']).exists())
            self.assertEqual(len([file for file in files if file.suffix == '.png']), 2 * len(rows))

    def test_column_outline_and_center_lines_preserve_membership(self):
        # every outline belongs to one column; selected centers connect only along that column.
        centers = np.array([[0, 2, 0], [0, 0, 0], [0, 100, 0], [10, 0, 0], [10, 2, 0]])
        covariance = np.tile(np.diag([0.01, 0.04, 0.09]), (5, 1, 1))
        axial, outline = column_geometry_segments(centers, covariance, 1.0, np.array([3, 2]), np.array([0, 1, 3, 4]))
        self.assertEqual(axial.shape, (2, 2, 3))
        np.testing.assert_array_equal(axial[:, 0, (0, 2)], axial[:, 1, (0, 2)])
        self.assertTrue(np.all(axial[:, 0, 1] <= axial[:, 1, 1]))
        self.assertAlmostEqual(outline[:, :, 1].min(), -0.2)
        self.assertAlmostEqual(outline[:, :, 1].max(), 2.2)
        for line in outline:
            self.assertTrue(np.all(np.abs(line[:, 0]) <= 0.100001) or np.all(np.abs(line[:, 0] - 10) <= 0.100001))

    def test_single_column_has_an_xz_outline(self):
        # one primitive still shows the shared footprint and axial support as a column envelope.
        centers = np.array([[0.0, 0.0, 0.0]])
        covariance = np.array([np.diag([4.0, 1.0, 9.0])])
        axial, outline = column_geometry_segments(centers, covariance, 2.0, np.array([1]), np.array([0]))
        self.assertEqual(len(axial), 0)
        np.testing.assert_allclose(outline.reshape(-1, 3).min(axis=0), [-4, -2, -6], atol=1e-12)
        np.testing.assert_allclose(outline.reshape(-1, 3).max(axis=0), [4, 2, 6], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
