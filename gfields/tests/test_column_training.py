"""Focused checks for shared-footprint training and a native LS smoke-test crop."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import drjit as dr
import mitsuba as mi
import gfields


def PrepareRealCrop(output_path, source_path=None, block_size=64):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path) if source_path else Path(__file__).resolve().parents[2] / "data" / "ls_data"
    slice_paths = sorted(source_path.glob("22_cc_Area_*.png"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
    if len(slice_paths) != 30:
        raise ValueError("Expected the original 30 LS slices")
    projection = np.zeros(np.asarray(Image.open(slice_paths[0])).shape[:2], dtype=np.float32)
    for slice_path in slice_paths:
        projection += np.asarray(Image.open(slice_path).convert("L"), dtype=np.float32) / (255.0 * len(slice_paths))
    candidates = []
    for top in range(0, projection.shape[0] - block_size + 1, block_size):
        for left in range(0, projection.shape[1] - block_size + 1, block_size):
            region = projection[top:top + block_size, left:left + block_size]
            candidates.append((float(region.var()), top, left))
    _, top, left = max(candidates)
    slices = [np.asarray(Image.open(path).convert("L").crop((left, top, left + block_size, top + block_size)), dtype=np.float32) / 255.0 for path in slice_paths]
    native_volume = np.stack(slices)
    padding = (block_size - len(slices)) // 2
    volume = np.pad(native_volume, ((padding, block_size - len(slices) - padding), (0, 0), (0, 0)))
    np.save(output_path / "ls_native30_crop64_cube.npy", volume)
    Image.fromarray(np.uint8(native_volume[15] * 255)).resize((512, 512), Image.Resampling.NEAREST).save(output_path / "native_crop_z015.png")
    metadata = {"source": str(source_path), "axis_order": "z,y,x", "crop_xy": [left, top, left + block_size, top + block_size], "native_shape": list(native_volume.shape), "cube_shape": list(volume.shape), "z_padding": [padding, block_size - len(slices) - padding], "selection": "maximum variance of the mean-z projection among nonoverlapping 64x64 windows", "resampled": False}
    (output_path / "crop_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path / "ls_native30_crop64_cube.npy"


def MakeArguments():
    return SimpleNamespace(column_count=4, components_per_column=2, column_slab_depth=4, column_lr=0.01, centers_lr=1.0, scales_lr=1.0, quats_lr=1.0, opacity_lr=1.0, init_scale=1.0, init_opacity=0.01, init_albedo=0.0)


class ColumnTrainingTests(unittest.TestCase):
    def setUp(self):
        from gfields.training.columns import InitializeColumns
        from gfields.training.configs import GenericConfig
        self.arguments = MakeArguments()
        self.volume = np.zeros((8, 8, 8), dtype=np.float32)
        self.volume[2:6, 1:7, 1:7] = 0.5
        self.columns, self.learning_rates = InitializeColumns(self.arguments, self.volume, GenericConfig())

    def test_shared_footprint_and_positive_initialization(self):
        centers, scales, quaternions, weights = self.columns.Expand()
        self.assertEqual(dr.width(centers), self.arguments.column_count * self.arguments.components_per_column)
        for values in (centers.x, centers.y, scales.x, scales.y):
            groups = values.numpy().reshape(-1, self.arguments.components_per_column)
            np.testing.assert_array_equal(groups, np.repeat(groups[:, :1], self.arguments.components_per_column, axis=1))
        self.assertTrue(np.all(scales.numpy() > 0.0))
        self.assertTrue(np.all(weights.numpy() > 0.0))
        self.assertEqual(dr.width(self.columns.column_optimizer["center_x"]), self.arguments.column_count)
        from gfields.training.columns import InitializeColumns
        from gfields.training.configs import GenericConfig
        anisotropic_volume = np.ones((8, 12, 16), dtype=np.float32)
        columns, _ = InitializeColumns(self.arguments, anisotropic_volume, GenericConfig())
        for name, half_voxel in zip(("scale_x", "scale_y", "axial_scales"), (1.0 / 16, 1.0 / 12, 1.0 / 8)):
            self.assertAlmostEqual(columns.column_optimizer.bounds[name][1], half_voxel)
            self.assertTrue(np.all(columns.column_optimizer[name].numpy() >= half_voxel))

    def test_render_gradient_and_raw_checkpoint_round_trip(self):
        from gfields.training.columns import ColumnPrimitives, SaveColumns, LoadColumns
        from gfields.training.regression import create_scene_dict_from_primitives
        scene_dict = create_scene_dict_from_primitives(-1, 0, use_anyhit=True, **ColumnPrimitives(self.columns))
        scene_dict["sensor"] = {"type": "orthographic", "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, -4], target=[0, 0, 0], up=[0, 1, 0]), "film": {"type": "hdrfilm", "width": 16, "height": 16, "rfilter": {"type": "box"}}, "sampler": {"type": "independent", "sample_count": 2}}
        scene = mi.load_dict(scene_dict)
        scene_params = mi.traverse(scene)
        self.columns.WriteToScene(scene_params)
        image = mi.render(scene, scene_params, spp=2, seed=17)
        weighted_pixels = 1.0 + mi.Float(np.linspace(0.0, 1.0, dr.width(image.array), dtype=np.float32))
        loss = dr.mean(dr.square(image.array) * weighted_pixels)
        dr.backward(loss)
        optimizer = self.columns.column_optimizer
        gradient = dr.grad(optimizer["density_weights"]).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(float(np.abs(gradient).sum()), 0.0)
        shared_gradient = dr.grad(optimizer["center_x"]).numpy()
        self.assertTrue(np.isfinite(shared_gradient).all())
        self.assertGreater(float(np.abs(shared_gradient).sum()), 0.0)
        original_weights = optimizer["density_weights"].numpy().copy()
        optimizer.step()
        self.columns.WriteToScene(scene_params)
        self.assertGreater(float(np.abs(optimizer["density_weights"].numpy() - original_weights).sum()), 0.0)
        checkpoint_root = Path(__file__).resolve().parents[2] / "results" / "column_test_results"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkpoint_root) as temporary_path:
            SaveColumns(self.columns, Path(temporary_path), iteration=7)
            restored, _, iteration = LoadColumns(Path(temporary_path), self.arguments)
            self.assertEqual(iteration, 7)
            self.assertEqual(restored.components_per_column, self.columns.components_per_column)
            self.assertEqual(restored.column_optimizer.bounds, optimizer.bounds)
            for name in optimizer.keys():
                np.testing.assert_array_equal(restored.column_optimizer[name].numpy(), optimizer[name].numpy())
            for original, loaded in zip(self.columns.Expand(), restored.Expand()):
                np.testing.assert_array_equal(original.numpy(), loaded.numpy())
            for current_optimizer in (optimizer, restored.column_optimizer):
                dr.clear_grad(current_optimizer.variables)
                dr.backward(dr.sum(dr.square(current_optimizer["density_weights"])))
                current_optimizer.step()
            for name in optimizer.keys():
                np.testing.assert_array_equal(restored.column_optimizer[name].numpy(), optimizer[name].numpy())

    def test_expanded_asset_without_column_sidecar_is_rejected(self):
        from gfields.training.columns import LoadColumns
        checkpoint_root = Path(__file__).resolve().parents[2] / "results" / "column_test_results"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkpoint_root) as temporary_path:
            asset_path = Path(temporary_path) / "optimized_asset_pyr0" / "npy_data"
            asset_path.mkdir(parents=True)
            np.save(asset_path / "centers.npy", np.zeros((8, 3), dtype=np.float32))
            with self.assertRaisesRegex(FileNotFoundError, "do not preserve sharing"):
                LoadColumns(Path(temporary_path), self.arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-real-data", action="store_true")
    parser.add_argument("--source", type=Path, default=None, help="Directory containing the original 30 numbered LS PNGs")
    parser.add_argument("--output", type=Path, default=Path("results/gaussian_column_training_results"))
    arguments = parser.parse_args()
    if arguments.prepare_real_data:
        print(PrepareRealCrop(arguments.output, arguments.source))
    else:
        unittest.main(argv=[__file__])
