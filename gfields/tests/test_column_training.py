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


def prepare_real_crop(output_path, source_path=None, block_size=64):
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


def make_arguments():
    return SimpleNamespace(use_columns_representation=True, column_count=4, primitives_per_column=2, column_slab_depth=4, column_lr=0.01, centers_lr=1.0, scales_lr=1.0, quats_lr=1.0, opacity_lr=1.0, init_scale=1.0, init_opacity=0.01, init_albedo=0.0)


def setup_column_optimizer(columns, learning_rates, checkpoint_path=None):
    from gfields.training.regression import load_columns
    from gfields.training.regression import (
        create_scene_dict_from_primitives, write_scene_params_to_optimizer, configure_optimizer,
    )
    scene = mi.load_dict(create_scene_dict_from_primitives(-1, 0, use_anyhit=True, use_columns_representation=True, gaussian_columns=columns))
    opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
    args = make_arguments()
    write_scene_params_to_optimizer(mi.traverse(scene), opt, level=0, args=args, gaussian_columns=columns)
    configure_optimizer(opt, args, learning_rates, level=0, gaussian_columns=columns)
    if checkpoint_path is not None:
        load_columns(checkpoint_path, args, opt=opt)
    return opt


class ColumnTrainingTests(unittest.TestCase):
    def test_checkpoint_statistics_with_suspended_gradients(self):
        from gfields.training.columns import save_columns
        expected_count = self.arguments.column_count * (5 + 3 * self.arguments.primitives_per_column)
        with tempfile.TemporaryDirectory() as directory:
            # Training saves checkpoints inside a suspended-gradient scope.
            with dr.suspend_grad():
                stats = self.columns.parameter_stats()
                save_columns(self.columns, directory, iteration=2)
            self.assertEqual(stats['trainable_parameter_count'], expected_count)
            path = Path(directory) / 'optimized_asset_pyr0' / 'npy_data' / 'columns.npz'
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata']))
                self.assertEqual(metadata['trainable_parameter_bytes'], 4 * expected_count)

    def test_y_padding_initializes_columns_along_the_100_active_layers(self):
        from gfields.training.regression import generate_random_columns
        from gfields.training.configs import GenericConfig
        volume = np.zeros((1024, 128, 8), dtype=np.float32)
        volume[:, 14:114, :] = 1.0
        columns = generate_random_columns(64, 4, 0.003, 1.0, 0.0,
                                          volume=volume, volume_config=GenericConfig(), column_slab_depth=8)
        centers, scales, quaternions, _ = columns.expand()
        self.assertTrue(np.all(centers.y.numpy() >= 2.0 * 14 / 128 - 1.0))
        self.assertTrue(np.all(centers.y.numpy() < 2.0 * 114 / 128 - 1.0))
        for values in (centers.x, centers.z, scales.x, scales.z):
            groups = values.numpy().reshape(64, 4)
            np.testing.assert_array_equal(groups, np.repeat(groups[:, :1], 4, axis=1))
        np.testing.assert_allclose(quaternions.x.numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(quaternions.z.numpy(), 0.0, atol=1e-7)
        np.testing.assert_allclose(columns.scale_lower_bounds['column_scale_xz'], [1 / 8, 1 / 1024])
        self.assertAlmostEqual(columns.scale_lower_bounds['primitive_scale_y'], 1 / 128)

    def setUp(self):
        from gfields.training.regression import generate_random_columns
        from gfields.training.configs import GenericConfig
        self.arguments = make_arguments()
        self.volume = np.zeros((8, 8, 8), dtype=np.float32)
        self.volume[2:6, 1:7, 1:7] = 0.5
        self.columns = generate_random_columns(
            self.arguments.column_count, self.arguments.primitives_per_column, self.arguments.init_opacity,
            self.arguments.init_scale, self.arguments.init_albedo, volume=self.volume, volume_config=GenericConfig(),
            column_slab_depth=self.arguments.column_slab_depth
        )

        from gfields.training.regression import create_lr_dict
        self.learning_rates = create_lr_dict(self.arguments, self.arguments.column_lr, level=0)
        self.assertIsNone(self.columns.column_optimizer)
        setup_column_optimizer(self.columns, self.learning_rates)

    def test_scene_parameter_round_trip_before_optimizer_creation(self):
        from gfields.training.regression import generate_random_columns
        from gfields.training.configs import GenericConfig
        columns = generate_random_columns(
            4, 2, 0.01, 1.0, 0.0, volume=self.volume,
            volume_config=GenericConfig(), column_slab_depth=4,
        )
        self.assertIsNone(columns.column_optimizer)
        expected = [value.numpy().copy() for value in columns.expand()]
        setup_column_optimizer(columns, self.learning_rates)
        for original, expanded in zip(expected, columns.expand()):
            np.testing.assert_allclose(expanded.numpy(), original, atol=1e-6)

    def test_independent_gaussian_optimizer_configuration(self):
        from gfields.training.regression import (
            create_scene_dict_from_primitives, write_scene_params_to_optimizer,
            configure_optimizer, write_optimizer_params_to_scene,
        )
        column_scene = create_scene_dict_from_primitives(-1, 0, use_anyhit=True, use_columns_representation=True, gaussian_columns=self.columns)
        primitive_arrays = {name: column_scene['primitives_pyr0'][name]
                            for name in ('centers', 'scales', 'quaternions', 'opacities', 'albedo', 'omega')}
        scene = mi.load_dict(create_scene_dict_from_primitives(-1, 0, use_anyhit=True, **primitive_arrays))
        params = mi.traverse(scene)
        original_geometry = params['primitives_pyr0.data'].numpy().copy()
        opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
        args = SimpleNamespace(use_columns_representation=False)
        write_scene_params_to_optimizer(params, opt, level=0, args=args)
        configure_optimizer(opt, args, {'centers_0': 0.02}, level=0)
        self.assertIn('centers_0', opt)
        self.assertNotIn('column_center_xz', opt)
        self.assertEqual(opt.bounds['scales_0'], (1.0, 1e-8))
        self.assertEqual(opt.bounds['omega_0'], (0.0, 0.0))
        write_optimizer_params_to_scene(opt, params, level=0)
        np.testing.assert_allclose(params['primitives_pyr0.data'].numpy(), original_geometry, atol=1e-6)

    def test_regularization_preserves_scalar_loss_and_image_gradient(self):
        from gfields.training.regression import opacity_scale_regularization
        for level, use_columns in ((0, False), (1, True)):
            for count in (1, 4):
                for scale_weight in (0.0, 0.5):
                    with self.subTest(level=level, count=count, scale_weight=scale_weight):
                        opt = gfields.optimizers.BoundedAdam(noise_init=0.0)
                        opt[f'opacities_{level}'] = dr.full(mi.Float, 0.2, count)
                        opt[f'scales_{level}'] = mi.Vector3f(
                            dr.full(mi.Float, 1.0, count),
                            dr.full(mi.Float, 2.0, count),
                            dr.full(mi.Float, 3.0, count),
                        )
                        args = SimpleNamespace(use_columns_representation=use_columns,
                                               lambda_opacity=0.1, lambda_scale=scale_weight)
                        opacity, scale = opacity_scale_regularization(opt, args, level)
                        image_loss = mi.Float(2.0)
                        dr.enable_grad(image_loss)
                        loss = image_loss + mi.TensorXf(0.0) + opacity + scale
                        self.assertEqual(dr.width(loss.array), 1)
                        self.assertAlmostEqual(float(loss.array[0]),
                                               2.0 + count * 0.02 + count * 6.0 * scale_weight, places=5)
                        dr.backward(loss)
                        np.testing.assert_allclose(dr.grad(image_loss).numpy(), [1.0])
                        np.testing.assert_allclose(dr.grad(opt[f'opacities_{level}']).numpy(), 0.1)
                        np.testing.assert_allclose(dr.grad(opt[f'scales_{level}']).numpy(), scale_weight)

    def test_legacy_separate_xy_checkpoint_restores_adam_state(self):
        from gfields.training.columns import save_columns
        from gfields.training.regression import load_columns
        opt = self.columns.column_optimizer

        def step(current):
            dr.clear_grad(current.variables)
            dr.backward(dr.sum(dr.square(current['column_center_xz']), axis=None)
                        + dr.sum(dr.square(current['column_scale_xz']), axis=None)
                        + dr.sum(dr.square(current['primitive_opacity'])))
            current.step()

        step(opt)
        with tempfile.TemporaryDirectory() as directory:
            save_columns(self.columns, directory, iteration=1)
            path = Path(directory) / 'optimized_asset_pyr0' / 'npy_data' / 'columns.npz'
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            metadata = json.loads(str(arrays['metadata']))
            metadata['format'] = 'gaussian_columns_v1'

            # Recreate the former scalar XY parameters, bounds, and Adam arrays.
            for grouped, x_name, y_name in (
                ('column_center_xz', 'column_center_x', 'column_center_z'),
                ('column_scale_xz', 'column_scale_x', 'column_scale_z'),
            ):
                for suffix in ('', '__first_moment', '__second_moment'):
                    x_values, y_values = arrays.pop(grouped + suffix)
                    arrays[x_name + suffix], arrays[y_name + suffix] = x_values, y_values
                steps = metadata['adam_steps']
                steps[x_name] = steps[y_name] = steps.pop(grouped)
            lower_x, lower_y = metadata['scale_lower_bounds'].pop('column_scale_xz')
            metadata['scale_lower_bounds'].update(column_scale_x=lower_x, column_scale_z=lower_y)
            for suffix in ('', '__first_moment', '__second_moment'):
                arrays['density_weights' + suffix] = arrays.pop('primitive_opacity' + suffix)
            metadata['adam_steps']['density_weights'] = metadata['adam_steps'].pop('primitive_opacity')
            arrays['metadata'] = np.asarray(json.dumps(metadata))
            np.savez(path, **arrays)

            restored, iteration = load_columns(path, self.arguments)
            self.assertEqual(iteration, 1)
            restored_opt = setup_column_optimizer(restored, self.learning_rates, checkpoint_path=path)
            step(opt)
            step(restored_opt)
            for name in opt.keys():
                np.testing.assert_array_equal(opt[name].numpy(), restored_opt[name].numpy())

    def test_shared_footprint_and_positive_initialization(self):
        centers, scales, quaternions, weights = self.columns.expand()
        self.assertEqual(dr.width(centers), self.arguments.column_count * self.arguments.primitives_per_column)
        for values in (centers.x, centers.z, scales.x, scales.z):
            groups = values.numpy().reshape(-1, self.arguments.primitives_per_column)
            np.testing.assert_array_equal(groups, np.repeat(groups[:, :1], self.arguments.primitives_per_column, axis=1))
        self.assertTrue(np.all(scales.numpy() > 0.0))
        self.assertTrue(np.all(weights.numpy() > 0.0))
        self.assertEqual(dr.width(self.columns.column_optimizer["column_center_xz"]), self.arguments.column_count)
        from gfields.training.regression import generate_random_columns
        from gfields.training.configs import GenericConfig
        anisotropic_volume = np.ones((8, 12, 16), dtype=np.float32)
        columns = generate_random_columns(
            self.arguments.column_count, self.arguments.primitives_per_column, self.arguments.init_opacity,
            self.arguments.init_scale, self.arguments.init_albedo, volume=anisotropic_volume, volume_config=GenericConfig(),
            column_slab_depth=self.arguments.column_slab_depth
        )
        self.assertIsNone(columns.column_optimizer)
        setup_column_optimizer(columns, self.learning_rates)
        expected_bounds = {'column_scale_xz': [1.0 / 16, 1.0 / 8], 'primitive_scale_y': 1.0 / 12}
        for name, half_voxel in expected_bounds.items():
            lower = np.asarray(columns.column_optimizer.bounds[name][1])
            np.testing.assert_allclose(lower.reshape(-1), np.asarray(half_voxel).reshape(-1))
            self.assertTrue(np.all(columns.column_optimizer[name].numpy() >= lower))

    def test_render_gradient_and_raw_checkpoint_round_trip(self):
        from gfields.training.columns import save_columns
        from gfields.training.regression import load_columns
        from gfields.training.regression import create_scene_dict_from_primitives, write_optimizer_params_to_scene
        scene_dict = create_scene_dict_from_primitives(-1, 0, use_anyhit=True, use_columns_representation=True, gaussian_columns=self.columns)
        scene_dict["sensor"] = {"type": "orthographic", "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, -4], target=[0, 0, 0], up=[0, 1, 0]), "film": {"type": "hdrfilm", "width": 16, "height": 16, "rfilter": {"type": "box"}}, "sampler": {"type": "independent", "sample_count": 2}}
        scene = mi.load_dict(scene_dict)
        scene_params = mi.traverse(scene)
        write_optimizer_params_to_scene(self.columns.column_optimizer, scene_params, level=0, gaussian_columns=self.columns)
        optimizer = self.columns.column_optimizer
        # Reuse the scene graph written after each step for the next render.
        for iteration in range(3):
            dr.clear_grad(optimizer.variables)
            image = mi.render(scene, scene_params, spp=2, seed=17 + iteration)
            weighted_pixels = 1.0 + mi.Float(np.linspace(0.0, 1.0, dr.width(image.array), dtype=np.float32))
            loss = dr.mean(dr.square(image.array) * weighted_pixels)
            dr.backward(loss)
            for parameter in (optimizer['primitive_opacity'], optimizer['column_center_xz'].x):
                gradient = dr.grad(parameter).numpy()
                self.assertTrue(np.isfinite(gradient).all())
                self.assertGreater(float(np.abs(gradient).sum()), 0.0)
            original_opacity = optimizer['primitive_opacity'].numpy().copy()
            optimizer.step()
            write_optimizer_params_to_scene(optimizer, scene_params, level=0, gaussian_columns=self.columns)
            self.assertGreater(float(np.abs(optimizer['primitive_opacity'].numpy() - original_opacity).sum()), 0.0)
        checkpoint_root = Path(__file__).resolve().parents[2] / "results" / "column_test_results"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkpoint_root) as temporary_path:
            save_columns(self.columns, Path(temporary_path), iteration=7)
            restored, iteration = load_columns(Path(temporary_path), self.arguments)
            self.assertIsNone(restored.column_optimizer)
            setup_column_optimizer(restored, self.learning_rates, checkpoint_path=Path(temporary_path))
            self.assertEqual(iteration, 7)
            self.assertEqual(restored.primitives_per_column, self.columns.primitives_per_column)
            for name in optimizer.bounds:
                for restored_bound, original_bound in zip(restored.column_optimizer.bounds[name], optimizer.bounds[name]):
                    np.testing.assert_array_equal(np.asarray(restored_bound), np.asarray(original_bound))
            for name in optimizer.keys():
                np.testing.assert_array_equal(restored.column_optimizer[name].numpy(), optimizer[name].numpy())
            for original, loaded in zip(self.columns.expand(), restored.expand()):
                np.testing.assert_array_equal(original.numpy(), loaded.numpy())
            for current_optimizer in (optimizer, restored.column_optimizer):
                dr.clear_grad(current_optimizer.variables)
                dr.backward(dr.sum(dr.square(current_optimizer["primitive_opacity"])))
                current_optimizer.step()
            for name in optimizer.keys():
                np.testing.assert_array_equal(restored.column_optimizer[name].numpy(), optimizer[name].numpy())

    def test_expanded_asset_without_column_sidecar_is_rejected(self):
        from gfields.training.regression import load_columns
        checkpoint_root = Path(__file__).resolve().parents[2] / "results" / "column_test_results"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=checkpoint_root) as temporary_path:
            asset_path = Path(temporary_path) / "optimized_asset_pyr0" / "npy_data"
            asset_path.mkdir(parents=True)
            np.save(asset_path / "centers.npy", np.zeros((8, 3), dtype=np.float32))
            with self.assertRaisesRegex(FileNotFoundError, "do not preserve sharing"):
                load_columns(Path(temporary_path), self.arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-real-data", action="store_true")
    parser.add_argument("--source", type=Path, default=None, help="Directory containing the original 30 numbered LS PNGs")
    parser.add_argument("--output", type=Path, default=Path("results/gaussian_column_training_results"))
    arguments = parser.parse_args()
    if arguments.prepare_real_data:
        print(prepare_real_crop(arguments.output, arguments.source))
    else:
        unittest.main(argv=[__file__])
