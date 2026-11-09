"""Shared-footprint Gaussian training helpers; rendering still uses expanded ellipsoids."""
import json
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw

import gfields
from gfields.integrators.common import GaussianColumn

PARAMETER_NAMES = ('center_x', 'center_y', 'scale_x', 'scale_y', 'rotation_angle', 'axial_centers', 'axial_scales', 'density_weights')


def CreateColumnOptimizer(parameters, args, scale_lower_bounds=None):
    optimizer = gfields.optimizers.BoundedAdam(lr=args.column_lr, noise_init=0.0)
    scale_lower_bounds = scale_lower_bounds or {'scale_x': 1e-5, 'scale_y': 1e-5, 'axial_scales': 1e-5}
    learning_rates = {}
    for name in PARAMETER_NAMES:
        parameter = np.asarray(parameters[name], dtype=np.float32).reshape(-1)
        if name in scale_lower_bounds:
            parameter = np.maximum(parameter, scale_lower_bounds[name])
        optimizer[name] = mi.Float(parameter)
        multiplier = args.opacity_lr if name == 'density_weights' else args.quats_lr if name == 'rotation_angle' else args.scales_lr if 'scale' in name else args.centers_lr
        learning_rates[name] = args.column_lr * multiplier
        if 'center' in name:
            optimizer.set_bounds(name, lower=-1.0, upper=1.0)
        elif 'scale' in name:
            optimizer.set_bounds(name, lower=float(scale_lower_bounds[name]), upper=1.0)
        elif name == 'density_weights':
            optimizer.set_bounds(name, lower=1e-8, upper=1.0)
    optimizer.set_learning_rate(learning_rates)
    return optimizer, learning_rates


def InitializeColumns(args, volume, volume_config):
    column_count, component_count = args.column_count, args.components_per_column
    if column_count <= 0 or component_count <= 0:
        raise ValueError('Column and component counts must be positive')
    if volume_config.opacity_scale <= 0.0:
        raise ValueError('Gaussian column training requires a positive opacity_scale')
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Columns require a finite 3D volume in z,y,x order')
    matrix = np.asarray(volume_config.to_world.matrix, dtype=np.float64)
    if not np.allclose(matrix[:3, :3], np.diag(np.diag(matrix[:3, :3]))):
        raise ValueError('Gaussian columns currently require an axis-aligned volume transform')
    rng = np.random.default_rng(getattr(args, 'seed_offset', 0))
    occupied = np.any(np.abs(volume) > 1e-6, axis=(1, 2))
    occupied_z = np.flatnonzero(occupied)
    start_z, end_z = (int(occupied_z[0]), int(occupied_z[-1]) + 1) if occupied_z.size else (0, volume.shape[0])
    slab_depth = args.column_slab_depth or end_z - start_z
    if slab_depth <= 0:
        raise ValueError('column_slab_depth must be nonnegative')
    slab_starts = np.arange(start_z, end_z, slab_depth)
    if column_count < len(slab_starts):
        raise ValueError('column_count must cover every initialized slab; increase it or column_slab_depth')
    slab_indices = np.arange(column_count) % len(slab_starts)
    centers_x, centers_y, axial_centers, axial_scales = [], [], [], []
    slab_projections = {}
    for slab_index, slab_start in enumerate(slab_starts):
        projection = np.maximum(volume[int(slab_start):min(int(slab_start) + slab_depth, end_z)], 0.0).sum(axis=0, dtype=np.float64)
        probabilities = projection.reshape(-1)
        total = probabilities.sum()
        slab_projections[slab_index] = (projection.shape, probabilities / total if total > 0.0 else None)
    for column_index, slab_index in enumerate(slab_indices):
        lower, upper = int(slab_starts[slab_index]), min(int(slab_starts[slab_index]) + slab_depth, end_z)
        projection_shape, probabilities = slab_projections[slab_index]
        pixel = rng.choice(np.prod(projection_shape), p=probabilities)
        pixel_y, pixel_x = np.unravel_index(pixel, projection_shape)
        centers_x.append(matrix[0, 0] * ((pixel_x + 0.5) / volume.shape[2]) + matrix[0, 3])
        centers_y.append(matrix[1, 1] * ((pixel_y + 0.5) / volume.shape[1]) + matrix[1, 3])
        normalized_z = (lower + (np.arange(component_count) + 0.5) * (upper - lower) / component_count) / volume.shape[0]
        axial_centers.extend(matrix[2, 2] * normalized_z + matrix[2, 3])
        axial_scales.extend([max(1e-4, abs(matrix[2, 2]) * (upper - lower) / volume.shape[0] / component_count * 0.65)] * component_count)
    columns_per_slab = column_count / len(slab_starts)
    footprint_scale = np.clip(abs(args.init_scale) * 0.5 / np.sqrt(columns_per_slab), 1e-4, 0.5)
    parameters = {'center_x': centers_x, 'center_y': centers_y, 'scale_x': np.full(column_count, footprint_scale), 'scale_y': np.full(column_count, footprint_scale), 'rotation_angle': rng.uniform(-np.pi, np.pi, column_count), 'axial_centers': axial_centers, 'axial_scales': axial_scales, 'density_weights': np.full(column_count * component_count, args.init_opacity)}
    world_voxel_size = np.abs(np.diag(matrix)[:3]) / np.asarray(volume.shape[::-1])
    scale_lower_bounds = dict(zip(('scale_x', 'scale_y', 'axial_scales'), np.maximum(0.5 * world_voxel_size, 1e-5)))
    optimizer, learning_rates = CreateColumnOptimizer(parameters, args, scale_lower_bounds)
    columns = GaussianColumn(optimizer, component_count)
    columns.init_albedo = float(args.init_albedo)
    return columns, learning_rates


def ColumnPrimitives(columns, init_albedo=None):
    centers, scales, quaternions, weights = columns.Expand()
    count = dr.width(weights)
    albedo = getattr(columns, 'init_albedo', 0.0) if init_albedo is None else init_albedo
    return {'centers': mi.TensorXf(centers.numpy().T), 'scales': mi.TensorXf(scales.numpy().T), 'quaternions': mi.TensorXf(quaternions.numpy().T), 'opacities': mi.TensorXf(weights.numpy().reshape(count, 1)), 'omega': mi.TensorXf(np.zeros((count, 1), dtype=np.float32)), 'albedo': mi.TensorXf(np.full((count, 3), albedo, dtype=np.float32))}


def ColumnParameterStats(columns):
    optimizer = columns.column_optimizer
    column_count = dr.width(optimizer['center_x'])
    component_count = dr.width(optimizer['axial_centers'])
    parameter_count = sum(dr.width(optimizer[name]) for name in PARAMETER_NAMES)
    return {'column_count': column_count, 'components_per_column': columns.components_per_column, 'expanded_gaussian_count': component_count, 'trainable_parameter_count': parameter_count, 'trainable_parameter_bytes': 4 * parameter_count, 'independent_axis_aligned_xy_gaussian_parameter_count': 8 * component_count, 'independent_full_3d_gaussian_parameter_count': 11 * component_count, 'rendering_uses_expanded_gaussians': True}


def SaveColumns(columns, output_path, iteration=0):
    checkpoint_dir = Path(output_path) / 'optimized_asset_pyr0' / 'npy_data'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = columns.column_optimizer
    arrays = {name: np.asarray(optimizer[name], dtype=np.float32) for name in PARAMETER_NAMES}
    scale_lower_bounds = {name: optimizer.bounds[name][1] for name in ('scale_x', 'scale_y', 'axial_scales')}
    metadata = {'format': 'gaussian_columns_v1', 'components_per_column': columns.components_per_column, 'iteration': int(iteration), 'init_albedo': getattr(columns, 'init_albedo', 0.0), 'scale_lower_bounds': scale_lower_bounds, 'adam_steps': {}, **ColumnParameterStats(columns)}
    for name, (_, _, _, extra) in optimizer.state.items():
        step, first_moment, second_moment, *rest = extra
        metadata['adam_steps'][name] = int(np.asarray(step).reshape(-1)[0])
        arrays[f'{name}__first_moment'] = np.asarray(first_moment)
        arrays[f'{name}__second_moment'] = np.asarray(second_moment)
        if rest and rest[0] is not None:
            arrays[f'{name}__maximum_moment'] = np.asarray(rest[0])
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    temporary_path = checkpoint_dir / 'columns.tmp.npz'
    np.savez(temporary_path, **arrays)
    temporary_path.replace(checkpoint_dir / 'columns.npz')
    (checkpoint_dir / 'column_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def LoadColumns(checkpoint_path, args):
    path = Path(checkpoint_path)
    candidates = (path, path / 'columns.npz', path / 'npy_data' / 'columns.npz', path / 'optimized_asset_pyr0' / 'npy_data' / 'columns.npz')
    path = next((candidate for candidate in candidates if candidate.is_file() and candidate.suffix == '.npz'), None)
    if path is None:
        raise FileNotFoundError('Column checkpoint columns.npz is required; expanded Gaussian assets do not preserve sharing')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata.get('format') != 'gaussian_columns_v1':
            raise ValueError('Unsupported column checkpoint format')
        parameters = {name: archive[name].copy() for name in PARAMETER_NAMES}
        columns_count = parameters['center_x'].size
        axial_count = columns_count * metadata['components_per_column']
        for name in PARAMETER_NAMES:
            expected_count = axial_count if name in ('axial_centers', 'axial_scales', 'density_weights') else columns_count
            if parameters[name].size != expected_count or not np.isfinite(parameters[name]).all():
                raise ValueError(f'Invalid column checkpoint parameter {name}')
        optimizer, learning_rates = CreateColumnOptimizer(parameters, args, metadata.get('scale_lower_bounds'))
        for name in PARAMETER_NAMES:
            value, promoted, learning_rate, extra = optimizer.state[name]
            step, first_moment, second_moment, *rest = extra
            if f'{name}__first_moment' not in archive:
                continue
            restored_rest = [type(rest[0])(archive[f'{name}__maximum_moment'])] if rest and rest[0] is not None and f'{name}__maximum_moment' in archive else rest
            optimizer.state[name] = (value, promoted, learning_rate, (type(step)(metadata['adam_steps'][name]), type(first_moment)(archive[f'{name}__first_moment']), type(second_moment)(archive[f'{name}__second_moment']), *restored_rest))
    columns = GaussianColumn(optimizer, metadata['components_per_column'])
    columns.init_albedo = float(metadata.get('init_albedo', 0.0))
    return columns, learning_rates, int(metadata.get('iteration', 0))


def SaveColumnSlices(columns, volume, volume_config, output_path, label='final'):
    if volume_config.opacity_scale <= 0.0:
        raise ValueError('Column density diagnostics require a positive opacity_scale')
    output_dir = Path(output_path) / 'column_slices'
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {name: np.asarray(columns.column_optimizer[name], dtype=np.float64).reshape(-1) for name in PARAMETER_NAMES}
    matrix = np.asarray(volume_config.to_world.matrix, dtype=np.float64)
    occupied_z = np.flatnonzero(np.any(np.abs(volume) > 1e-6, axis=(1, 2)))
    lower, upper = (int(occupied_z[0]), int(occupied_z[-1])) if occupied_z.size else (0, volume.shape[0] - 1)
    slice_indices = np.unique(np.linspace(lower, upper, 3).round().astype(int))
    pixel_y, pixel_x = np.meshgrid(np.arange(volume.shape[1]), np.arange(volume.shape[2]), indexing='ij')
    world_x = matrix[0, 0] * ((pixel_x + 0.5) / volume.shape[2]) + matrix[0, 3]
    world_y = matrix[1, 1] * ((pixel_y + 0.5) / volume.shape[1]) + matrix[1, 3]
    intensity_peak = max(float(np.max(volume)), 1e-8)
    panels, metrics, squared_error_sum, occupied_squared_error_sum = [], [], 0.0, 0.0
    evaluate_full_volume = volume.size <= 2_000_000
    footprints = []
    for column_index in range(parameters['center_x'].size):
        offset_x, offset_y = world_x - parameters['center_x'][column_index], world_y - parameters['center_y'][column_index]
        cosine, sine = np.cos(parameters['rotation_angle'][column_index]), np.sin(parameters['rotation_angle'][column_index])
        scale_x, scale_y = parameters['scale_x'][column_index], parameters['scale_y'][column_index]
        footprints.append(np.exp(-0.5 * (((cosine * offset_x + sine * offset_y) / scale_x) ** 2 + ((-sine * offset_x + cosine * offset_y) / scale_y) ** 2)) / (2.0 * np.pi * scale_x * scale_y))
    evaluated_slices = range(volume.shape[0]) if evaluate_full_volume else slice_indices
    for slice_index in evaluated_slices:
        world_z = matrix[2, 2] * ((slice_index + 0.5) / volume.shape[0]) + matrix[2, 3]
        reconstructed = np.zeros_like(world_x)
        for column_index in range(parameters['center_x'].size):
            start = column_index * columns.components_per_column
            indices = slice(start, start + columns.components_per_column)
            profile = np.sum(parameters['density_weights'][indices] * np.exp(-0.5 * ((world_z - parameters['axial_centers'][indices]) / parameters['axial_scales'][indices]) ** 2) / (np.sqrt(2.0 * np.pi) * parameters['axial_scales'][indices]))
            reconstructed += footprints[column_index] * profile / volume_config.opacity_scale
        target = np.asarray(volume[slice_index], dtype=np.float64)
        error = np.abs(reconstructed - target)
        mse = float(np.mean(error ** 2))
        squared_error_sum += mse
        if lower <= slice_index <= upper:
            occupied_squared_error_sum += mse
        metrics.append({'z_index': int(slice_index), 'mse': mse, 'psnr_db': float(-10.0 * np.log10(mse / intensity_peak ** 2)) if mse > 0.0 else None})
        if slice_index not in slice_indices:
            continue
        for name, pixels, maximum in (('target', target, intensity_peak), ('reconstructed', reconstructed, intensity_peak), ('absolute_error', error, intensity_peak)):
            image = Image.fromarray(np.uint8(np.clip(pixels / maximum, 0.0, 1.0) * 255.0))
            image.save(output_dir / f'{label}__z{slice_index:04d}__{name}.png')
            panels.append((f'z={slice_index}: {name}', image))
    panel_size = 256
    contact = Image.new('RGB', (3 * panel_size, len(slice_indices) * (panel_size + 25)), 'white')
    draw = ImageDraw.Draw(contact)
    for panel_index, (title, image) in enumerate(panels):
        position_x, position_y = (panel_index % 3) * panel_size, (panel_index // 3) * (panel_size + 25)
        draw.text((position_x + 4, position_y + 5), title, fill='black')
        contact.paste(image.resize((panel_size, panel_size)), (position_x, position_y + 25))
    contact.save(output_dir / f'{label}__slice_comparison.png')
    overall_mse = squared_error_sum / volume.shape[0] if evaluate_full_volume else None
    occupied_z_mse = occupied_squared_error_sum / (upper - lower + 1) if evaluate_full_volume else None
    summary = {'label': label, 'full_volume_metrics_computed': evaluate_full_volume, 'full_volume_metrics_skip_reason': None if evaluate_full_volume else 'Volume exceeds 2,000,000 voxels; only three representative slices evaluated.', 'overall_mse': overall_mse, 'overall_psnr_db': float(-10.0 * np.log10(overall_mse / intensity_peak ** 2)) if overall_mse is not None and overall_mse > 0.0 else None, 'occupied_z_mse': occupied_z_mse, 'occupied_z_psnr_db': float(-10.0 * np.log10(occupied_z_mse / intensity_peak ** 2)) if occupied_z_mse is not None and occupied_z_mse > 0.0 else None, 'occupied_z_range': [lower, upper + 1], 'slice_metrics': metrics, 'density_convention': 'sum(weight * normalized Gaussian pdf) / volume opacity_scale', 'support_convention': 'Analytic Gaussian tails are included in full; renderer uses finite ellipsoid AABBs and may differ near support boundaries.', 'display_maximum': intensity_peak, 'error_display_maximum': intensity_peak, **ColumnParameterStats(columns)}
    (output_dir / f'{label}__metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
