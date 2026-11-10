"""Shared-footprint Gaussian training helpers; rendering still uses expanded ellipsoids."""

import json
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
from PIL import Image, ImageDraw

from gfields.integrators.common import GaussianColumn

parameter_names = GaussianColumn.parameter_names


def save_columns(columns, output_path, iteration=0):
    checkpoint_dir = Path(output_path) / 'optimized_asset_pyr0' / 'npy_data'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = columns.column_optimizer
    arrays = {name: np.asarray(optimizer[name], dtype=np.float32) for name in parameter_names}
    scale_lower_bounds = {
        'column_scale_xy': np.asarray(optimizer.bounds['column_scale_xy'][1]).reshape(2).tolist(),
        'primitive_scale_z': float(optimizer.bounds['primitive_scale_z'][1]),
    }

    metadata = {
        'format': 'gaussian_columns_v2',
        'primitives_per_column': columns.primitives_per_column,
        'iteration': int(iteration),
        'init_albedo': getattr(columns, 'init_albedo', 0.0),
        'scale_lower_bounds': scale_lower_bounds,
        'adam_steps': {},
        **columns.parameter_stats()
    }

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


def save_column_slices(columns, volume, volume_config, output_path, label='final'):
    if volume_config.opacity_scale <= 0.0:
        raise ValueError('Column density diagnostics require a positive opacity_scale')

    output_dir = Path(output_path) / 'column_slices'
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = {name: np.asarray(columns.parameters[name], dtype=np.float64) for name in parameter_names}
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
    for column_index in range(parameters['column_center_xy'].shape[1]):
        offset_x, offset_y = world_x - parameters['column_center_xy'][0, column_index], world_y - parameters['column_center_xy'][1, column_index]
        cosine, sine = np.cos(parameters['column_quaternions'][column_index]), np.sin(parameters['column_quaternions'][column_index])
        column_scale_x, column_scale_y = parameters['column_scale_xy'][0, column_index], parameters['column_scale_xy'][1, column_index]
        footprints.append(np.exp(-0.5 * (((cosine * offset_x + sine * offset_y) / column_scale_x) ** 2 + ((-sine * offset_x + cosine * offset_y) / column_scale_y) ** 2)) / (2.0 * np.pi * column_scale_x * column_scale_y))

    evaluated_slices = range(volume.shape[0]) if evaluate_full_volume else slice_indices
    for slice_index in evaluated_slices:
        world_z = matrix[2, 2] * ((slice_index + 0.5) / volume.shape[0]) + matrix[2, 3]
        reconstructed = np.zeros_like(world_x)

        for column_index in range(parameters['column_center_xy'].shape[1]):
            start = column_index * columns.primitives_per_column
            indices = slice(start, start + columns.primitives_per_column)
            profile = np.sum(parameters['primitive_opacity'][indices] * np.exp(-0.5 * ((world_z - parameters['primitive_center_z'][indices]) / parameters['primitive_scale_z'][indices]) ** 2) / (np.sqrt(2.0 * np.pi) * parameters['primitive_scale_z'][indices]))
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

    summary = {
        'label': label,
        'full_volume_metrics_computed': evaluate_full_volume,
        'full_volume_metrics_skip_reason': None if evaluate_full_volume else 'Volume exceeds 2,000,000 voxels; only three representative slices evaluated.',
        'overall_mse': overall_mse,
        'overall_psnr_db': float(-10.0 * np.log10(overall_mse / intensity_peak ** 2)) if overall_mse is not None and overall_mse > 0.0 else None,
        'occupied_z_mse': occupied_z_mse,
        'occupied_z_psnr_db': float(-10.0 * np.log10(occupied_z_mse / intensity_peak ** 2)) if occupied_z_mse is not None and occupied_z_mse > 0.0 else None,
        'occupied_z_range': [lower, upper + 1],
        'slice_metrics': metrics,
        'density_convention': 'sum(weight * normalized Gaussian pdf) / volume opacity_scale',
        'support_convention': 'Analytic Gaussian tails are included in full; renderer uses finite ellipsoid AABBs and may differ near support boundaries.',
        'display_maximum': intensity_peak,
        'error_display_maximum': intensity_peak,
        **columns.parameter_stats()
    }

    (output_dir / f'{label}__metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
