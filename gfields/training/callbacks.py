# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Composable callback container for the training loop.

Each callback hooks into stages of the optimization loop:

    setup()                  -- once before the loop
    on_before_opt_step(...)  -- per iteration, before opt.step()
    on_suspend_grad(...)     -- per iteration, after metrics are computed
    on_optim_end()           -- once after the loop
"""
from __future__ import annotations

from abc import ABC
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


class CallbackABC(ABC):
    args: Namespace
    level: int

    def setup(self) -> None:
        pass

    def on_before_opt_step(self, epoch: int, opt) -> None:  # noqa: ANN001 - mi.ad.Optimizer
        pass

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        pass

    def on_optim_end(self) -> None:
        pass


class Callbacks(CallbackABC):
    """Fan-out container for a list of :class:`CallbackABC`."""

    def __init__(self, args: Namespace, level: int) -> None:
        self.args = args
        self.level = level
        self.callbacks: List[CallbackABC] = []

    def register_callbacks(self, callbacks: List[CallbackABC]) -> None:
        for cb in callbacks:
            cb.args = self.args
            cb.level = self.level
        self.callbacks.extend(callbacks)

    def setup(self) -> None:
        for cb in self.callbacks:
            cb.setup()

    def on_before_opt_step(self, epoch: int, opt) -> None:  # noqa: ANN001
        for cb in self.callbacks:
            cb.on_before_opt_step(epoch, opt)

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        for cb in self.callbacks:
            cb.on_suspend_grad(epoch, opt, metrics)

    def on_optim_end(self) -> None:
        for cb in self.callbacks:
            cb.on_optim_end()


class GradientTrackerCallback(CallbackABC):
    """Tracks gradient statistics across iterations and writes per-level plots."""

    def __init__(self) -> None:
        from .regression import GradientTracker  # local import avoids circularity
        self.gradient_tracker = GradientTracker()

    def setup(self) -> None:
        self.gradient_tracker.track_grads = self.args.track_grads

    def on_before_opt_step(self, epoch: int, opt) -> None:  # noqa: ANN001
        self.gradient_tracker.track(opt)

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        if (epoch + 1) % 10 == 0:
            self.gradient_tracker.plot_statistics(self.args.output / f"stats_{self.level}")

    def on_optim_end(self) -> None:
        self.gradient_tracker.plot_statistics(self.args.output / f"stats_{self.level}")


def export_npy_to_txt(output_path: Path) -> None:
    """Mirror numeric NPY files as text, preserving relative paths and shapes."""
    import numpy as np

    output_path = Path(output_path)
    for path in output_path.rglob('*.npy'):
        values = np.load(path, allow_pickle=False)
        target = output_path / 'npy_txt' / path.relative_to(output_path).with_suffix('.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = values.reshape(-1, values.shape[-1]) if values.ndim > 1 and values.shape[-1] else values.reshape(-1, 1)
        np.savetxt(target, rows, fmt='%.18e', header=f'shape: {values.shape}')


class MetricsAccumulatorCallback(CallbackABC):
    """Accumulates per-iteration scalar metrics and dumps plots + ``.npy`` files."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.iterations: List[int] = []

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        import numpy as np

        self.iterations.append(epoch + 1)
        for key, value in metrics.items():
            if not isinstance(value, float):
                raise TypeError(f"Expected float metric '{key}', got {type(value)}")
            self.metrics[key].append(value)
        if (epoch + 1) % 10 == 0:
            self.on_optim_end()

    def on_optim_end(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        from .regression import plot_loss

        ref_path = Path(".") / "hierarchical_gabor_outputs" / "bunny_32x300"
        for metric, values in self.metrics.items():
            plot_loss(values, label=metric, output_file=self.args.output / f"{metric.lower()}_pyr{self.level}.png", iterations=self.iterations)
            np.save(self.args.output / f"{metric.lower()}_pyr{self.level}.npy", np.array(values))

            if ref_path.exists() and (npy_file := ref_path / f"{metric.lower()}_pyr{self.level}.npy").exists():
                ref = np.load(npy_file)
                plt.plot(range(1, len(ref) + 1), ref, label="baseline", linestyle="--")
                plt.plot(range(1, len(values) + 1), values, label="current")
                plt.title(f"Comparison of {metric} with baseline")
                plt.legend()
                plt.savefig(self.args.output / f"{metric.lower()}_comparison_pyr{self.level}.png")
                plt.close()

        export_npy_to_txt(self.args.output)


class PlotCallback(CallbackABC):
    """Periodic histogram plots of opacities / scales / density / Gabor frequency."""

    def __init__(self, columns=None) -> None:
        self.columns = columns

    def _column_histogram_data(self):
        import numpy as np

        # Count shared footprint scales once before expansion.
        parameters = self.columns.parameters
        shared_scales = parameters['column_scale_xz'].numpy()
        axial_scales = parameters['primitive_scale_y'].numpy()
        opacity = parameters['primitive_opacity'].numpy()
        area = np.repeat(np.prod(shared_scales, axis=0), self.columns.primitives_per_column)
        density = opacity / ((2.0 * np.pi) ** 1.5 * area * axial_scales)
        before = {'opacities': opacity, 'scales': np.concatenate((shared_scales.reshape(-1), axial_scales)),
                  'column_scale_xz': shared_scales.reshape(-1), 'primitive_scale_y': axial_scales,
                  'density': density}

        # Use expanded primitive geometry for the matching Gaussian histograms.
        _, scales, _, expanded_opacity = self.columns.expand()
        expanded_scales = scales.numpy()
        expanded_opacity = expanded_opacity.numpy()
        after = {'opacities': expanded_opacity, 'scales': expanded_scales.reshape(-1),
                 'density': expanded_opacity / ((2.0 * np.pi) ** 1.5 * np.prod(expanded_scales, axis=0))}
        return before, after

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        import drjit as dr
        import mitsuba as mi

        if (epoch + 1) % 10 in (0, 1) and self.level == 1:
            b: mi.Float = dr.norm(opt["omega_1"] / opt["scales_1"])
            alive_indices = dr.compress(opt["opacities_1"] > 1e-6)
            self._make_histogram(epoch + 1, "b", b.numpy(), bins=100, range=(0, 1000))
            om = dr.gather(mi.Float, opt["omega_1"], alive_indices)
            self._make_histogram(epoch + 1, "omega_1", om.numpy(), bins=100, range=(0, 3.2))

        if (epoch + 1) % 10 != 0 and epoch != 0:
            return

        if getattr(self.args, 'use_columns_representation', False) and self.level == 0:
            for stage, values in zip(('before_expansion', 'after_expansion'), self._column_histogram_data()):
                for name, data in values.items():
                    self._make_histogram(epoch + 1, name, data, stage=stage, bins=100)
            return

        density: mi.Float = opt[f"opacities_{self.level}"] / (
            dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(opt[f"scales_{self.level}"])
        )
        self._make_histogram(epoch + 1, "Density", density.numpy(), bins=100)
        self._make_histogram(
            epoch + 1,
            "opacities",
            opt[f"opacities_{self.level}"].numpy().reshape(-1),
            bins=100,
        )
        self._make_histogram(
            epoch + 1,
            "scales",
            opt[f"scales_{self.level}"].numpy().reshape(-1),
            bins=100,
        )

    def _make_histogram(self, epoch: int, name: str, data, stage=None, **kwargs: Any) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        data = np.asarray(data).reshape(-1)
        data = data[np.isfinite(data)]
        if not data.size:
            return
        plt.figure()
        plt.hist(data, **kwargs, density=True)
        plt.title(f"{name} Histogram it={epoch:04d}")
        output = self.args.output / f"stats_{self.level}"
        if stage is not None:
            output /= stage
        output.mkdir(parents=True, exist_ok=True)
        plt.savefig(output / f"{name}_histogram_pyr{self.level}_{epoch:04d}.png")
        plt.close()


class ColumnSlicesCallback(CallbackABC):
    """Save column density slices once at the end of training."""

    def __init__(self, columns, volume, volume_config) -> None:
        self.columns = columns
        self.volume = volume
        self.volume_config = volume_config

    def on_optim_end(self) -> None:
        import json
        import numpy as np

        columns = self.columns
        volume = self.volume
        volume_config = self.volume_config
        output_path = self.args.output
        label = 'final'
        # Import image utilities only when generating slice diagnostics.
        from PIL import Image, ImageDraw

        if volume_config.opacity_scale <= 0.0:
            raise ValueError('Column density diagnostics require a positive opacity_scale')

        output_dir = Path(output_path) / 'column_slices'
        output_dir.mkdir(parents=True, exist_ok=True)

        parameters = {name: np.asarray(columns.parameters[name], dtype=np.float64) for name in columns.parameter_names}
        matrix = np.asarray(volume_config.to_world.matrix, dtype=np.float64)
        # Compare slices perpendicular to Y, preserving the original world transform.
        volume = np.swapaxes(volume, 0, 1)
        matrix = matrix[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]
        occupied_y = np.flatnonzero(np.any(np.abs(volume) > 1e-6, axis=(1, 2)))
        lower, upper = (int(occupied_y[0]), int(occupied_y[-1])) if occupied_y.size else (0, volume.shape[0] - 1)
        slice_indices = np.unique(np.linspace(lower, upper, 3).round().astype(int))

        pixel_y, pixel_x = np.meshgrid(np.arange(volume.shape[1]), np.arange(volume.shape[2]), indexing='ij')
        world_x = matrix[0, 0] * ((pixel_x + 0.5) / volume.shape[2]) + matrix[0, 3]
        world_y = matrix[1, 1] * ((pixel_y + 0.5) / volume.shape[1]) + matrix[1, 3]

        intensity_peak = max(float(np.max(volume)), 1e-8)
        panels, metrics, squared_error_sum, occupied_squared_error_sum = [], [], 0.0, 0.0
        evaluate_full_volume = volume.size <= 2_000_000

        footprints = []
        for column_index in range(parameters['column_center_xz'].shape[1]):
            offset_x, offset_y = world_x - parameters['column_center_xz'][0, column_index], world_y - parameters['column_center_xz'][1, column_index]
            cosine, sine = np.cos(parameters['column_quaternions'][column_index]), -np.sin(parameters['column_quaternions'][column_index])
            column_scale_x, column_scale_y = parameters['column_scale_xz'][0, column_index], parameters['column_scale_xz'][1, column_index]
            footprints.append(np.exp(-0.5 * (((cosine * offset_x + sine * offset_y) / column_scale_x) ** 2 + ((-sine * offset_x + cosine * offset_y) / column_scale_y) ** 2)) / (2.0 * np.pi * column_scale_x * column_scale_y))

        evaluated_slices = range(volume.shape[0]) if evaluate_full_volume else slice_indices
        for slice_index in evaluated_slices:
            world_axis_y = matrix[2, 2] * ((slice_index + 0.5) / volume.shape[0]) + matrix[2, 3]
            reconstructed = np.zeros_like(world_x)

            for column_index in range(parameters['column_center_xz'].shape[1]):
                start = column_index * columns.primitives_per_column
                indices = slice(start, start + columns.primitives_per_column)
                profile = np.sum(parameters['primitive_opacity'][indices] * np.exp(-0.5 * ((world_axis_y - parameters['primitive_center_y'][indices]) / parameters['primitive_scale_y'][indices]) ** 2) / (np.sqrt(2.0 * np.pi) * parameters['primitive_scale_y'][indices]))
                reconstructed += footprints[column_index] * profile / volume_config.opacity_scale

            target = np.asarray(volume[slice_index], dtype=np.float64)
            error = np.abs(reconstructed - target)
            mse = float(np.mean(error ** 2))
            squared_error_sum += mse
            if lower <= slice_index <= upper:
                occupied_squared_error_sum += mse
            metrics.append({'y_index': int(slice_index), 'mse': mse, 'psnr_db': float(-10.0 * np.log10(mse / intensity_peak ** 2)) if mse > 0.0 else None})

            if slice_index not in slice_indices:
                continue
            for name, pixels, maximum in (('target', target, intensity_peak), ('reconstructed', reconstructed, intensity_peak), ('absolute_error', error, intensity_peak)):
                image = Image.fromarray(np.uint8(np.clip(pixels / maximum, 0.0, 1.0) * 255.0))
                image.save(output_dir / f'{label}__y{slice_index:04d}__{name}.png')
                panels.append((f'y={slice_index}: {name}', image))

        panel_size = 256
        contact = Image.new('RGB', (3 * panel_size, len(slice_indices) * (panel_size + 25)), 'white')
        draw = ImageDraw.Draw(contact)

        for panel_index, (title, image) in enumerate(panels):
            position_x, position_y = (panel_index % 3) * panel_size, (panel_index // 3) * (panel_size + 25)
            draw.text((position_x + 4, position_y + 5), title, fill='black')
            contact.paste(image.resize((panel_size, panel_size)), (position_x, position_y + 25))
        contact.save(output_dir / f'{label}__slice_comparison.png')

        overall_mse = squared_error_sum / volume.shape[0] if evaluate_full_volume else None
        occupied_y_mse = occupied_squared_error_sum / (upper - lower + 1) if evaluate_full_volume else None

        summary = {
            'label': label,
            'full_volume_metrics_computed': evaluate_full_volume,
            'full_volume_metrics_skip_reason': None if evaluate_full_volume else 'Volume exceeds 2,000,000 voxels; only three representative slices evaluated.',
            'overall_mse': overall_mse,
            'overall_psnr_db': float(-10.0 * np.log10(overall_mse / intensity_peak ** 2)) if overall_mse is not None and overall_mse > 0.0 else None,
            'occupied_y_mse': occupied_y_mse,
            'occupied_y_psnr_db': float(-10.0 * np.log10(occupied_y_mse / intensity_peak ** 2)) if occupied_y_mse is not None and occupied_y_mse > 0.0 else None,
            'occupied_y_range': [lower, upper + 1],
            'slice_metrics': metrics,
            'density_convention': 'sum(weight * normalized Gaussian pdf) / volume opacity_scale',
            'support_convention': 'Analytic Gaussian tails are included in full; renderer uses finite ellipsoid AABBs and may differ near support boundaries.',
            'display_maximum': intensity_peak,
            'error_display_maximum': intensity_peak,
            **columns.parameter_stats()
        }

        (output_dir / f'{label}__metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary


__all__ = [
    "CallbackABC",
    "Callbacks",
    "GradientTrackerCallback",
    "MetricsAccumulatorCallback",
    "PlotCallback",
    "ColumnSlicesCallback",
]
