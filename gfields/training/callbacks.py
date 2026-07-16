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


class MetricsAccumulatorCallback(CallbackABC):
    """Accumulates per-iteration scalar metrics and dumps plots + ``.npy`` files."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    def on_suspend_grad(self, epoch: int, opt, metrics: Dict[str, float]) -> None:  # noqa: ANN001
        import numpy as np

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
            plot_loss(values, label=metric, output_file=self.args.output / f"{metric.lower()}_pyr{self.level}.png")
            np.save(self.args.output / f"{metric.lower()}_pyr{self.level}.npy", np.array(values))

            if ref_path.exists() and (npy_file := ref_path / f"{metric.lower()}_pyr{self.level}.npy").exists():
                ref = np.load(npy_file)
                plt.plot(range(1, len(ref) + 1), ref, label="baseline", linestyle="--")
                plt.plot(range(1, len(values) + 1), values, label="current")
                plt.title(f"Comparison of {metric} with baseline")
                plt.legend()
                plt.savefig(self.args.output / f"{metric.lower()}_comparison_pyr{self.level}.png")
                plt.close()


class PlotCallback(CallbackABC):
    """Periodic histogram plots of opacities / scales / density / Gabor frequency."""

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

        density: mi.Float = opt[f"opacities_{self.level}"] / (
            dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(opt[f"scales_{self.level}"])
        )
        self._make_histogram(epoch + 1, "Density", density.numpy(), bins=100, range=(-5, 5))
        self._make_histogram(
            epoch + 1,
            "opacities",
            opt[f"opacities_{self.level}"].numpy().reshape(-1),
            bins=100,
            range=(-0.025, 0.025),
        )
        self._make_histogram(
            epoch + 1,
            "scales",
            opt[f"scales_{self.level}"].numpy().reshape(-1),
            bins=100,
            range=(0, 0.4),
        )

    def _make_histogram(self, epoch: int, name: str, data, **kwargs: Any) -> None:
        import matplotlib.pyplot as plt

        plt.hist(data, **kwargs, density=True)
        plt.title(f"{name} Histogram it={epoch:04d}")
        (self.args.output / f"stats_{self.level}").mkdir(exist_ok=True)
        plt.savefig(self.args.output / f"stats_{self.level}" / f"{name}_histogram_pyr{self.level}_{epoch:04d}.png")
        plt.close()


__all__ = [
    "CallbackABC",
    "Callbacks",
    "GradientTrackerCallback",
    "MetricsAccumulatorCallback",
    "PlotCallback",
]
