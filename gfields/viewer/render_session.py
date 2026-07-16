# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Progressive-rendering state machine for the interactive viewer.

A :class:`RenderSession` owns a Mitsuba scene, accumulates render samples one
at a time while the view and parameters are unchanged, and resets the
accumulator the moment anything in the state key changes (camera, resolution,
runtime integrator parameters, environment pose). Runtime parameters flow
through ``mi.traverse`` so the underlying kernels are reused; the render call
is additionally wrapped in ``dr.freeze`` when the integrator supports it, so
steady-state accumulation replays a frozen kernel without retracing.

Outer ``dr.freeze`` around ``mi.render`` is **disabled by default** (``try_freeze=False``).
With the gfields integrators on Mitsuba 3.9 it aborts the process with an
uncatchable ``jit_var: unknown variable`` error at record time — the
``try/except`` fallback cannot recover. Ordinary rendering still reuses
compiled kernels through Dr.Jit's kernel cache, so interactive accumulation
remains fast without freeze.

When freeze is enabled, the scene and its parameter set are passed as *explicit
arguments* to the frozen function (rather than captured from a closure):
``dr.freeze`` only traverses its declared inputs, so a captured
``SceneParameters`` leaks a variable and aborts recording with "created before
recording was started".

Per-frame GPU kernel execution time is measured through Dr.Jit's kernel
history (``dr.JitFlag.KernelHistory``) and reported as a smoothed cost per
sample, mirroring the timings produced by ``gfields.benchmark``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import drjit as dr
import mitsuba as mi


class RenderSession:
    """Accumulating renderer with dirty-state tracking and per-sample timing."""

    def __init__(self, scene: mi.Scene, *, max_spp: int = 1024,
                 first_chunk_spp: int = 1, chunk_spp: int = 1,
                 try_freeze: bool = False, measure_cost: bool = True,
                 accumulate_tau: bool = False,
                 linear_tau_samples: bool = True,
                 emitter_scale: float = 1.0):
        self.scene = scene
        self.params = mi.traverse(scene)
        self.max_spp = max_spp
        # Stochastic tomography must average τ per sample; never let Mitsuba
        # average transmittance inside a multi-spp chunk.
        self.accumulate_tau = bool(accumulate_tau)
        self.linear_tau_samples = bool(linear_tau_samples)
        self.emitter_scale = float(emitter_scale)
        if self.accumulate_tau:
            first_chunk_spp = 1
            chunk_spp = 1
        self.first_chunk_spp = first_chunk_spp
        self.chunk_spp = chunk_spp
        self.measure_cost = measure_cost and hasattr(dr, 'kernel_history')

        # Discover runtime-tunable parameter keys
        self.freq_key = self._find_key('freq_threshold')
        self.opacity_keys = sorted(
            k for k in self.params.keys() if k.endswith('.opacities'))
        self.albedo_keys = sorted(
            k for k in self.params.keys() if k.endswith('.albedo'))
        self._opacity_base: dict[str, np.ndarray] = {}
        self._albedo_base: dict[str, np.ndarray] = {}
        self._opacity_weight = float('nan')
        self._albedo_scale = float('nan')
        self.envmap_to_world_key = None
        self.envmap_scale_key = None
        for key in self.params.keys():
            if key.endswith('.to_world') and ('env' in key.lower() or 'emitter' in key.lower()):
                self.envmap_to_world_key = key
            if key.endswith('.scale') and ('env' in key.lower() or 'emitter' in key.lower()):
                self.envmap_scale_key = key

        self._accum: Optional[np.ndarray] = None
        self._spp_done = 0
        self._seed = 0
        self._state_key: Any = None
        self._sensor: Optional[mi.Sensor] = None
        self._sensor_state = None

        # Per-sample cost (smoothed GPU kernel execution time, milliseconds)
        self._last_exec_ms = 0.0
        self._ms_per_sample = 0.0

        if self.measure_cost:
            dr.set_flag(dr.JitFlag.KernelHistory, True)

        self._render = self._make_render_fn(try_freeze)

    # ------------------------------------------------------------- helpers

    def _find_key(self, suffix: str):
        for key in self.params.keys():
            if key.endswith(suffix):
                return key
        return None

    def _make_render_fn(self, try_freeze: bool):
        # Scene + params are explicit arguments so ``dr.freeze`` traverses them
        # as inputs (a closure-captured ``SceneParameters`` leaks a variable and
        # aborts recording). ``seed`` is a Dr.Jit variable (see ``step``) so a
        # new seed each frame replays rather than retraces.
        def render(scene, params, sensor, seed, spp):
            return mi.render(scene, params, sensor=sensor, seed=seed, spp=spp)

        if not try_freeze or not hasattr(dr, 'freeze'):
            return lambda sensor, seed, spp: render(
                self.scene, self.params, sensor, seed, spp)

        frozen = dr.freeze(render)
        state = {'use_frozen': True}

        def render_maybe_frozen(sensor, seed, spp):
            if state['use_frozen']:
                try:
                    return frozen(self.scene, self.params, sensor, seed, spp)
                except Exception as exc:  # noqa: BLE001 - fall back once, keep going
                    print(f'[viewer] dr.freeze unavailable for this integrator '
                          f'({type(exc).__name__}); falling back to cached-kernel '
                          f'rendering.')
                    state['use_frozen'] = False
            return render(self.scene, self.params, sensor, seed, spp)

        return render_maybe_frozen

    # ----------------------------------------------------------- parameters

    def set_runtime_param(self, key: Optional[str], value) -> bool:
        """Write a traversable parameter; returns True when it changed."""
        if key is None or key not in self.params.keys():
            return False
        self.params[key] = value
        self.params.update()
        return True

    def seed_opacity_weight(self, weight: float,
                            base_levels: list[dict[str, Any]]) -> None:
        """Record opacity weight baked into the scene at build time."""
        self._opacity_weight = float(weight)
        if not self._opacity_base:
            for i, key in enumerate(self.opacity_keys):
                if i < len(base_levels):
                    self._opacity_base[key] = np.asarray(
                        base_levels[i]['opacities'], dtype=np.float32)

    def seed_albedo_scale(self, scale: float,
                          base_levels: list[dict[str, Any]]) -> None:
        """Record albedo scale baked into the scene at build time."""
        self._albedo_scale = float(scale)
        if not self._albedo_base:
            for i, key in enumerate(self.albedo_keys):
                if i < len(base_levels):
                    self._albedo_base[key] = np.asarray(
                        base_levels[i]['albedo'], dtype=np.float32)

    def set_opacity_weight(self, weight: float,
                           base_levels: list[dict[str, Any]]) -> None:
        """Scale primitive opacities from unscaled *base_levels* numpy data."""
        weight = float(weight)
        if not self.opacity_keys or weight == self._opacity_weight:
            return
        changed = False
        for i, key in enumerate(self.opacity_keys):
            if i >= len(base_levels):
                continue
            if key not in self.params:
                continue
            if key not in self._opacity_base:
                self._opacity_base[key] = np.asarray(
                    base_levels[i]['opacities'], dtype=np.float32)
            base = self._opacity_base[key]
            target_shape = np.array(self.params[key], dtype=np.float32).shape
            scaled = (np.asarray(base, dtype=np.float32).reshape(-1) * weight
                      ).reshape(target_shape)
            self.params[key] = mi.TensorXf(
                np.ravel(scaled), shape=tuple(int(s) for s in target_shape))
            changed = True
        if changed:
            self.params.update()
            self._opacity_weight = weight

    def set_albedo_scale(self, scale: float,
                         base_levels: list[dict[str, Any]],
                         *, absolute: bool = False) -> None:
        """Set primitive albedo.

        ``absolute=False`` (default): multiply the unscaled *base_levels* albedo
        by ``scale`` (used by tomography, which ignores albedo anyway).

        ``absolute=True``: set every primitive's albedo to the constant ``scale``,
        ignoring the stored values. **Required for multiple scattering**: most
        assets are density-only (optimized for tomography) and store
        ``albedo == 0``, so a multiplicative scale yields ``0·s = 0`` -> a pure
        absorber -> a black volume against the lit background. Mirrors the
        headless renderer's absolute ``--albedo`` for MS.
        """
        scale = float(scale)
        key_state = (bool(absolute), scale)
        if not self.albedo_keys or key_state == self._albedo_scale:
            return
        changed = False
        for i, key in enumerate(self.albedo_keys):
            if key not in self.params:
                continue
            if key not in self._albedo_base and i < len(base_levels):
                self._albedo_base[key] = np.asarray(
                    base_levels[i]['albedo'], dtype=np.float32)
            target_shape = np.array(self.params[key], dtype=np.float32).shape
            if absolute:
                scaled = np.full(target_shape, scale, dtype=np.float32)
            else:
                base = self._albedo_base.get(key)
                if base is None:
                    continue
                base_arr = np.asarray(base, dtype=np.float32).reshape(-1, 3) * scale
                if base_arr.size != int(np.prod(target_shape)):
                    # The stored base no longer aligns 1:1 with the scene params
                    # (LOD band-pruning drops kernels from a level, so the built
                    # scene has fewer primitives than the asset's base_levels).
                    # Albedo is cosmetic for tomography, so fill the current
                    # params with the mean base albedo * scale instead of crashing.
                    fill = float(np.mean(base_arr)) if base_arr.size else scale
                    scaled = np.full(target_shape, fill, dtype=np.float32)
                else:
                    scaled = base_arr.reshape(target_shape)
            self.params[key] = mi.TensorXf(
                np.ravel(scaled), shape=tuple(int(s) for s in target_shape))
            changed = True
        if changed:
            self.params.update()
            self._albedo_scale = key_state

    def set_camera(self, origin, target, up, fov: float,
                   width: int, height: int, *,
                   camera_mode: str = 'orthographic',
                   perspective_fov: float | None = None,
                   ortho_scale: float | None = None) -> None:
        origin = np.asarray(origin, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        if perspective_fov is None:
            perspective_fov = float(fov)
        if ortho_scale is None:
            dist = float(max(np.linalg.norm(origin - target), 1e-6))
            ortho_scale = dist * np.tan(np.deg2rad(float(fov)) / 2.0)
        ortho_scale = float(ortho_scale)

        state = (
            tuple(np.round(origin, 6)), tuple(np.round(target, 6)),
            tuple(np.round(up, 6)), camera_mode,
            round(ortho_scale, 5), round(float(perspective_fov), 3),
            int(width), int(height),
        )
        if state == self._sensor_state:
            return
        self._sensor_state = state
        T = mi.ScalarTransform4f
        to_world = T().look_at(origin=[float(c) for c in origin],
                               target=[float(c) for c in target],
                               up=[float(c) for c in up])
        film = {'type': 'hdrfilm', 'width': width, 'height': height,
                'pixel_filter': {'type': 'box'}}
        sampler = {'type': 'independent'}
        if camera_mode == 'perspective':
            self._sensor = mi.load_dict({
                'type': 'perspective',
                'fov': float(perspective_fov),
                'fov_axis': 'y',
                'near_clip': 0.01,
                'far_clip': 1e5,
                'to_world': to_world,
                'film': film,
                'sampler': sampler,
            })
        else:
            self._sensor = mi.load_dict({
                'type': 'orthographic',
                'to_world': to_world @ T().scale(
                    [ortho_scale, ortho_scale, 1.0]),
                'near_clip': 0.01,
                'far_clip': 1e5,
                'film': film,
                'sampler': sampler,
            })

    # ----------------------------------------------------------- main loop

    def update_state(self, runtime_state: Any) -> None:
        """Combine the sensor and runtime state into the dirty key; reset
        accumulation when anything changed."""
        key = (self._sensor_state, runtime_state)
        if key != self._state_key:
            self._state_key = key
            self._accum = None
            self._spp_done = 0

    @property
    def spp_done(self) -> int:
        return self._spp_done

    @property
    def converged(self) -> bool:
        return self._spp_done >= self.max_spp

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._accum

    @property
    def last_exec_ms(self) -> float:
        """GPU kernel execution time of the most recent rendered chunk (ms)."""
        return self._last_exec_ms

    @property
    def ms_per_sample(self) -> float:
        """Smoothed GPU kernel execution time per sample-per-pixel (ms)."""
        return self._ms_per_sample

    def _record_cost(self, spp: int) -> None:
        if not self.measure_cost:
            return
        # kernel_history() returns and drains the kernels launched since the
        # previous call; the image readback in ``step`` already forced a sync,
        # so all execution times are available (CUDA-event based, no launch
        # blocking required).
        history = dr.kernel_history([dr.KernelType.JIT])
        exec_ms = float(sum(k.get('execution_time', 0.0) for k in history))
        self._last_exec_ms = exec_ms
        per_sample = exec_ms / max(spp, 1)
        # Exponential moving average keeps the readout stable frame-to-frame.
        if self._ms_per_sample <= 0.0:
            self._ms_per_sample = per_sample
        else:
            self._ms_per_sample = 0.85 * self._ms_per_sample + 0.15 * per_sample

    def step(self) -> Optional[np.ndarray]:
        """Render one chunk and fold it into the accumulator."""
        if self._sensor is None or self.converged:
            return self._accum
        spp = self.first_chunk_spp if self._accum is None else self.chunk_spp
        spp = min(spp, self.max_spp - self._spp_done)
        self._seed += 1
        if self.measure_cost:
            dr.kernel_history()  # drain anything pending so timing is per-frame
        img = self._render(self._sensor, mi.UInt32(self._seed), spp)
        arr = np.array(img, dtype=np.float32)[..., :3]  # forces device sync
        self._record_cost(spp)
        if self.accumulate_tau:
            from .sample_accum_helpers import film_sample_to_tau
            tau = film_sample_to_tau(
                arr,
                linear_tau=self.linear_tau_samples,
                emitter_scale=self.emitter_scale,
            ).astype(np.float32)
            if self._accum is None:
                self._accum = tau
            else:
                n = self._spp_done
                self._accum = (self._accum * n + tau * spp) / (n + spp)
        elif self._accum is None:
            self._accum = arr
        else:
            n = self._spp_done
            self._accum = (self._accum * n + arr * spp) / (n + spp)
        self._spp_done += spp
        return self._accum
