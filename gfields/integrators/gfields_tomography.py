# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator
from .common import *
from gfields.lod import LODSelector, continuous_lod_weight


class GfieldsTomographyIntegrator(RBIntegrator):
    '''
    Absorption-only tomography integrator for volumetric primitives. Defaults
    reproduce the upstream `volprim_tomography`; options add stochastic
    level-of-detail (`pyramid_level_count`, `stochastic_selection_criteria`,
    `power_law_beta`), steerable orientation selection (`asset_encoding`,
    `num_orientations`, `orientation_selection`, `orientation_threshold`,
    `bin_rotation`), optical-depth training output (`output='optical_depth'`),
    a static `level_mask`, and continuous frequency LOD (`freq_threshold`,
    `freq_band_sigma`; exposed via `mi.traverse` for interactive use). Per-ray
    level selection lives in the overridable `_ray_mask()` hook.
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", 64))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces
        self._max_depth_opaque = dr.opaque(
            mi.UInt32, max_depth if max_depth != -1 else 0xFFFFFFFF)
        self.use_kernel_freeze = bool(props.get('use_kernel_freeze', False))

        # Kernel parameters required for the volumetric transmittance model
        props['kernel_full_range'] = props.get('kernel_full_range', True)
        props['kernel_normalized'] = props.get('kernel_normalized', True)
        self.kernel = Kernel.factory(props)
        self.kernel_full_range = bool(props.get('kernel_full_range', True))

        self.output = str(props.get('output', 'radiance'))
        assert self.output in ('radiance', 'optical_depth')
        self.render_marginalization = bool(props.get('render_marginalization', False))

        self.num_bounces_gaussian_ms = int(props.get('num_bounces_gaussian_ms', 0))

        level_mask = int(props.get('level_mask', 255))
        if not (0 < level_mask <= 255):
            raise Exception(f'"level_mask" must be in [1, 255], got {level_mask}')
        self.level_mask = level_mask

        # Level-of-detail selection
        self.lod = LODSelector(props, allow_dynamic_orientation=True)

        # Continuous frequency-domain LOD filter
        freq_threshold = float(props.get('freq_threshold', -1.0))
        self.enable_continuous_lod = freq_threshold >= 0.0
        self.freq_threshold = mi.Float(freq_threshold)
        self.freq_band_sigma = float(props.get('freq_band_sigma', 2.0))

        self.accumulate_beta = self.render_marginalization or self.output == 'optical_depth'
        if (self.lod.main_needs_weights()
                and self.output == 'radiance'
                and not self.render_marginalization):
            mi.Log(mi.LogLevel.Warn,
                   "[gfields_tomography] stochastic LOD forces "
                   "render_marginalization=True; average tau per pixel then apply "
                   "exp(-tau) (times emitter radiance) for display.")
            self.render_marginalization = True
            self.accumulate_beta = True
        self.activation_func = (lambda x: x) if self.accumulate_beta else (lambda x: dr.exp(-x))

        # Primal-only OptiX any-hit fast path
        self.use_anyhit = bool(props.get('use_anyhit', False))

    def traverse(self, callback):
        if self.enable_continuous_lod:
            callback.put('freq_threshold', self.freq_threshold,
                         mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    def _ray_mask(self, sampler, ray, active, depth=mi.UInt32(0)):
        '''
        Per-ray visibility mask: the static `level_mask` combined with the
        stochastic LOD selection. Override in subclasses for custom per-ray
        policies (e.g. gaze-driven foveated level selection).
        '''
        mask = mi.UInt32(self.level_mask)
        if self.lod.is_gauss_ms:
            self.lod.set_gauss_only(active & (
                depth >= mi.UInt32(self.num_bounces_gaussian_ms)))
        if self.lod.main_needs_weights():
            pyrs = sampler.next_1d()
            needs_ori = self.lod.orientation_selection not in ['deterministic', 'threshold']
            ori_s = sampler.next_1d() if needs_ori else mi.Float(0)
            mask = mask & self.lod.select_mask(pyrs, ori_s, ray.d, active)
        else:
            mask = mask & self.lod.select_mask(
                mi.Float(0.0), mi.Float(0.0), ray.d, active)
        return mask

    def eval_transmission(self, si, ray, sampler, active):
        '''
        Evaluate the transmission model on intersected volumetric primitives.
        '''
        ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active)

        # A Gaussian is a Gabor with omega == 0, so every kernel uses the same
        # signature. `kernel_full_range` integrates the analytic full support;
        # otherwise integrate only the ray-ellipsoid segment of this primitive.
        if self.kernel_full_range:
            density = self.kernel.density_integral(ray, ellipsoid, None, None, active)
        else:
            active_local, tmin, tmax = ray_ellipsoid_intersection(ray, ellipsoid, active)
            segment_size = dr.abs(tmax - tmin)
            ray_local = mi.Ray3f(ray)
            ray_local.o = ray.o + ray.d * tmin
            density = self.kernel.density_integral(ray_local, ellipsoid, mi.Float(0.0), segment_size, active_local)

        # Importance-sampling weight for the stochastic LOD criteria plus
        # (optionally) the continuous frequency-filter weight.
        if self.lod.main_needs_weights():
            density = density * self.lod.weight_for_shape(si.shape, ray.d, active)
        if self.enable_continuous_lod:
            density = density * continuous_lod_weight(
                ellipsoid, self.freq_threshold, self.freq_band_sigma, active)

        contribution = density * ellipsoid.opacity
        contribution = dr.select(dr.isfinite(contribution), contribution, 0.0)

        return self.activation_func(contribution)

    @dr.syntax
    def _sample_impl(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # This integrator only supports environment emitters
        emitter = scene.environment()

        # --------------------- Configure integrator state ---------------------

        primal = mode == dr.ADMode.Primal
        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        depth = mi.UInt32(0)

        if not primal:  # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)  # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0)  # Differential radiance
        β  = mi.Spectrum(0.0) if self.accumulate_beta else mi.Spectrum(1.0) # Path throughput weight

        # ----------------------------- Main loop ------------------------------

        stochastic_lod = self.lod.main_needs_weights()
        if stochastic_lod:
            ray.mask = self._ray_mask(sampler, ray, active, depth)

        while dr.hint(active, label=f"Primitive tracing ({mode.name})"):

            if not stochastic_lod:
                ray.mask = self._ray_mask(sampler, ray, active, depth)

            # --------------------- Find next intersection ---------------------
            si = scene.ray_intersect(
                ray,
                coherent=(depth == 0),
                ray_flags=mi.RayFlags.All,
                active=active,
            )

            # Check for valid intersection (ray hit a primitive)
            valid_hit = active & si.is_valid()

            # -------------- Interaction with environment emitter --------------
            # Handle rays that escaped (no intersection) BEFORE processing hits

            escaped_medium = active & ~si.is_valid()
            escaped_medium &= ~((depth == 0) & mi.Bool(self.hide_emitters))

            if self.output == 'optical_depth':
                # Training mode: the escape contribution is the accumulated
                # optical depth itself, no emitter involved.
                L_dir = dr.select(escaped_medium, β, 0.0)
            else:
                # For correct environment map evaluation, explicitly set wi
                # to -ray.d (ray_intersect does not guarantee wi for misses).
                si.wi = dr.select(escaped_medium, -ray.d, si.wi)
                L_dir = dr.select(escaped_medium, β * emitter.eval(si, escaped_medium), 0.0)

            # Update L for escaped rays
            L[active] = (L + L_dir) if primal else (L - L_dir)

            # ----------------- Primitive transmission evaluation --------------
            # Only evaluate transmission for valid hits to avoid garbage memory access

            with dr.resume_grad(when=not primal):
                transmission = self.eval_transmission(si, mi.Ray3f(ray), sampler, valid_hit)
                # Ensure transmission is safe for inactive/invalid lanes
                transmission = dr.select(valid_hit, transmission, 0.0 if self.accumulate_beta else 1.0)

            # Update β only for valid hits (masked update)
            if self.accumulate_beta:
                β[valid_hit] = β + transmission
            else:
                β[valid_hit] = β * transmission

            # -------------- Differential phase only (PRB logic) ---------------

            with dr.resume_grad(when=not primal):
                if not primal:
                    # Differentiable reflected indirect radiance for primitives
                    if self.render_marginalization and self.output == 'radiance':
                        Lr_ind = L + transmission - dr.detach(transmission)
                    else:
                        # Safe division: use 1.0 for invalid lanes to avoid div-by-zero
                        safe_transmission = dr.select(valid_hit & (dr.detach(transmission) != 0),
                                                       dr.detach(transmission), 1.0)
                        Lr_ind = L * transmission / safe_transmission

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = dr.select(valid_hit & dr.isfinite(Lr_ind), Lr_ind, 0.0)

                    if mode == dr.ADMode.Backward:
                        if self.render_marginalization and self.output == 'radiance':
                            dr.backward_from(δL + Lo)
                        else:
                            dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # ------- Update loop variables based on current interaction -------

            # Spawn new ray (don't use si.spawn_ray to avoid self intersections)
            ray.o[valid_hit] = si.p + ray.d * 5e-6

            # ----------------------- Stopping criterion -----------------------

            depth[valid_hit] += 1
            active &= valid_hit

            # Don't estimate next recursion if we exceeded number of bounces
            active &= depth < self.max_depth

        return L if primal else δL, True, [], L

    def _anyhit_supported(self, scene):
        return (hasattr(scene, 'ray_transmittance_ellipsoids')
                and self.kernel_full_range
                and not self.lod.main_needs_weights()
                and not self.enable_continuous_lod
                and not self.lod.is_gauss_ms
                and not self.hide_emitters)

    def _sample_anyhit(self, scene, sampler, ray, active):
        '''Primal evaluation via one any-hit traversal per ray: the shader
        accumulates the full-line signed optical depth of every intersected
        primitive (order-independent), replacing the sequential hit walk.'''
        emitter = scene.environment()
        ray = mi.Ray3f(dr.detach(ray))
        active = mi.Bool(active)
        ray.mask = self._ray_mask(sampler, ray, active)

        # The query returns the accumulated signed optical depth directly
        # (deterministic; +inf if a non-ellipsoid shape blocks the ray).
        tau = scene.ray_transmittance_ellipsoids(
            ray, mi.UInt32(0), int(mi.RayFlags.TransmittanceEllipsoids), active)
        tau = dr.select(dr.isfinite(tau), tau, 1e30)

        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wi = -ray.d

        if self.accumulate_beta:
            if self.output == 'optical_depth':
                L = mi.Spectrum(tau)
            else:  # render_marginalization radiance: tau times emitter
                L = mi.Spectrum(tau) * emitter.eval(si, active)
        else:
            L = mi.Spectrum(dr.exp(-tau)) * emitter.eval(si, active)
        L = dr.select(active, L, 0.0)
        return L, True, [], L

    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        if mode == dr.ADMode.Primal and self.use_anyhit:
            supported = self._anyhit_supported(scene)
            if not getattr(self, '_anyhit_reported', False):
                self._anyhit_reported = True
                mi.Log(mi.LogLevel.Info,
                       f'[gfields_tomography] use_anyhit requested: primal '
                       f'{"ENGAGED" if supported else "FALLBACK to hit walk (unsupported scene config)"}')
            if supported:
                return self._sample_anyhit(scene, sampler, ray, active)
        if self.use_kernel_freeze:
            return _frozen_tomography_sample(
                self, mode, scene, sampler, ray, δL, state_in, active, **kwargs)
        return self._sample_impl(
            mode, scene, sampler, ray, δL, state_in, active, **kwargs)

    def to_string(self):
        return "GfieldsTomographyIntegrator[]"


def _tomography_freeze_state(integrator, *args, **kwargs):
    return integrator


@dr.freeze(state_fn=_tomography_freeze_state)
def _frozen_tomography_sample(integrator, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
    return integrator._sample_impl(
        mode, scene, sampler, ray, δL, state_in, active, **kwargs)

mi.register_integrator("gfields_tomography", lambda props: GfieldsTomographyIntegrator(props))
