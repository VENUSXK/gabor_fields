# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator, mis_weight
from .common import *
from gfields.lod import LODSelector, continuous_lod_weight
from volprim.integrators.stack import alloc_stack


class GfieldsPRBIntegrator(RBIntegrator):
    '''
    Volumetric Primitives path tracer (path replay backpropagation).

    With default parameters this reproduces the baseline `volprim_prb`
    from Don't Splat Your Gaussians (Condor et al. 2025). The hierarchical
     / steerable / continuous-LOD modes are exposed as options:

    - stochastic level-of-detail rendering via per-ray visibility masks
      (`pyramid_level_count`, `stochastic_selection_criteria[_nee]`,
      `power_law_beta[_nee]`, `num_bounces_gaussian_ms`),
    - orientation-selective steerable assets (`asset_encoding='steerable'`,
      `num_orientations`, `orientation_selection[_nee]`,
      `orientation_threshold`, `orientation_decay_rate`, `bin_rotation`),
    - continuous frequency-domain LOD filtering (`freq_threshold`,
      `freq_band_sigma`); `freq_threshold` is exposed through `mi.traverse`
      so it can be driven interactively without scene rebuilds.

    Parameters:
        max_depth (int): Maximum path depth. A value of -1 indicates no limit.
        rr_depth (int): Minimum path depth before enabling the Russian roulette path termination.
        use_nee (bool): Use next-event estimation for indirect light sampling.
        use_indirect (bool): Use indirect light sampling.
        hide_emitters (bool): Hide emitters from indirect light sampling.
        phasefunction (dict): Phase function to use for volumetric interactions.
        kernel_type (str): Kernel for the volumetric primitives, one of
            ['gaussian', 'epanechnikov', 'gabor'].
        max_overlaps (int): Maximum number of overlapping primitives.
        max_depth_primitive (int): Maximum depth for primitive ray intersection.
        rr_depth_primitive (int): Minimum depth before enabling Russian roulette for primitive ray intersection.
        solver_max_iterations (int): Maximum number of iterations for the solver.
        solver_type (str): Solver for volumetric interactions. One of ['bisection', 'newton', 'disabled']
        num_bounces_gaussian_ms (int): Number of full-pyramid bounces before
            collapsing the **primary path** to the Gaussian (pyr0) layer
            when `stochastic_selection_criteria == 'gaussian_ms'`. NEE is
            unaffected and runs its own independent criterion. Default: 0.
        freq_threshold (float): Continuous-LOD frequency cutoff (world
            frequency). Gabor primitives whose spectral band lies above the
            cutoff fade out smoothly. Negative disables filtering (default).
        freq_band_sigma (float): Spectral band half-width multiplier for the
            continuous filter (default: 2.0).
    '''
    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        max_depth = int(props.get("max_depth", -1))
        if max_depth < 0 and max_depth != -1:
            raise Exception('"max_depth" must be set to -1 (infinite) or a value >= 0')
        self.max_depth = mi.UInt32(max_depth if max_depth != -1 else 0xFFFFFFFF) # Map -1 (infinity) to 2^32-1 bounces

        rr_depth       = int(props.get('rr_depth', -1))
        self.rr_depth  = mi.UInt32(rr_depth if rr_depth > 0 else 2**32-1)
        self.use_rr    = rr_depth < max_depth

        rr_nee_depth       = int(props.get('rr_nee_depth', -1))
        self.use_rr_nee    = rr_nee_depth > 0
        self.rr_nee_depth  = mi.UInt32(rr_nee_depth if rr_nee_depth > 0 else 2**32-1)
        
        self.use_nee       = bool(props.get("use_nee", True))     # Slow!
        self.use_indirect  = bool(props.get("use_indirect", True))

        # NEE origin-overlap integration convention (opt-in, like the primary
        # path's shell truncation). For primitives that already overlap the
        # shadow-ray origin:
        #   False (default): the fast UNBOUNDED forward integral
        #     density_integral(ray, 0, maxt) -- no ray-ellipsoid edge finding,
        #     the kernel's forward support up to the emitter. Substantially
        #     cheaper (one closed-form erf, no quadratic solve per primitive).
        #   True: the exact segment [origin, shell-exit] (capped at the emitter)
        #     via ray_ellipsoid_intersection -- the shell-TRUNCATED optical
        #     depth, matching the truncated distance sampler.
        # Down-ray primitives are always integrated over the full domain (the
        # any-hit query / full-line walk), independent of this flag. Exactness:
        # the two agree once extents are large (the clipped forward tail beyond
        # the shell exit vanishes); at small extent with high opacity they can
        # differ a few percent. Which is "more correct" is a convention: if the
        # primitives are taken as unbounded, the truncated form is the one that
        # closes the gap to it. See docs/prb_stochastic.md.
        self.nee_truncate_at_shell = bool(props.get("nee_truncate_at_shell", False))

        # Propagate the NEE shadow-ray transmittance gradient, the
        # self-/inter-shadowing term when the NEE transmittance is
        # evaluated detached. 
        # Default OFF because the self-shadowing case (a dense primitive
        # occluding its own scatter vertices' NEE) is still biased: the scatter
        # vertex position is itself opacity-dependent (free-flight sampling) and
        # the per-vertex adjoint here treats it as fixed, so the gradient can
        # overshoot / flip sign in dense single-primitive tests. For multi-
        # primitive gabor assets inter-shadowing dominates, so enabling it is a
        # reasonable opt-in. Use under your own risk. Jorge
        self.grad_nee_transmittance = bool(props.get("grad_nee_transmittance", False))

        self.phase = props.get('phasefunction')
        if self.phase is None:
            self.phase = mi.load_dict({'type': 'isotropic'})

        self.phase_is_microflake = bool(
            mi.has_flag(self.phase.flags(), mi.PhaseFunctionFlags.Microflake))

        props['kernel_full_range'] = props.get('kernel_full_range', False)
        props['kernel_normalized'] = props.get('kernel_normalized', True)
        self.kernel = Kernel.factory(props)

        self.max_overlaps        = int(props.get("max_overlaps", 32))
        self.max_depth_primitive = int(props.get("max_depth_primitive", 256))
        self.rr_depth_primitive  = int(props.get("rr_depth_primitive",  256))

        self.solver_max_iterations = int(props.get('solver_max_iterations', 4))
        self.solver_type = str(props.get('solver_type', 'bisection'))
        assert self.solver_type in ['bisection', 'newton', 'disabled']

        self.kernel_type_ = props.get('kernel_type', 'gaussian')

        # LOD selection (pyramid levels, stochastic criteria, steerable
        # orientations). 
        self.lod = LODSelector(props, allow_dynamic_orientation=True)

        # `gaussian_ms`: full-pyramid bounces before the primary path collapses
        # to the Gaussian (pyr0) layer
        self.num_bounces_gaussian_ms = int(props.get('num_bounces_gaussian_ms', 0))

        # Continuous frequency-domain LOD filter. Negative threshold disables it
        # (compiled out); non-negative enables it and exposes `freq_threshold`
        # via `traverse` for interactive control.
        freq_threshold = float(props.get('freq_threshold', -1.0))
        self.enable_continuous_lod = freq_threshold >= 0.0
        self.freq_threshold = mi.Float(freq_threshold)
        self.freq_band_sigma = float(props.get('freq_band_sigma', 2.0))

    def _shape_weight(self, shape, ellipsoid, ray_d, active, for_nee=False):
        """Combined per-primitive weight: stochastic-LOD importance weight
        times (optionally) the continuous frequency-filter weight."""
        w = self.lod.weight_for_shape(shape, ray_d, active, for_nee)
        if self.enable_continuous_lod:
            w = w * continuous_lod_weight(ellipsoid, self.freq_threshold,
                                          self.freq_band_sigma, active)
        return w

    def _projected_area(self, ray, active):
        '''Microflake extinction modulation `projected_area(wi) = sqrt(wi^T S wi)`
        for the ray direction (even in wi, so the sign is irrelevant). Exact for a
        spatially-constant SGGX `S`; for a spatially-varying `S` it evaluates at the
        ray origin (the analytic segment integral cannot carry a per-point `S`).'''
        mei = dr.zeros(mi.MediumInteraction3f)
        mei.wi = dr.normalize(ray.d)
        mei.p = ray.o
        mei.sh_frame = mi.Frame3f(mei.wi)
        return self.phase.projected_area(mei, active)

    def traverse(self, callback):
        if self.enable_continuous_lod:
            callback.put('freq_threshold', self.freq_threshold,
                         mi.ParamFlags.NonDifferentiable)

    def parameters_changed(self, keys):
        pass

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        # Check for required attributes on ellipsoids
        check_ellipsoids_attribute(scene, ['opacities', 'albedo'])

        # ------------------------------------------------------------------
        # Prepare integrator state
        # ------------------------------------------------------------------

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(dr.detach(ray))
        active    = mi.Bool(active)
        valid_ray = mi.Bool(False)
        depth     = mi.UInt32(0)

        prev_event = dr.zeros(mi.Interaction3f)
        prev_event_pdf = mi.Float(1.0)

        bsdf_ctx  = mi.BSDFContext()
        phase_ctx = mi.PhaseFunctionContext(sampler)

        if not primal: # If the gradient is zero, stop early
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β  = mi.Spectrum(1.0)

        # Primitives overlapping with the current path vertex
        primitives = alloc_stack(PrimitiveID, mi.UInt32, alloc_size=self.max_overlaps)

        # Reset the per-render gaussian_ms override state; it is updated
        # inside the loop before any helper that can apply the override.
        self.lod.set_gauss_only(mi.Bool(False))

        while active:

            # Update the Gaussian-only override for this iteration: once `depth`
            # (completed scatter events) reaches the
            # threshold, the primary path uses the Gaussian (pyr0) layer only.
            # NEE is unaffected. This is biased but can be much faster. Use with caution.
            if self.lod.is_gauss_ms:
                self.lod.set_gauss_only(active & (
                    depth >= mi.UInt32(self.num_bounces_gaussian_ms)))

            # ------------------------------------------------------------------
            # Sample volumetric interaction
            # ------------------------------------------------------------------

            primitives, si, weight, _, _, _, tr_β, sampled_t = primitive_tracing(
                scene, sampler, ray, primitives, depth,
                callback=self.sample_segment,
                payload=(self, scene, sampler.next_1d(), mi.Float(1.0), mi.Float(dr.inf)),
                active=active,
                max_depth_primitive=self.max_depth_primitive,
                rr_depth_primitive=self.rr_depth_primitive,
                stochastic_selection_criteria=self.lod.criteria,
                orientation_selection=self.lod.orientation_selection,
                select_mask=lambda pyrs, ori_s, ray_d, lanes, _criteria:
                    self.lod.select_mask(pyrs, ori_s, ray_d, lanes)
            )

            # Kill this path if it reached the maximum number of primitive
            # ray-intersection before sampling a medium interaction
            β *= weight
            active &= dr.any(β != 0.0)

            mei = dr.zeros(mi.MediumInteraction3f)
            # Normalize wi to prevent numerical drift causing HG phase function NaN
            # (cos_theta > 1 causes negative values in HG denominator)
            mei.wi = dr.normalize(-ray.d)
            mei.sh_frame = mi.Frame3f(mei.wi)
            mei.t = sampled_t
            mei.p = ray(sampled_t)

            active_surface = active & ~mei.is_valid() & si.is_valid()
            active_medium  = active &  mei.is_valid()
            escaped_medium = active & ~mei.is_valid() & ~si.is_valid()
            valid_ray |= active_medium | active_surface

            # Save depth BEFORE incrementing for use in emitter evaluation
            # (direct camera rays should not apply MIS weight)
            depth_before_scatter = mi.UInt32(depth)
            
            depth[active_surface | active_medium] += 1
            active_medium  &= depth < self.max_depth
            active_surface &= depth < self.max_depth

            # Propagate derivatives for the transmittance term (cancelled out by sampling PDF)
            if not primal:
                # Note: We are not handling visibility masking properly here and the gradient will likely be incorrect
                ray2 = dr.zeros(mi.Ray3f)
                ray2.o = mi.Point3f(ray.o)
                ray2.d = dr.select(escaped_medium, -mei.wi, dr.normalize(dr.select(active_surface, si.p, mei.p) - ray2.o))
                ray2.maxt = dr.select(escaped_medium, dr.inf, dr.norm(ray2.d))
                ray2.mask = mi.UInt32(255)
                active2 = active_medium | active_surface | (escaped_medium & mi.Bool(self.use_indirect))
                self.eval_transmittance(scene, sampler, ray2, primitives, L, δL, mode, active2)

            # ------------------------------------------------------------------
            # Update path throughput based on volumetric interaction albedo
            # ------------------------------------------------------------------

            albedo = self.eval_albedo(scene, ray, mei.p, primitives, L, δL, mode, active_medium)
            β[active_medium] *= albedo

            # ------------------------------------------------------------------
            # Intersection with emitters (both area emitters and environment)
            # ------------------------------------------------------------------

            if self.use_indirect:
                # Check for emitter at surface interaction (area lights)
                emitter = si.emitter(scene)
                active_e_surface = active_surface & (emitter != None)
                
                # Check for environment emitter (escaped rays)
                env_emitter = scene.environment()
                active_e_env = escaped_medium & (env_emitter != None)
                
                # Combined active emitter mask (before hide_emitters check)
                active_e = active_e_surface | active_e_env
                
                # Hide emitters if requested (for camera rays)
                # Use depth_before_scatter to check if this is a direct camera ray
                active_e &= ~((depth_before_scatter == 0) & mi.Bool(self.hide_emitters))
                
                # For direct camera rays, don't apply MIS weight (following prbvolpath)
                # Use depth_before_scatter since depth has already been incremented
                count_direct = (depth_before_scatter == 0)
                
                # Get the PDF of sampling this emitter using next event estimation
                if self.use_nee:
                    ds = mi.DirectionSample3f(scene, si, prev_event)
                    emitter_pdf = scene.pdf_emitter_direction(prev_event, ds, active_e)
                else:
                    emitter_pdf = mi.Float(0.0)

                with dr.resume_grad(when=not primal):
                    # Evaluate emitter contribution
                    # Area emitter contribution
                    emitter_val_surface = emitter.eval(si, active_e_surface)
                    
                    # Environment emitter contribution
                    emitter_val_env = mi.Spectrum(0.0)
                    if env_emitter is not None:
                        emitter_val_env = env_emitter.eval(si, active_e_env)
                    
                    # Combine emitter values
                    emitter_val = dr.select(active_e_surface, emitter_val_surface, emitter_val_env)
                    
                    # Compute contribution with proper MIS weight handling
                    # For direct rays (count_direct), don't apply MIS weight
                    contrib_direct = β * emitter_val
                    contrib_indirect = β * mis_weight(prev_event_pdf, emitter_pdf) * emitter_val
                    Lr_dir = dr.select(count_direct, contrib_direct, contrib_indirect)
                    Lr_dir = dr.select(active_e, Lr_dir, 0.0)

                    # Propagate derivatives to/from `emitter_val`
                    if not primal:
                        if dr.grad_enabled(Lr_dir):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_dir)
                            else:
                                δL += dr.forward_to(Lr_dir)

                L = (L + Lr_dir) if primal else (L - Lr_dir)

            # ------------------------------------------------------------------
            # Emitter sampling (Next Event Estimation)
            # ------------------------------------------------------------------

            bsdf = si.bsdf(ray)

            if dr.hint(self.use_nee, mode='scalar'):
                ref = dr.zeros(mi.Interaction3f)
                ref[active_medium]  = mei
                ref[active_surface] = si

                active_surface_nee = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                nee_active = active_surface_nee | active_medium

                with dr.resume_grad(when=not primal):
                    # Sample emitter direction using scene's emitter sampling (supports all emitter types)
                    # Disable built-in visibility test (test_visibility=False) because it treats
                    # ellipsoid volumes as hard occluders. We handle visibility ourselves via
                    # eval_transmittance_nee which computes proper volumetric transmittance.
                    ds, emitter_val = scene.sample_emitter_direction(ref, sampler.next_2d(), False, nee_active)
                    
                    # Check if the sampled direction is valid
                    active_nee = nee_active & (ds.pdf > 0.0)
                    
                    ray2 = ref.spawn_ray_to(ds.p)
                    
                    # Ensure ray2 is valid (finite direction and positive maxt)
                    ray2_valid = dr.all(dr.isfinite(ray2.d)) & (ray2.maxt > 1e-6)
                    active_nee = active_nee & ray2_valid

                    # Snapshot the sampler *before* the primal value walk so the
                    # adjoint transmittance replay below re-draws the identical
                    # stochastic-LOD masks / RR randoms and thus visits the same
                    # shadow segments (PRB path replay; cf. the `nee_sampler`
                    # clone in prbvolpath.sample_emitter). Only needed when the
                    # (experimental) transmittance-gradient replay runs.
                    nee_sampler = (sampler.clone()
                                   if (not primal and self.grad_nee_transmittance)
                                   else sampler)

                    transmittance = self.eval_transmittance_nee(scene, sampler, ray2, primitives, L, δL, dr.ADMode.Primal, active_nee)

                    # Query the BSDF for that emitter-sampled direction
                    bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, si.to_local(ds.d), active_surface_nee)
                    phase_val, phase_pdf = self.phase.eval_pdf(phase_ctx, mei, ds.d, active_medium)
                    phase_pdf = dr.detach(dr.select(ds.delta, 0.0, phase_pdf))
                    if not self.use_indirect:
                        phase_pdf = mi.Float(0.0)

                    nee_val = dr.select(active_surface_nee, bsdf_val, phase_val)
                    nee_pdf = dr.select(ds.delta, 0.0, dr.select(active_surface_nee, bsdf_pdf, phase_pdf))

                    if not self.use_indirect:
                        nee_pdf = mi.Float(0.0)

                    Lr_nee = β * nee_val * mis_weight(ds.pdf, nee_pdf) * transmittance * emitter_val
                    Lr_nee = dr.select(active_nee, Lr_nee, 0.0)

                    # Propagate derivatives to/from `emitter_val` and `phase_val`
                    if not primal:
                        if dr.grad_enabled(Lr_nee):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_nee)
                            else:
                                δL += dr.forward_to(Lr_nee)
                

                # Propagate the NEE shadow-ray transmittance gradient
                if not primal and self.grad_nee_transmittance:
                    self.eval_transmittance_nee(scene, nee_sampler, ray2, primitives,
                                                dr.detach(Lr_nee), δL, mode, active_nee)

                L = (L + Lr_nee) if primal else (L - Lr_nee)

            # ------------------------------------------------------------------
            # Phase function sampling
            # ------------------------------------------------------------------

            # TODO support spatially varying phase function

            # Sample with gradients suspended (mirrors `bsdf.sample` and upstream
            # `prbvolpath`): the sampled direction/weight must not carry the phase
            # function's parameter gradient. That goes through the detach-then-
            # reattach block below.
            with dr.suspend_grad():
                wo, phase_weight, phase_pdf_sampled = self.phase.sample(phase_ctx, mei,
                                                                        sampler.next_1d(),
                                                                        sampler.next_2d(),
                                                                        active_medium)
                # Normalize to prevent numerical drift accumulating over bounces
                wo = dr.normalize(wo)
                phase_val_unused, phase_pdf = self.phase.eval_pdf(phase_ctx, mei, wo, active_medium)

            active_medium &= (phase_pdf > 0.0) & dr.all(dr.isfinite(phase_weight))

            # Re-evaluate the phase function at the detached direction in an
            # attached manner and inject its gradient (standard PRB mechanism for
            # differentiable scattering params, e.g. HG's `g`; mirrors the BSDF
            # `Lo` block below).
            if not primal:
                with dr.resume_grad():
                    phase_eval, _ = self.phase.eval_pdf(phase_ctx, mei, wo, active_medium)
                    if dr.grad_enabled(phase_eval):
                        Lo = phase_eval * dr.detach(dr.select(active_medium, L / dr.maximum(1e-8, phase_eval), 0.0))
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * Lo)
                        else:
                            δL += dr.forward_to(Lo)

            β[active_medium] *= phase_weight

            ray.o[active_medium]    = mei.p
            ray.d[active_medium]    = wo
            ray.maxt[active_medium] = dr.inf

            prev_event[active_medium] = dr.detach(mei)
            prev_event_pdf[active_medium] = phase_pdf

            # ------------------------------------------------------------------
            # BSDF sampling
            # ------------------------------------------------------------------

            bs, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                          sampler.next_1d(active_surface),
                                          sampler.next_2d(active_surface),
                                          active_surface)
            active_surface &= bs.pdf > 0

            if not primal:
                with dr.resume_grad():
                    bsdf_eval = bsdf.eval(bsdf_ctx, si, bs.wo, active_surface)
                    if dr.grad_enabled(bsdf_eval):
                        Lo = bsdf_eval * dr.detach(dr.select(active_surface, L / dr.maximum(1e-8, bsdf_eval), 0.0))
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * Lo)
                        else:
                            δL += dr.forward_to(Lo)

            β[active_surface] *= bsdf_weight
            ray[active_surface] = si.spawn_ray(si.to_world(bs.wo))

            prev_event[active_surface] = dr.detach(si)
            prev_event_pdf[active_surface] = bs.pdf

            # ------------------------------------------------------------------
            # Termination criterion (russian roulette)
            # ------------------------------------------------------------------

            sample_rr = sampler.next_1d() # Ensures the same sequence of random number is drawn for the primal and adjoint passes.
            if primal and self.use_rr:
                q = dr.minimum(dr.max(β), 0.99)
                perform_rr = (depth > self.rr_depth)
                active &= (sample_rr < q) | ~perform_rr
                β[perform_rr] = β * dr.rcp(q)

            # Kill path if has insignificant contribution
            active &= dr.any(β > 0.005)
            active &= dr.any((β != 0.0))
            active &= active_surface | active_medium

        return L if primal else δL, valid_ray, [], L
    
    @dr.syntax
    def eval_transmittance_nee(self, scene, sampler, ray, primitives, L, δL, mode, active):
        '''
        Evaluate the transmittance along a ray segment for NEE.
        Uses precomputed level weights looked up by ray.mask.
        '''
        # Early exit if the scene doesn't contains volumetric primitives
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return mi.Float(1.0)

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(ray)
        L = dr.detach(L)

        # Microflake (SGGX) extinction scaling for this shadow ray's direction
        # (constant along the ray for a spatially-constant S).
        pa = self._projected_area(ray, active) if self.phase_is_microflake else None

        # ----------------------------------------------------------------------
        # Process overlapping primitives (apply NEE weights via lookup)
        # ----------------------------------------------------------------------
        total_tr = mi.Float(1.0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        it = mi.UInt32(0)
        while active1:
            prim_id = primitives.value(it, active1)
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
                sigma_t = ellipsoid.opacity
                if self.phase_is_microflake:
                    sigma_t = sigma_t * pa
                # Overlapping primitive. Truncated (nee_truncate_at_shell):
                # integrate only the segment from the shadow-ray origin (inside
                # the shell) to the shell EXIT, capped at the emitter. Unbounded
                # (default, faster): the forward integral over [0, maxt] with no
                # edge finding. Down-ray primitives below always use the full
                # domain (tmin=tmax=None).
                if dr.hint(self.nee_truncate_at_shell, mode='scalar'):
                    _valid, _t0, t1 = ray_ellipsoid_intersection(ray, ellipsoid, active1)
                    # Detach the shell-exit bound: differentiate the integrand,
                    # not the moving boundary.
                    t1 = dr.detach(dr.minimum(t1, ray.maxt))
                else:
                    t1 = ray.maxt
                density = self.kernel.density_integral(ray, ellipsoid, tmin=mi.Float(0.0), tmax=t1, active=active1)
                
                weight = mi.Float(1.0)
                if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                    pass
                else:
                    weight = self._shape_weight(prim_id.shape, ellipsoid, ray.d, active1)
                density *= weight
                tr = dr.exp(-density * sigma_t)

                valid = active1
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)

                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active1 & ((1.0 - tr) > 1e-2)
                    contrib[~valid] = 0.0

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            # Accumulate the detached transmittance
            total_tr[valid] *= dr.detach(tr)
            it += 1
            active1 &= (it < primitives.size())

        # ----------------------------------------------------------------------
        # Trace ray: handle both hard surfaces and ellipsoid volumes
        # ----------------------------------------------------------------------

        ray.maxt = ray.maxt * (1.0 - 1e-4)
        it = mi.UInt32(0)
        active2 = mi.Bool(active)
        # Sample NEE pyramid level and orientation for ellipsoid visibility
        if self.lod.nee_is_deterministic():
            ray.mask = mi.UInt32(255)
        else:
            pyrs2 = sampler.next_1d()
            needs_ori_nee = self.lod.orientation_selection_nee not in ['deterministic', 'threshold']
            ori_s2 = sampler.next_1d() if needs_ori_nee else mi.Float(0)
            ray.mask = self.lod.select_mask(pyrs2, ori_s2, ray.d, active2, for_nee=True)
            
        while active2:
            si = scene.ray_intersect(ray, coherent=False, ray_flags=mi.RayFlags.All, active=active2)
            hit_valid = si.is_valid()
            active2 &= hit_valid

            # Check if we hit a hard surface (non-ellipsoid) → full occlusion
            is_ellipsoids = active2 & si.shape.is_ellipsoids()
            hit_hard_surface = active2 & ~is_ellipsoids
            total_tr[hit_hard_surface] = 0.0
            active2 &= ~hit_hard_surface  # Stop tracing rays that hit hard surfaces

            # For ellipsoid hits, compute transmittance and continue
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active2)
                sigma_t = ellipsoid.opacity
                if self.phase_is_microflake:
                    sigma_t = sigma_t * pa
                density = mi.Float(0.0)
                density += self.kernel.density_integral(ray, ellipsoid, tmin=None, tmax=None, active=active2)
                
                weight = mi.Float(1.0)
                if self.lod.nee_is_deterministic() and not self.enable_continuous_lod:
                    pass
                else:
                    weight = self._shape_weight(si.shape, ellipsoid, ray.d, active2, for_nee=True)
                density *= weight

                tr = dr.exp(-density * sigma_t)

                # Guard against NaN: only use valid, finite transmittance values
                valid = active2 & dr.isfinite(tr)
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)
                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active2 & ((1.0 - tr) > 1e-2)
                    contrib = dr.select(valid, contrib, 0.0)

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            total_tr[valid] *= dr.detach(tr)
            if self.use_rr_nee:
                q = dr.minimum(total_tr, 0.99)
                perform_rr = (it > self.rr_nee_depth) & (q > 0.0)
                valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
                total_tr[active2 & perform_rr] *= dr.rcp(q)
                total_tr[active2 & ~valid_rr] = 0.0
                active2 &= valid_rr
            # Kill paths that are too long
            if self.max_depth_primitive > 0:
                total_tr[active2 & (it >= self.max_depth_primitive)] = 0.0
                active2 &= (it < self.max_depth_primitive)
            # Update rays
            ray.o = spawn_ray_origin(si, ray)
            ray.maxt -= si.t

            it += 1

        # Clamp final transmittance to [0, 1]. We allow Gabor negative/positive densities
        # to cancel out during the loop, but prevent energy amplification at the end. This
        # can create a bit of bias if you are very aggressive with the NEE parameters, but
        # I think it is better than allowing weird energy amplification artifacts - Jorge.
        total_tr = dr.clamp(total_tr, 0.0, 1.0)
        return total_tr
    
    @dr.syntax
    def eval_transmittance(self, scene, sampler, ray, primitives, L, δL, mode, active):
        '''
        Evaluate the transmittance along a ray segment.
        Uses precomputed level weights looked up by ray.mask.
        '''
        # Early exit if the scene doesn't contains volumetric primitives
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return mi.Float(1.0)

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(ray)
        L = dr.detach(L)
        
        # Microflake (SGGX) extinction scaling for this ray's direction
        # (constant along the ray for a spatially-constant S).
        pa = self._projected_area(ray, active) if self.phase_is_microflake else None

        total_tr = mi.Float(1.0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        it = mi.UInt32(0)
        while active1:
            prim_id = primitives.value(it, active1)
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
                sigma_t = ellipsoid.opacity
                if self.phase_is_microflake:
                    sigma_t = sigma_t * pa
                density = mi.Float(0.0)
                density = self.kernel.density_integral(ray, ellipsoid, tmin=mi.Float(0.0), tmax=ray.maxt, active=active1)
                
                weight = mi.Float(1.0)
                if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                    pass
                else:
                    weight = self._shape_weight(prim_id.shape, ellipsoid, ray.d, active1)
                density *= weight

                tr = dr.exp(-density * sigma_t)

                valid = active1
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)

                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active1 & ((1.0 - tr) > 1e-2)
                    contrib[~valid] = 0.0

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            total_tr[valid] *= tr
            it += 1
            active1 &= (it < primitives.size())

        # ----------------------------------------------------------------------
        # Trace ray: handle both hard surfaces and ellipsoid volumes in one loop
        # ----------------------------------------------------------------------

        it = mi.UInt32(0)
        active2 = mi.Bool(active)
        
        # Find the next intersection along the ray
        if self.lod.main_is_deterministic():
            ray.mask = mi.UInt32(255)
        else:
            pyrs = sampler.next_1d()
            needs_ori = self.lod.orientation_selection not in ['deterministic', 'threshold']
            ori_s = sampler.next_1d() if needs_ori else mi.Float(0)
            ray.mask = self.lod.select_mask(pyrs, ori_s, ray.d, active2)
        # Force ray.mask to the Gaussian bit past the gauss-only depth threshold
        ray.mask = self.lod.apply_gauss_only(ray.mask)
        while active2:
            si = scene.ray_intersect(ray, coherent=False, ray_flags=mi.RayFlags.All, active=active2)
            hit_valid = si.is_valid()
            active2 &= hit_valid

            # Check if we hit a hard surface (non-ellipsoid) 
            is_ellipsoids = active2 & si.shape.is_ellipsoids()
            hit_hard_surface = active2 & ~is_ellipsoids
            total_tr[hit_hard_surface] = 0.0
            active2 &= ~hit_hard_surface  # Stop tracing rays that hit hard surfaces
            
            # For ellipsoid hits, compute transmittance and continue
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active2)
                sigma_t = ellipsoid.opacity
                if self.phase_is_microflake:
                    sigma_t = sigma_t * pa
                density = self.kernel.density_integral(ray, ellipsoid, tmin=None, tmax=None, active=active2)
                
                weight = mi.Float(1.0)
                if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                    pass
                else:
                    weight = self._shape_weight(si.shape, ellipsoid, ray.d, active2)
                density *= weight
                tr = dr.exp(-density * sigma_t)

                valid = active2
                if not primal:
                    d_tr = dr.replace_grad(1.0, tr)
                    contrib = d_tr * dr.detach(safe_rcp(tr) * L)

                    # Avoid NaNs
                    valid = active2 & ((1.0 - tr) > 1e-2)
                    contrib = dr.select(valid, contrib, 0.0)

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            fwd_grad = dr.forward_to(contrib)
                            δL += fwd_grad

            total_tr[valid] *= tr

            # Update rays
            ray.o = spawn_ray_origin(si, ray)
            ray.maxt -= si.t

            it += 1

            if primal:
                # Perform Russian Roulette
                if self.rr_depth_primitive > 0:
                    q = dr.minimum(total_tr, 0.99)
                    perform_rr = (it > self.rr_depth_primitive) & (q > 0.0)
                    valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
                    total_tr[active2 & perform_rr] *= dr.rcp(q)
                    total_tr[active2 & ~valid_rr] = 0.0
                    active2 &= valid_rr

                # Kill paths that are too long
                if self.max_depth_primitive > 0:
                    total_tr[active2 & (it >= self.max_depth_primitive)] = 0.0
                    active2 &= (it < self.max_depth_primitive)

        total_tr = dr.clamp(total_tr, 0.0, 1.0)
        return total_tr
    


    @dr.syntax
    def eval_albedo(self, scene, ray, p, primitives, L, δL, mode, active):
        '''
        Returns the albedo value of the volumetric primitives involved in the
        interaction. Uses precomputed level weights.
        '''
        # Early exit if the scene doesn't contains volumetric primitives
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return dr.zeros(mi.Spectrum)

        accum_albedo = dr.zeros(mi.Spectrum)
        accum_pdf    = dr.zeros(mi.Float)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        it1 = mi.UInt32(0)
        while active1:
            prim_id = primitives.value(it1, active1)
            si1 = dr.zeros(mi.SurfaceInteraction3f)
            si1.prim_index = prim_id.index
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            sigma_t   = ellipsoid.opacity
            albedo    = dr.detach(prim_id.shape.eval_attribute_3('albedo', si1, active1))
            pdf = mi.Float(0.0)
            pdf = self.kernel.pdf(p, ellipsoid, active1)
            weight = mi.Float(1.0)
            if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                pass
            else:
                weight = self._shape_weight(prim_id.shape, ellipsoid, ray.d, active1)
            pdf *= weight * sigma_t
            accum_pdf += pdf
            accum_albedo += pdf * albedo
            it1 += 1
            active1 &= (it1 < primitives.size())

        # ----------------------------------------------------------------------
        # Adjoint albedo propagation
        # ----------------------------------------------------------------------
        # TODO fix albedo derivatives when stochastically sampling frequency levels
        
        if mode != dr.ADMode.Primal:
            L_albedo = dr.detach(L) * safe_rcp(accum_albedo * safe_rcp(accum_pdf))

            # Propagate derivatives for albedo terms
            active2 = mi.Bool(active) & ~primitives.is_empty()
            it2 = mi.UInt32(0)
            while active2:
                prim_id = primitives.value(it2, active2)

                with dr.resume_grad():
                    si1 = dr.zeros(mi.SurfaceInteraction3f)
                    si1.prim_index = prim_id.index
                    ellipsoid = Ellipsoid.gather(prim_id.shape, si1.prim_index, active2)
                    sigma_t   = ellipsoid.opacity
                    albedo    = prim_id.shape.eval_attribute_3('albedo', si1, active2)
                    pdf = self.kernel.pdf(p, ellipsoid, active2)
                    weight = mi.Float(1.0)
                    if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                        pass
                    else:
                        weight = self._shape_weight(prim_id.shape, ellipsoid, ray.d, active2)
                    pdf *= weight * sigma_t

                    contrib = mi.Spectrum(0.0)

                    # 1. Derivative of the `accum_albedo` term in `accum_albedo / accum_pdf`
                    d_albedo_pdf = dr.replace_grad(1.0, albedo * pdf)
                    contrib += d_albedo_pdf * dr.detach(safe_rcp(accum_pdf) * L_albedo)

                    # 2. Derivative of the `accum_pdf` term in `accum_albedo / accum_pdf`
                    d_pdf = dr.replace_grad(1.0, pdf)
                    contrib += -d_pdf * dr.detach(accum_albedo * safe_rcp(accum_pdf * accum_pdf) * L_albedo)

                    # 3. Derivative of the pdf term that cancels out in the weight (as it is also contained in the PDF)
                    contrib += d_pdf * dr.detach(L * safe_rcp(accum_pdf))

                    contrib = dr.select(active2, contrib, 0.0)

                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            δL += dr.forward_to(contrib)

                it2 += 1
                active2 &= (it2 < primitives.size())

        albedo = accum_albedo * safe_rcp(accum_pdf)
        return albedo

    @dr.syntax
    def sample_segment(self, payload, primitives, ray, accum_t, seg_t0, seg_t1, active):
        # Fetch integrator state from payload
        self, scene, sample, β, sampled_t = payload
        
        seg_tau = mi.Float(0.0)
        # Microflake (SGGX) extinction scaling for this segment's ray direction.
        pa = self._projected_area(ray, active) if self.phase_is_microflake else None
        it  = mi.UInt32(0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        while active1:
            density = mi.Float(0.0)
            prim_id = primitives.value(it, active1)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            density += self.kernel.density_integral(ray, ellipsoid, seg_t0, seg_t1, active1)
            weight = mi.Float(1.0)
            if self.lod.main_is_deterministic() and not self.enable_continuous_lod:
                pass
            else:
                weight = self._shape_weight(prim_id.shape, ellipsoid, ray.d, active1)
            density *= weight
            sigma_t = ellipsoid.opacity
            if self.phase_is_microflake:
                sigma_t = sigma_t * pa
            seg_tau[active1] += density * sigma_t
            it += 1
            active1 &= (it < primitives.size())
        
        seg_tr = dr.exp(-seg_tau)
        
        # ----------------------------------------------------------------------
        # Test segment sampling condition
        # ----------------------------------------------------------------------

        chi_i = β * seg_tr
        success = active & (chi_i < sample) & ~primitives.is_empty()

        if self.solver_type == 'disabled':
            # Sample uniformly within the segment (slightly biased but much faster)
            t_s = seg_t0 + sample * (seg_t1 - seg_t0)
        else:
            large_enough_segment = success & ((seg_t1 - seg_t0) > 8e-2)
            t_s = dr.select(
                large_enough_segment,
                self.primitives_sample_interaction_segment(scene, ray, primitives, seg_t0, seg_t1, sample, β, large_enough_segment),
                seg_t0 + sample * (seg_t1 - seg_t0)
            )

        sampled_t[success] = t_s + accum_t

        β[active] *= seg_tr
        β[success] = mi.Float(sample)

        return ~success, (self, scene, sample, β, sampled_t)

    @dr.syntax
    def primitives_sample_interaction_segment(self, scene, ray, primitives, seg_t0, seg_t1, sample, β, active):
        '''
        Sample a scattering distance for a segment defined by [seg_t0, seg_t1].
        Uses precomputed level weights.
        '''
        active = mi.Bool(active) & ~primitives.is_empty()

        # Initial guess for the Newton solver
        t_s = (seg_t0 + seg_t1) / 2.0
        chi = -dr.log(sample / β)
        # Microflake (SGGX) extinction scaling for this segment's ray direction
        pa = self._projected_area(ray, active) if self.phase_is_microflake else None
        if self.kernel_type_ == "gaussian":
            # Handle the case where we have a single primitive
            single_case = active & (primitives.size() == 1)
            prim_id = primitives.value(mi.UInt32(0), single_case)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, single_case)
            sigma_t = ellipsoid.opacity
            if self.phase_is_microflake:
                sigma_t = sigma_t * pa
            t_s = self.kernel.inv_cdf(ray, ellipsoid, sigma_t, chi, single_case)
            success = single_case
            active &= ~success

        # ----------------------------------------------------------------------
        # Fetching the primitive attributes (including visibility_mask for weight lookup)
        # ----------------------------------------------------------------------

        active1 = mi.Bool(active) & ~primitives.is_empty()
        it1 = mi.UInt32(0)
        ellipsoids = alloc_stack(Ellipsoid, mi.UInt32, alloc_size=primitives.alloc_size())
        sigmats    = alloc_stack(mi.Float,  mi.UInt32, alloc_size=primitives.alloc_size())
        # `vis_masks` only needed for criteria carrying per-shape IS weights;
        # `deterministic`/`gaussian_ms` use weight 1.0 unconditionally.
        needs_iw = self.lod.main_needs_weights()
        vis_masks = None
        if needs_iw:
            vis_masks  = alloc_stack(mi.UInt32, mi.UInt32, alloc_size=primitives.alloc_size())

        while active1:
            prim_id = primitives.value(it1, active1)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active1)
            sigma_t = ellipsoid.opacity
            if self.phase_is_microflake:
                sigma_t = sigma_t * pa

            ellipsoids.push(ellipsoid, active1)
            sigmats.push(sigma_t, active1)
            if needs_iw:
                vis_masks.push(prim_id.shape.visibility_mask(), active1)

            it1 += 1
            active1 &= (it1 < primitives.size())

        # ----------------------------------------------------------------------
        # Newton / bisection solver (uses precomputed weights via visibility_mask)
        # ----------------------------------------------------------------------
        
        it = mi.UInt32(0)
        while active:
            cdf = mi.Float(0.0)
            pdf = mi.Float(0.0)

            it2  = mi.UInt32(0)
            active2 = mi.Bool(active)
            while active2:
                ellipsoid = ellipsoids.value(it2, active2)
                sigmat    = sigmats.value(it2, active2)
                vis_mask = None
                if needs_iw:
                    vis_mask  = vis_masks.value(it2, active2)

                # Compute integral value with precomputed weight lookup by visibility_mask
                density = self.kernel.density_integral(ray, ellipsoid, seg_t0, t_s, active2)
                
                # Weight lookup by visibility_mask (skipped for weight-free
                # criteria: `deterministic`/`gaussian_ms` use 1.0).
                weight = mi.Float(1.0)
                if needs_iw:
                    for mask_val, w in self.lod.level_weight_map.items():
                        weight = dr.select(active2 & (vis_mask == mi.UInt32(mask_val)), mi.Float(w), weight)
                if self.enable_continuous_lod:
                    weight = weight * continuous_lod_weight(
                        ellipsoid, self.freq_threshold, self.freq_band_sigma, active2)
                density *= weight
                cdf += density * sigmat 
                
                density_pdf = mi.Float(0.0)
                if self.solver_type == 'newton':
                    density_pdf = self.kernel.pdf(ray(t_s), ellipsoid, active2)
                    density_pdf *= weight
                    pdf += density_pdf * sigmat

                it2 += 1
                active2 &= (it2 < ellipsoids.size())

            step = mi.Float(0.0)
            if self.solver_type == 'newton':
                # Newton algorithm
                step = (dr.exp(-chi) - dr.exp(-cdf)) / pdf
                t_s[active] = t_s - step
            else:
                # Bisection algorithm
                step = (seg_t1 - seg_t0) / dr.power(2.0, it + 2)
                t_s[active] = dr.select(cdf > chi, t_s - step, t_s + step)

            # Avoid NaNs and accelerate solver by clipping steps outside of segment
            diverge = active & ((t_s < seg_t0) | (t_s > seg_t1))
            t_s[diverge] = dr.clip(t_s, seg_t0, seg_t1)

            # If diverged at the last step, return t_s as the center of the segment
            t_s[diverge & ((it + 1 == self.solver_max_iterations))] = 0.5 * (seg_t0 + seg_t1)

            it += 1
            active &= (it < self.solver_max_iterations)
            active &= (dr.abs(step) > 5e-6) # TODO select this value according to something (i.e minimum intersection displacement? can possibly be bigger, this is very safe)

        return t_s

    def to_string(self):
        return 'GfieldsPRBIntegrator[]'

mi.register_integrator("gfields_prb", lambda props: GfieldsPRBIntegrator(props))

