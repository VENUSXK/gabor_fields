# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import mis_weight
from .common import *
from volprim.integrators.stack import alloc_stack

from .gfields_prb import GfieldsPRBIntegrator


class GfieldsPRBStochasticIntegrator(GfieldsPRBIntegrator):
    '''
    Stochastic **Gaussian-only** volumetric-primitive path tracer: min-of-free-
    flights distance sampling. Same estimand as `gfields_prb`, different
    distance-sampling algorithm: instead of collecting the overlapping primitive
    set and integrating the mixture optical depth per segment (with an iterative
    solver), it draws one closed-form inverse-CDF free flight per Gaussian
    primitive and the shortest sampled distance wins -- the minimum of
    independent per-primitive flights follows exactly the mixture free-flight
    distribution. No acceptance test, no overlap bookkeeping, no Newton/bisection
    solver. By default kernels are treated as unbounded (the extent shells serve
    only as BVH proxies that discover which primitives a ray touches), matching
    full-range tomography and the full-line NEE integrals. This diverges from
    gfields_prb, which truncates transport at the extent shells: the clipped-tail
    mass is a few percent at extent 3 with high opacity (NOT always negligible),
    vanishing by extent >= 4. Set `truncate_at_shell=True` to draw each free
    flight from the shell-restricted density instead -- the exact estimand
    gfields_prb integrates, agreeing with it at every extent (validated against
    an independent brute-force reference; ~2% runtime cost). A small stack of the
    primitives covering the current vertex provides the albedo and the NEE rays'
    origin-interior handling.

    ONLY `kernel_type='gaussian'` is supported: Gaussian density is non-negative,
    so its exact inverse CDF exists in closed form and every sampled collision is
    real. A Gabor mixture's SIGNED density (envelope * cos) breaks this (a
    non-monotone CDF has no direct inverse, and delta-tracking's positive-
    majorant assumption fails), so the integrator raises for non-Gaussian kernels
    -- use `gfields_prb` for Gabor assets. See docs/prb_stochastic.md.

    Additional properties (all `gfields_prb` properties are honored;
    `solver_type` / `solver_max_iterations` are unused and `max_overlaps`
    only sizes the coverage stack):

        nee_transmittance (str): 'analytic' (default) multiplies the
            per-primitive analytic transmittances like `gfields_prb` (zero
            added variance; with `use_anyhit` it is one deterministic
            optical-depth traversal per shadow ray, otherwise it walks
            every primitive to the emitter with `rr_nee_depth` Russian
            roulette). 'stochastic' terminates shadow rays with one
            Bernoulli test per intersected primitive instead (unbiased,
            early-exits at the first absorption) -- measured NOT to pay off
            at equal variance, kept for comparison and the NEE-gradient replay.
    '''

    def __init__(self, props=mi.Properties()):
        super().__init__(props)

        self.nee_transmittance = str(props.get('nee_transmittance', 'analytic'))
        assert self.nee_transmittance in ('stochastic', 'analytic')

        self.use_anyhit = bool(props.get('use_anyhit', True))
        self.use_anyhit_sampling = bool(props.get('use_anyhit_sampling', False))
        self.grad_nee_transmittance = bool(props.get('grad_nee_transmittance', False))
        self.residual_weight_clamp = float(props.get('residual_weight_clamp', 1.0))
        self.truncate_at_shell = bool(props.get('truncate_at_shell', False))
        self.nee_truncate_at_shell = bool(
            props.get('nee_truncate_at_shell', self.truncate_at_shell))

        if self.kernel_type_ != 'gaussian':
            raise Exception(
                f"gfields_prb_stochastic supports only kernel_type='gaussian' "
                f"(got '{self.kernel_type_}'): its distance sampler is biased for "
                f"signed Gabor mixtures. Use gfields_prb for Gabor assets.")

    # ------------------------------------------------------------------
    # Analytic per-primitive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inv_cdf(ray, ellipsoid, sigma, chi, active, truncate=False):
        '''Closed-form free-flight inverse CDF of one primitive: the first
        distance t (world units, from `ray.o`, forward only) at which the
        envelope optical depth sigma * int k reaches `chi`, in
        density_integral's exact normalization; dr.inf when the remaining
        mass is smaller. For a Gaussian primitive the envelope is the kernel
        itself, so this is its exact free flight.

        `truncate=False` (default): integrate over the kernel's full unbounded
        support (extent shell is a pure discovery proxy). `truncate=True`:
        restrict the density to the ray's segment inside the extent shell
        [t_in, t_out] (from `ray_ellipsoid_intersection`) -- the near tail
        before shell entry is dropped and no collision is reported past the
        shell exit, matching gfields_prb's per-segment truncated transport.'''
        s = ellipsoid.scale
        w = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        J = dr.rcp(dr.norm(w))
        w *= J
        B = dr.dot(w, p)
        C = dr.dot(p, p)
        A = sigma * J * dr.exp(-0.5 * dr.fma(-B, B, C)) \
            * dr.rcp(2.0 * dr.pi * dr.prod(s))
        sqrt_half = dr.sqrt(0.5)

        if truncate:
            # Whitened arc length s_w = t / J (s_w = t * ||w_unscaled||). The
            # forward-relevant shell segment is [max(t_in, 0), t_out]; the
            # density-integral lower/upper CDF anchors are erf at those arcs.
            valid, t_in, t_out = ray_ellipsoid_intersection(ray, ellipsoid, active)
            s_in = dr.maximum(t_in, 0.0) * dr.rcp(J)
            s_out = t_out * dr.rcp(J)
            e_lo = dr.erf(sqrt_half * (s_in + B))
            e_hi = dr.erf(sqrt_half * (s_out + B))
            arg = e_lo + 2.0 * chi / dr.maximum(A, 1e-30)
            t = J * (dr.sqrt(2.0) * dr.erfinv(arg) - B)
            escape = dr.isnan(t) | (arg >= e_hi) | ~valid
            t = dr.select(escape, dr.inf, t)
            return dr.select(active & (chi > 0.0), t, dr.inf)

        arg = dr.erf(sqrt_half * B) + 2.0 * chi / dr.maximum(A, 1e-30)
        t = J * (dr.sqrt(2.0) * dr.erfinv(arg) - B)
        t = dr.select(dr.isnan(t) | (arg >= 1.0), dr.inf, t)
        return dr.select(active & (chi > 0.0), t, dr.inf)


    def _sigma_eff(self, shape, ellipsoid, ray_d, active, for_nee=False):
        '''Per-primitive extinction. Gaussian-only single-level integrator: no
        LOD / orientation / continuous-frequency weighting (that machinery lives
        in gfields_prb for the pyramid/steerable Gabor assets), so this is just
        the opacity.'''
        return ellipsoid.opacity

    @staticmethod
    def _contains(shape, index, p, active):
        '''Whether point `p` lies inside the primitive's extent shell.'''
        ell = Ellipsoid.gather(shape, index, active)
        q = ell.rot.T * (p - ell.center) / (ell.scale * ell.extent)
        r2 = dr.squared_norm(q)
        return active & dr.isfinite(r2) & (r2 <= 1.0)

    @dr.syntax
    def _filter_enclosing(self, primitives, p, active):
        '''Compact `primitives` in place, keeping primitives containing `p`.'''
        count = mi.UInt32(0)
        it = mi.UInt32(0)
        active1 = mi.Bool(active) & ~primitives.is_empty()
        while active1:
            prim_id = primitives.value(it, active1)
            keep = self._contains(prim_id.shape, prim_id.index, p, active1)
            primitives.write(prim_id, count, keep)
            count[keep] += 1
            it += 1
            active1 &= (it < primitives.size())
        primitives.resize(count, active)
        return primitives

    @staticmethod
    def _select_walk_mask(sampler, ray, active):
        '''Gaussian-only single-level integrator: every level is always visible
        (no stochastic-Laplacian / orientation per-ray selection).'''
        return mi.UInt32(255)

    # ------------------------------------------------------------------
    # Any-hit fast paths: single-traversal distance sampling + coverage
    # ------------------------------------------------------------------

    def _anyhit_sampling_supported(self, scene):
        '''The any-hit distance-sampling queries implement the plain-opacity
        Gaussian free flight (no LOD weighting). Disabled under
        truncate_at_shell: the shader inverts with its own support convention,
        so shell-truncated sampling stays on the Python walk (which truncates
        each per-primitive flight to [t_in, t_out]).'''
        return (self.use_anyhit and self.use_anyhit_sampling
                and not self.truncate_at_shell
                and hasattr(scene, 'ray_sample_ellipsoids')
                and hasattr(scene, 'ray_collect_ellipsoids'))

    def _collect_range(self, scene):
        '''Upper bound on any shell's forward exit distance from a contained
        point: twice the largest shell radius across the scene's ellipsoid
        shapes (max scale times the extent multiplier). Bounding the collect
        ray this tightly is what keeps the coverage traversal from sweeping
        every primitive ahead of the event point. Falls back to the scene
        bbox diagonal when the shape parameters are unavailable.'''
        r_max = 0.0
        try:
            for sh in scene.shapes():
                if not sh.is_ellipsoids():
                    continue
                sp = mi.traverse(sh)
                data = sp['data']
                ext = float(sp['extent'])
                n = dr.width(data) // 10
                idx = dr.arange(mi.UInt32, n) * 10
                s = dr.maximum(dr.maximum(
                    dr.gather(mi.Float, data, idx + 3),
                    dr.gather(mi.Float, data, idx + 4)),
                    dr.gather(mi.Float, data, idx + 5))
                r_max = max(r_max, float(dr.max(s)[0]) * ext)
        except (KeyError, RuntimeError, TypeError):
            r_max = 0.0
        if r_max <= 0.0:
            bb = scene.bbox()
            return float(dr.norm(bb.max - bb.min)) * 1.01
        return 2.0 * r_max * 1.05

    def _collect_covering(self, scene, x, d, mask, primitives, active):
        '''Rebuild `primitives` as the covering set at point `x` with one
        any-hit traversal (Scene.ray_collect_ellipsoids): the shader
        appends the primitives whose extent shells contain `x` to this
        lane's slot of the preallocated buffer. Lanes outside `active`
        keep their current stack.'''
        k_cap = int(self.max_overlaps)
        cray = dr.zeros(mi.Ray3f)
        cray.o = mi.Point3f(x)
        cray.d = mi.Vector3f(d)
        cray.maxt = mi.Float(self._ah_collect_maxt)
        cray.mask = mi.UInt32(mask)
        count = scene.ray_collect_ellipsoids(cray, self._ah_slot_addr,
                                             k_cap, active)
        count = dr.minimum(count, mi.UInt32(k_cap))
        primitives.resize(count, active)
        return self._fill_stack_from_collect(primitives, count, active)

    @dr.syntax
    def _fill_stack_from_collect(self, primitives, count, active):
        '''Copy this lane's collected (registry id, prim index) pairs into
        the stack. The buffer writes happen behind the JIT's back: folding
        the count into every gather index orders the reads after the
        traversal. A symbolic loop over the actual count -- typical
        coverage is a handful -- not an unrolled capacity loop.'''
        base = self._ah_slot * mi.UInt32(2 * int(self.max_overlaps)) \
            + (count & mi.UInt32(0))
        k = mi.UInt32(0)
        active1 = mi.Bool(active) & (k < count)
        while active1:
            reg = dr.gather(mi.UInt32, self._ah_buf, base + 2 * k, active1)
            prm = dr.gather(mi.UInt32, self._ah_buf, base + 2 * k + 1, active1)
            shape_ptr = dr.reinterpret_array(mi.ShapePtr, reg)
            primitives.write(PrimitiveID(shape_ptr, prm), k, active1)
            k += 1
            active1 &= (k < count)
        return primitives

    def _sample_interaction_gaussian_anyhit(self, scene, sampler, ray, depth,
                                            primitives, active):
        '''Single-traversal replacement for the Gaussian free-flight walk:
        `Scene.ray_sample_ellipsoids` min-folds every intersected
        primitive's inverse-CDF flight (primitives covering the ray origin
        included via interior-hit reporting) and commits the closest
        blocking surface; a collect traversal then rebuilds the covering
        stack at the event point. Same free-flight law as the walk -- the
        minimum over independent per-primitive flights -- so the estimator
        is unchanged; only the walk-length RR/caps do not apply (weight is
        always 1).'''
        active = mi.Bool(active)
        ray = mi.Ray3f(ray)
        ray.mask = self._select_walk_mask(sampler, ray, active)

        seed = mi.UInt32(sampler.next_1d() * 4294967296.0)
        t_q, _win_s, _win_p, pi = scene.ray_sample_ellipsoids(
            ray, seed, mi.Float(0.0), active)

        # The query clips flights at the committed surface: t_q is +inf
        # whenever the surface (or escape) wins.
        sampled_t = dr.select(active, t_q, dr.inf)
        surface_valid = active & pi.is_valid() & ~(sampled_t < dr.inf)
        surface_si = pi.compute_surface_interaction(
            ray, mi.RayFlags.All, active & pi.is_valid())
        surface_si.t = dr.select(surface_valid, pi.t, dr.inf)

        p_evt = dr.select(sampled_t < dr.inf, ray(sampled_t),
                          dr.select(surface_valid, surface_si.p,
                                    mi.Point3f(dr.inf)))
        primitives = self._collect_covering(
            scene, p_evt, ray.d, ray.mask, primitives,
            active & ((sampled_t < dr.inf) | surface_valid))
        return surface_si, sampled_t, mi.Float(1.0), primitives


    # ------------------------------------------------------------------
    # Distance sampling, Gaussian kernel (exact min-of-free-flights walk)
    # ------------------------------------------------------------------

    @dr.syntax
    def _sample_interaction_gaussian(self, scene, sampler, ray, depth,
                                     primitives, active):
        '''One inverse-CDF free flight per primitive; the shortest sampled
        distance wins. `primitives` holds the primitives covering the ray
        origin and is extended with the primitives whose shells the walk
        enters (BVH discovery; coverage for the albedo / NEE at the sampled
        vertex).'''
        active = mi.Bool(active)
        ray = mi.Ray3f(ray)
        ray.mask = self._select_walk_mask(sampler, ray, active)
        ray0 = mi.Ray3f(ray)            # all flights anchor at the segment origin

        weight     = mi.Float(1.0)
        t_cand     = mi.Float(dr.inf)   # candidate medium event (world units)
        accum_t    = mi.Float(0.0)
        surface_si = dr.zeros(mi.SurfaceInteraction3f)
        surface_t  = mi.Float(dr.inf)

        # ---- 1. Primitives covering the segment origin.
        it0 = mi.UInt32(0)
        active0 = mi.Bool(active) & ~primitives.is_empty()
        while active0:
            prim_id = primitives.value(it0, active0)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active0)
            sigma = self._sigma_eff(prim_id.shape, ellipsoid, ray0.d, active0)
            chi = -dr.log(dr.maximum(1.0 - sampler.next_1d(), 1e-38))
            t_i = self._inv_cdf(ray0, ellipsoid, sigma, chi, active0,
                                self.truncate_at_shell)
            t_cand = dr.minimum(t_cand, t_i)
            it0 += 1
            active0 &= (it0 < primitives.size())

        # ---- 2. Walk newly discovered primitives front-to-back.
        it = mi.UInt32(0)
        active1 = mi.Bool(active)
        while active1:
            si = scene.ray_intersect(ray,
                                     coherent=(depth == 0) & (it == 0),
                                     ray_flags=mi.RayFlags.All,
                                     active=active1)
            hit = active1 & si.is_valid()
            is_prim = hit & si.shape.is_ellipsoids()
            hit_surface = hit & ~is_prim

            # Blocking surface: record and stop (interactions sampled beyond
            # it are discarded at resolution time, which conditions every
            # per-primitive free flight on the surface exactly).
            surface_si[hit_surface] = si
            surface_t[hit_surface] = accum_t + si.t

            # Primitives discovered at or beyond the current candidate
            # cannot produce an earlier collision (up to the negligible
            # kernel mass ahead of their discovery point).
            proc = is_prim & (accum_t + si.t < t_cand)

            ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, proc)
            sigma = self._sigma_eff(si.shape, ellipsoid, ray0.d, proc)
            chi = -dr.log(dr.maximum(1.0 - sampler.next_1d(), 1e-38))
            # Flight from the segment origin (identical law to the any-hit
            # query; the walk ray only serves discovery). Unbounded by default;
            # shell-truncated to [t_in, t_out] when truncate_at_shell.
            t_i = self._inv_cdf(ray0, ellipsoid, sigma, chi, proc,
                                self.truncate_at_shell)
            t_cand[proc] = dr.minimum(t_cand, t_i)

            # Remember the primitive: it may contain the interaction point
            # (albedo coverage, continuation and NEE rays starting inside).
            full = proc & (primitives.size() >= primitives.alloc_size())
            primitives = self._filter_enclosing(
                primitives, mi.Point3f(ray.o), full)
            primitives.push(PrimitiveID(si.shape, si.prim_index),
                            proc & (primitives.size() < primitives.alloc_size()))

            # Russian roulette / hard cap on the walk length (parity with
            # primitive_tracing).
            it += 1
            if self.rr_depth_primitive > 0:
                q = dr.minimum(weight, 0.9)
                perform_rr = (it > self.rr_depth_primitive)
                valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
                weight[proc & perform_rr] *= dr.rcp(q)
                weight[proc & ~valid_rr] = 0.0
                proc &= valid_rr
            if self.max_depth_primitive > 0:
                weight[proc & (it >= self.max_depth_primitive)] = 0.0
                proc &= (it < self.max_depth_primitive)

            ray.o = spawn_ray_origin(si, ray)
            accum_t += si.t
            active1 &= proc

        sampled_t = dr.select(t_cand < surface_t, t_cand, dr.inf)
        surface_valid = surface_si.is_valid() & ~(t_cand < surface_t)
        surface_si.t = dr.select(surface_valid, surface_t, dr.inf)
        return surface_si, sampled_t, weight, primitives


    def sample_interaction_stochastic(self, scene, sampler, ray, depth,
                                      primitives, active):
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            si = scene.ray_intersect(ray, coherent=(depth == 0),
                                     ray_flags=mi.RayFlags.All, active=active)
            return si, mi.Float(dr.inf), mi.Float(1.0), primitives
        # Any-hit fast path (state prepared in sample()'s preamble; absent when
        # this method is driven externally -> closest hit mode). Gaussian-only: the
        # exact free-flight sampler has no acceptance test.
        if dr.hint(getattr(self, '_ah_sampling', False), mode='scalar'):
            return self._sample_interaction_gaussian_anyhit(
                scene, sampler, ray, depth, primitives, active)
        return self._sample_interaction_gaussian(
            scene, sampler, ray, depth, primitives, active)

    # ------------------------------------------------------------------
    # Analytic NEE transmittance (single-traversal deterministic fast path)
    # ------------------------------------------------------------------

    def eval_transmittance_nee(self, scene, sampler, ray, primitives, L, δL,
                               mode, active):
        '''Analytic NEE via the deterministic any-hit shader (default; on by
        `use_anyhit`, like gfields_tomography). One TransmittanceEllipsoids
        traversal accumulates the full-domain signed optical depth of every
        primitive the shadow ray ENTERS on the way to the emitter, in arbitrary
        (order-independent) order, zero added variance; opaque blockers return
        +inf. Primitives already overlapping the shadow-ray origin are handled
        separately from the coverage stack: their forward optical depth from the
        origin, integrated either unbounded over [0, maxt] (default, fast) or --
        with `nee_truncate_at_shell` -- over the exact segment to the shell exit.
        Falls back to the parent's walk for the adjoint replay
        (grad_nee_transmittance) and weighted-LOD configs.'''
        if dr.hint(mode == dr.ADMode.Primal
                   and self.use_anyhit
                   and not self.grad_nee_transmittance
                   and hasattr(scene, 'ray_transmittance_ellipsoids')
                   and has_ellipsoids_shapes(scene), mode='scalar'):
            return self._eval_transmittance_nee_anyhit(scene, ray,
                                                       primitives, active)
        return super().eval_transmittance_nee(scene, sampler, ray,
                                              primitives, L, δL, mode, active)

    @dr.syntax
    def _eval_transmittance_nee_anyhit(self, scene, ray, primitives, active):
        active = mi.Bool(active)
        ray = mi.Ray3f(ray)
        total_tau = mi.Float(0.0)

        # Primitives already overlapping the shadow-ray origin: forward optical
        # depth from the origin, either unbounded over [0, maxt] (default, fast)
        # or -- with nee_truncate_at_shell -- over the exact segment to the shell
        # exit (ray_ellipsoid_intersection, capped at the emitter), mirroring
        # gfields_tomography.eval_transmission. The down-ray full-domain query
        # below does not see these primitives (their entry is behind the origin,
        # back-face culled), so there is no double count.
        it0 = mi.UInt32(0)
        active0 = mi.Bool(active) & ~primitives.is_empty()
        while active0:
            prim_id = primitives.value(it0, active0)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active0)
            sigma = self._sigma_eff(prim_id.shape, ellipsoid, ray.d, active0,
                                    for_nee=True)
            if dr.hint(self.nee_truncate_at_shell, mode='scalar'):
                _valid, _t_in, t_out = ray_ellipsoid_intersection(
                    ray, ellipsoid, active0)
                seg_hi = dr.minimum(t_out, ray.maxt)
            else:
                seg_hi = ray.maxt
            tau = sigma * self.kernel.density_integral(
                ray, ellipsoid, mi.Float(0.0), seg_hi, active0)
            total_tau[active0 & dr.isfinite(tau)] += tau
            it0 += 1
            active0 &= (it0 < primitives.size())

        # Every primitive entered on the way to the emitter: one traversal
        # returning the summed full-line signed optical depth.
        ray.maxt = ray.maxt * (1.0 - 1e-4)
        ray.mask = mi.UInt32(255)  # gate requires deterministic NEE LOD
        tau_q = scene.ray_transmittance_ellipsoids(
            ray, mi.UInt32(0), int(mi.RayFlags.TransmittanceEllipsoids),
            active)
        total_tau += dr.select(dr.isfinite(tau_q), tau_q, mi.Float(1e30))

        return dr.select(active, dr.clip(dr.exp(-total_tau), 0.0, 1.0), 1.0)

    # ------------------------------------------------------------------
    # Stochastic NEE transmittance (early-terminating Bernoulli walk)
    # ------------------------------------------------------------------

    @dr.syntax
    def eval_transmittance_nee_stochastic(self, scene, sampler, ray,
                                          primitives, L, δL, mode, active):
        '''
        Unbiased shadow-ray transmittance: one Bernoulli test per intersected
        primitive with the analytic per-primitive transmittance as survival
        probability, terminating the walk at the first absorption (dense
        media cost O(1) instead of O(#primitives to the emitter)).
        Net-amplifying signed primitives (tau < 0) multiply their bounded
        analytic factor deterministically; the final clamp to [0, 1] matches
        `gfields_prb.eval_transmittance_nee`.

        Gradient (`mode != Primal`): the Bernoulli walk makes T̂ a *discrete*
        estimator (primitive i survives with probability p_i = exp(-tau_i) for
        tau_i >= 0, E[T̂] = prod exp(-tau_i) = T). Its score-function derivative
        is d ln p_i / dθ = -d tau_i / dθ per *survived* primitive, and the
        deterministic exp(-tau_i) amplification of signed (tau_i < 0)
        primitives contributes the same -d tau_i / dθ. So each processed
        primitive injects
        δL · L · (-tau_i) with L the detached adjoint NEE radiance (Lr_nee);
        rays absorbed downstream carry Lr_nee = 0 (T̂ = 0) and drop out, so the
        estimator reduces in expectation to the true C · ∂T/∂θ. Call the
        gradient pass with the value walk's cloned sampler so the replayed
        survival decisions (and hence the survived-primitive set) match.
        '''
        if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
            return mi.Float(1.0)

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(ray)
        L = dr.detach(L)
        total_tr = mi.Float(1.0)

        # Primitives covering the shadow-ray origin: the exact segment from the
        # origin (inside the shell) to the shell exit, capped at the emitter --
        # keeping all three NEE estimators (analytic any-hit, Python Bernoulli,
        # any-hit Bernoulli) mutually consistent and matching the parent walk.
        it0 = mi.UInt32(0)
        active0 = mi.Bool(active) & ~primitives.is_empty()
        while active0:
            prim_id = primitives.value(it0, active0)
            with dr.resume_grad(when=not primal):
                ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active0)
                sigma = self._sigma_eff(prim_id.shape, ellipsoid, ray.d, active0,
                                        for_nee=True)
                if dr.hint(self.nee_truncate_at_shell, mode='scalar'):
                    _valid, _t_in, t_out = ray_ellipsoid_intersection(
                        ray, ellipsoid, active0)
                    # Detach the shell-exit bound: differentiate the integrand only.
                    seg_hi = dr.detach(dr.minimum(t_out, ray.maxt))
                else:
                    seg_hi = ray.maxt
                tau = sigma * self.kernel.density_integral(
                    ray, ellipsoid, mi.Float(0.0), seg_hi, active0)
                tau_det = dr.detach(tau)
                # Survival decision is detached (replayed identically in the
                # adjoint pass); differentiate only the injected -tau term.
                alive = active0 & (sampler.next_1d() < dr.exp(-dr.maximum(tau_det, 0.0)))
                if not primal:
                    contrib = dr.select(alive & dr.isfinite(tau), -tau * L, 0.0)
                    if dr.grad_enabled(contrib):
                        if mode == dr.ADMode.Backward:
                            dr.backward_from(δL * contrib)
                        else:
                            δL += dr.forward_to(contrib)
            total_tr[active0 & ~alive] = 0.0
            total_tr[alive & (tau_det < 0.0)] *= dr.exp(-tau_det)
            it0 += 1
            active0 &= alive & (it0 < primitives.size())
        active = mi.Bool(active) & (total_tr > 0.0)

        # Walk to the emitter with per-hit Bernoulli termination. Gaussian-only:
        # every level is visible (no per-ray NEE LOD selection).
        ray.maxt = ray.maxt * (1.0 - 1e-4)
        ray.mask = mi.UInt32(255)

        if dr.hint(primal and self.use_anyhit
                   and not self.grad_nee_transmittance
                   and hasattr(scene, 'ray_transmittance_ellipsoids'), mode='scalar'):
            # One any-hit traversal performs every per-primitive Bernoulli
            # test inside the shader (stateless per-primitive hash of the
            # seed, so unordered BVH traversal is exact) and terminates at
            # the first absorption. Hard-surface hits block the ray.
            seed = mi.UInt32(sampler.next_1d() * 4294967296.0)
            t_walk = scene.ray_transmittance_ellipsoids(
                ray, seed, int(mi.RayFlags.StochasticEllipsoids), active)
            total_tr[active] *= t_walk
        else:
            it = mi.UInt32(0)
            active2 = mi.Bool(active)
            while active2:
                si = scene.ray_intersect(ray, coherent=False,
                                         ray_flags=mi.RayFlags.All, active=active2)
                active2 &= si.is_valid()

                is_prim = active2 & si.shape.is_ellipsoids()
                hit_hard = active2 & ~is_prim
                total_tr[hit_hard] = 0.0
                active2 &= is_prim

                with dr.resume_grad(when=not primal):
                    ellipsoid = Ellipsoid.gather(si.shape, si.prim_index, active2)
                    sigma = self._sigma_eff(si.shape, ellipsoid, ray.d, active2,
                                            for_nee=True)
                    tau = sigma * self.kernel.density_integral(
                        ray, ellipsoid, None, None, active2)
                    tau_det = dr.detach(tau)
                    alive = active2 & (sampler.next_1d() < dr.exp(-dr.maximum(tau_det, 0.0)))
                    if not primal:
                        contrib = dr.select(alive & dr.isfinite(tau), -tau * L, 0.0)
                        if dr.grad_enabled(contrib):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * contrib)
                            else:
                                δL += dr.forward_to(contrib)
                total_tr[active2 & ~alive] = 0.0
                total_tr[alive & (tau_det < 0.0)] *= dr.exp(-tau_det)
                active2 &= alive

                if self.max_depth_primitive > 0:
                    total_tr[active2 & (it >= self.max_depth_primitive)] = 0.0
                    active2 &= (it < self.max_depth_primitive)

                ray.o = spawn_ray_origin(si, ray)
                ray.maxt -= si.t
                it += 1

        return dr.clip(total_tr, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main sampling loop (parent structure, stochastic volumetric block)
    # ------------------------------------------------------------------

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        check_ellipsoids_attribute(scene, ['opacities', 'albedo'])

        primal = (mode == dr.ADMode.Primal)
        ray = mi.Ray3f(dr.detach(ray))
        active    = mi.Bool(active)
        valid_ray = mi.Bool(False)
        depth     = mi.UInt32(0)

        prev_event = dr.zeros(mi.Interaction3f)
        prev_event_pdf = mi.Float(1.0)

        bsdf_ctx  = mi.BSDFContext()
        phase_ctx = mi.PhaseFunctionContext(sampler)

        if not primal:
            active &= dr.any((δL != 0))

        L  = mi.Spectrum(0.0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β  = mi.Spectrum(1.0)

        # Primitives covering the current path vertex (starts empty: camera
        # origins inside primitives are not handled, as in gfields_prb --
        # except by the any-hit sampler, whose interior-hit reporting sees
        # them regardless of the stack).
        primitives = alloc_stack(PrimitiveID, mi.UInt32, alloc_size=self.max_overlaps)

        # Any-hit fast-path state: distance sampling and coverage collection
        # replace the hit walks with single traversals. The collect buffer is
        # preallocated and evaluated here because the any-hit shader writes
        # it behind the JIT's back inside the symbolic loop below (every
        # gather folds the returned count into its index to order after the
        # traversal).
        self._ah_sampling = self._anyhit_sampling_supported(scene)
        if dr.hint(self.use_anyhit and self.use_anyhit_sampling and
                   not getattr(self, '_ah_report_done', False), mode='scalar'):
            self._ah_report_done = True
            mi.Log(mi.LogLevel.Info,
                   '[gfields_prb_stochastic] use_anyhit_sampling: '
                   + ('ENGAGED' if self._ah_sampling else
                      'FALLBACK to hit walk (unsupported scene/LOD config)'))
        if dr.hint(self._ah_sampling, mode='scalar'):
            k_cap = int(self.max_overlaps)
            lanes = max(dr.width(ray.o), 1)
            buf = dr.zeros(mi.UInt32, 2 * k_cap * lanes)
            dr.eval(buf)
            self._ah_buf = buf
            self._ah_slot = dr.arange(mi.UInt32, lanes)
            # Per-lane slot addresses, resolved here: device pointers cannot
            # be queried inside the symbolic loop below.
            base = mi.device_address(buf)
            self._ah_slot_addr = (mi.UInt64(base) + mi.UInt64(self._ah_slot)
                                  * mi.UInt64(2 * k_cap * 4))
            self._ah_collect_maxt = self._collect_range(scene)

        while active:

            # --------------------------------------------------------------
            # Stochastic volumetric interaction sampling
            # --------------------------------------------------------------

            si, sampled_t, weight, primitives = self.sample_interaction_stochastic(
                scene, sampler, ray, depth, primitives, active)

            β *= weight
            active &= dr.any(β != 0.0)

            mei = dr.zeros(mi.MediumInteraction3f)
            mei.wi = dr.normalize(-ray.d)
            mei.sh_frame = mi.Frame3f(mei.wi)
            mei.t = sampled_t
            mei.p = ray(sampled_t)

            active_surface = active & ~mei.is_valid() & si.is_valid()
            active_medium  = active &  mei.is_valid()
            escaped_medium = active & ~mei.is_valid() & ~si.is_valid()
            valid_ray |= active_medium | active_surface

            # Reduce the stack to the primitives covering the event point
            # (escaped rays keep an empty stack: nothing contains infinity).
            p_evt = dr.select(active_medium, mei.p,
                              dr.select(active_surface, si.p, mi.Point3f(dr.inf)))
            primitives = self._filter_enclosing(primitives, p_evt, active)

            depth_before_scatter = mi.UInt32(depth)

            depth[active_surface | active_medium] += 1
            active_medium  &= depth < self.max_depth
            active_surface &= depth < self.max_depth

            # Transmittance derivatives (PRB): reuse the parent's analytic
            # walk, whose expectation matches this estimator.
            if not primal:
                ray2 = dr.zeros(mi.Ray3f)
                ray2.o = mi.Point3f(ray.o)
                ray2.d = dr.select(escaped_medium, -mei.wi, dr.normalize(dr.select(active_surface, si.p, mei.p) - ray2.o))
                ray2.maxt = dr.select(escaped_medium, dr.inf, dr.norm(ray2.d))
                ray2.mask = mi.UInt32(255)
                active2 = active_medium | active_surface | (escaped_medium & mi.Bool(self.use_indirect))
                self.eval_transmittance(scene, sampler, ray2, primitives, L, δL, mode, active2)

            # --------------------------------------------------------------
            # Path throughput: mixture albedo over the covering primitives
            # (the parent's estimator and adjoint apply unchanged since the
            # filtered stack is exactly the covering set at mei.p).
            # --------------------------------------------------------------

            albedo = self.eval_albedo(scene, ray, mei.p, primitives, L, δL, mode, active_medium)
            β[active_medium] *= albedo

            # --------------------------------------------------------------
            # Intersection with emitters (area + environment)
            # --------------------------------------------------------------

            if self.use_indirect:
                emitter = si.emitter(scene)
                active_e_surface = active_surface & (emitter != None)

                env_emitter = scene.environment()
                active_e_env = escaped_medium & (env_emitter != None)

                active_e = active_e_surface | active_e_env
                active_e &= ~((depth_before_scatter == 0) & mi.Bool(self.hide_emitters))

                count_direct = (depth_before_scatter == 0)

                if self.use_nee:
                    ds = mi.DirectionSample3f(scene, si, prev_event)
                    emitter_pdf = scene.pdf_emitter_direction(prev_event, ds, active_e)
                else:
                    emitter_pdf = mi.Float(0.0)

                with dr.resume_grad(when=not primal):
                    emitter_val_surface = emitter.eval(si, active_e_surface)

                    emitter_val_env = mi.Spectrum(0.0)
                    if env_emitter is not None:
                        emitter_val_env = env_emitter.eval(si, active_e_env)

                    emitter_val = dr.select(active_e_surface, emitter_val_surface, emitter_val_env)

                    contrib_direct = β * emitter_val
                    contrib_indirect = β * mis_weight(prev_event_pdf, emitter_pdf) * emitter_val
                    Lr_dir = dr.select(count_direct, contrib_direct, contrib_indirect)
                    Lr_dir = dr.select(active_e, Lr_dir, 0.0)

                    if not primal:
                        if dr.grad_enabled(Lr_dir):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_dir)
                            else:
                                δL += dr.forward_to(Lr_dir)

                L = (L + Lr_dir) if primal else (L - Lr_dir)

            # --------------------------------------------------------------
            # Emitter sampling (NEE)
            # --------------------------------------------------------------

            bsdf = si.bsdf(ray)

            if dr.hint(self.use_nee, mode='scalar'):
                ref = dr.zeros(mi.Interaction3f)
                ref[active_medium]  = mei
                ref[active_surface] = si

                active_surface_nee = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
                nee_active = active_surface_nee | active_medium

                with dr.resume_grad(when=not primal):
                    ds, emitter_val = scene.sample_emitter_direction(ref, sampler.next_2d(), False, nee_active)

                    active_nee = nee_active & (ds.pdf > 0.0)

                    ray2 = ref.spawn_ray_to(ds.p)

                    ray2_valid = dr.all(dr.isfinite(ray2.d)) & (ray2.maxt > 1e-6)
                    active_nee = active_nee & ray2_valid

                    # Snapshot the sampler before the primal value walk so the
                    # adjoint transmittance replay below re-draws the identical
                    # Bernoulli / stochastic-LOD randoms (PRB path replay). Only
                    # needed when the (experimental) replay actually runs.
                    nee_sampler = (sampler.clone()
                                   if (not primal and self.grad_nee_transmittance)
                                   else sampler)

                    if dr.hint(self.nee_transmittance == 'stochastic', mode='scalar'):
                        transmittance = self.eval_transmittance_nee_stochastic(
                            scene, sampler, ray2, primitives, L, δL,
                            dr.ADMode.Primal, active_nee)
                    else:
                        transmittance = self.eval_transmittance_nee(
                            scene, sampler, ray2, primitives, L, δL,
                            dr.ADMode.Primal, active_nee)

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

                    if not primal:
                        if dr.grad_enabled(Lr_nee):
                            if mode == dr.ADMode.Backward:
                                dr.backward_from(δL * Lr_nee)
                            else:
                                δL += dr.forward_to(Lr_nee)

                # Adjoint NEE shadow-transmittance gradient (∂T_nee/∂θ), replayed
                # on the cloned sampler with adjoint radiance Lr_nee. The value
                # walk above evaluated `transmittance` in Primal mode (detached),
                # so the block above does not carry it; this re-walk injects it.
                # Both transmittance estimators support the mode-aware adjoint:
                # the analytic one via the detached-transmittance estimator, the
                # stochastic Bernoulli walk via the per-primitive score-function
                # gradient (see eval_transmittance_nee[_stochastic]).
                if not primal and self.grad_nee_transmittance:
                    if dr.hint(self.nee_transmittance == 'stochastic', mode='scalar'):
                        self.eval_transmittance_nee_stochastic(
                            scene, nee_sampler, ray2, primitives,
                            dr.detach(Lr_nee), δL, mode, active_nee)
                    else:
                        self.eval_transmittance_nee(
                            scene, nee_sampler, ray2, primitives,
                            dr.detach(Lr_nee), δL, mode, active_nee)

                L = (L + Lr_nee) if primal else (L - Lr_nee)

            # --------------------------------------------------------------
            # Phase function sampling
            # --------------------------------------------------------------

            with dr.suspend_grad():
                wo, phase_weight, phase_pdf_sampled = self.phase.sample(phase_ctx, mei,
                                                                        sampler.next_1d(),
                                                                        sampler.next_2d(),
                                                                        active_medium)
                wo = dr.normalize(wo)
                phase_val_unused, phase_pdf = self.phase.eval_pdf(phase_ctx, mei, wo, active_medium)

            active_medium &= (phase_pdf > 0.0) & dr.all(dr.isfinite(phase_weight))

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

            # --------------------------------------------------------------
            # BSDF sampling
            # --------------------------------------------------------------

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

            # --------------------------------------------------------------
            # Termination (Russian roulette)
            # --------------------------------------------------------------

            sample_rr = sampler.next_1d()
            if primal and self.use_rr:
                q = dr.minimum(dr.max(β), 0.99)
                perform_rr = (depth > self.rr_depth)
                active &= (sample_rr < q) | ~perform_rr
                β[perform_rr] = β * dr.rcp(q)

            active &= dr.any(β > 0.005)
            active &= dr.any((β != 0.0))
            active &= active_surface | active_medium

        return L if primal else δL, valid_ray, [], L

    def to_string(self):
        return 'GfieldsPRBStochasticIntegrator[]'


mi.register_integrator("gfields_prb_stochastic",
                       lambda props: GfieldsPRBStochasticIntegrator(props))
