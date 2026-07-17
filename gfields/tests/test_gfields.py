# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Different tests for gfields.

Usage:
    python -m pytest gfields/tests/test_gfields.py -q     # via pytest
    python gfields/tests/test_gfields.py                  # plain runner

All tests require a CUDA Mitsuba build with the gfields plugins.
They auto-skip when ``cuda_ad_rgb`` is unavailable.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest


def _cuda_available() -> bool:
    try:
        import mitsuba as mi
        return "cuda_ad_rgb" in mi.variants()
    except ImportError:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_available(),
                                reason="cuda_ad_rgb variant unavailable")


def _setup_mi():
    import mitsuba as mi
    if mi.variant() is None:
        mi.set_variant("cuda_ad_rgb")
    return mi


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

def _prims_dict(mi, centers, gabor, omega=5.0):
    # Gabor-ness is carried by omega alone: a Gaussian band has omega == 0.
    n = len(centers)
    omega_val = omega if gabor else 0.0
    return {
        "type": "ellipsoidsmesh", "shell": "uv_sphere",
        "centers": mi.TensorXf(sum(centers, []), shape=(n, 3)),
        "scales": mi.TensorXf([0.3] * (3 * n), shape=(n, 3)),
        "quaternions": mi.TensorXf([0, 0, 0, 1] * n, shape=(n, 4)),
        "opacities": mi.TensorXf([1.0] * n, shape=(n, 1)),
        "omega": mi.TensorXf([omega_val] * n, shape=(n, 1)),
        "albedo": mi.TensorXf([1, 1, 1] * n, shape=(n, 3)),
        "extent": 3.0,
    }


def _ortho_sensor(mi, res=64):
    T = mi.ScalarTransform4f
    return mi.load_dict({
        "type": "orthographic",
        "to_world": (T().look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0])
                     @ T().scale([1.5, 1.5, 1])),
        "film": {"type": "hdrfilm", "width": res, "height": res},
    })


# ---------------------------------------------------------------------------
# 1. density_integral additivity (closed-form erf line integral)
# ---------------------------------------------------------------------------

@needs_cuda
def test_density_integral_additivity():
    """A segmented sum of ``density_integral`` must equal the full-segment
    integral. This validates that the closed-form (erf-based) line integrals in
    ``integrators.common`` account for arbitrary segment positions -- the
    property that lets the integrators split a ray freely. ``GaborKernel``
    reduces to a Gaussian when ``beta == 0`` (cos(0) = 1) and exercises the
    Faddeeva erf-difference path when ``beta > 0``."""
    mi = _setup_mi()
    from gfields.integrators.common import (
        Ellipsoid, GaussianKernel, GaborKernel, ray_ellipsoid_intersection,
    )

    def ellipsoid(beta):
        return Ellipsoid(
            center=mi.Point3f([0, 0, 0]), scale=mi.Vector3f([1, 1, 1]),
            quat=mi.Quaternion4f([0, 0, 0, 1]),
            rot=mi.Matrix3f([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            extent=mi.Float(3.0), beta=mi.Float(beta), opacity=mi.Float(1.0))

    ray = mi.Ray3f(o=mi.Point3f([-5.0, 0.0, 0.0]), d=mi.Vector3f([1.0, 0.0, 0.0]))
    active = mi.Bool(True)

    def additivity_error(kernel, ell, num_segs):
        _, t_min, t_max = ray_ellipsoid_intersection(ray, ell, active)
        full = float(kernel.density_integral(ray, ell, t_min, t_max, active)[0])
        seg_len = (t_max - t_min) / num_segs
        total = sum(
            float(kernel.density_integral(
                ray, ell, t_min + i * seg_len, t_min + (i + 1) * seg_len, active)[0])
            for i in range(num_segs))
        return abs(total - full) / max(abs(full), 1e-10)

    # GaussianKernel and GaborKernel in Gaussian mode (beta = 0): analytic erf,
    # exact additivity.
    ell0 = ellipsoid(0.0)
    for kernel, name in ((GaussianKernel(mi.Properties()), "GaussianKernel"),
                         (GaborKernel(mi.Properties()), "GaborKernel(beta=0)")):
        for num_segs in (2, 10, 50):
            err = additivity_error(kernel, ell0, num_segs)
            assert err < 1e-4, f"{name} additivity failed ({num_segs} segs): rel {err:.2e}"

    # GaborKernel in Gabor mode (beta = 2); looser tol (Faddeeva/Taylor erf).
    gabor = GaborKernel(mi.Properties())
    ell2 = ellipsoid(2.0)
    for num_segs in (2, 10, 20):
        err = additivity_error(gabor, ell2, num_segs)
        assert err < 1e-2, f"GaborKernel(beta=2) additivity failed ({num_segs} segs): rel {err:.2e}"


# ---------------------------------------------------------------------------
# 2. continuous LOD equivalences (integrator freq_threshold)
# ---------------------------------------------------------------------------

@needs_cuda
def test_continuous_lod_equivalences():
    """freq_threshold=inf must equal filter-off; threshold=0 must equal a
    Gaussian-only scene (all Gabors faded out)."""
    mi = _setup_mi()
    import gfields  # noqa: F401

    sensor = _ortho_sensor(mi)
    gabor_shape = _prims_dict(mi, [[0.4, 0, 0]], True)
    gauss_shape = _prims_dict(mi, [[-0.4, 0, 0]], False)

    def render(threshold, include_gabor=True):
        d = {"type": "scene", "e": {"type": "constant"},
             "integrator": {"type": "gfields_tomography",
                            "kernel_type": "gabor_gaussian_mixture",
                            "kernel_full_range": False,
                            "kernel_normalized": False,
                            "freq_threshold": threshold}}
        if include_gabor:
            d["p1"] = dict(gabor_shape)
        d["p2"] = dict(gauss_shape)
        return np.array(mi.render(mi.load_dict(d), sensor=sensor, spp=16, seed=0))

    off, huge, zero = render(-1.0), render(1e6), render(0.0)
    gauss_only = render(-1.0, include_gabor=False)

    # A very large threshold makes every band fully visible: must match filter-off.
    assert np.allclose(huge, off, atol=1e-5)

    # A zero threshold fades the Gabor to weight 0, collapsing the render onto
    # the Gaussian-only scene. The small residual (~2e-3) is the integrator
    # still walking the now-zero-weight Gabor shell where it overlaps the
    # Gaussian (vs removing the primitive entirely), so compare loosely.
    assert np.allclose(zero, gauss_only, atol=3e-3)


# ---------------------------------------------------------------------------
# 3. numpy <-> Dr.Jit LOD band-weight parity
# ---------------------------------------------------------------------------

@needs_cuda
def test_band_weight_numpy_drjit_parity():
    """The numpy band weights and the Dr.Jit continuous_lod_weight -- both in
    gfields.lod -- must agree."""
    mi = _setup_mi()
    from gfields.integrators.common import Ellipsoid
    from gfields.lod import (
        continuous_lod_weight, gabor_band_weights, gabor_frequency_band,
    )

    rng = np.random.default_rng(7)
    n = 256
    scales = rng.uniform(0.02, 0.5, size=(n, 3)).astype(np.float32)
    omega = rng.uniform(0.5, 8.0, size=n).astype(np.float32)
    threshold, band_sigma = 30.0, 2.0

    f_lo, f_hi = gabor_frequency_band(omega, scales, band_sigma=band_sigma)
    w_np = gabor_band_weights(f_lo, f_hi, threshold)

    ell = Ellipsoid(
        center=mi.Point3f(0), scale=mi.Vector3f(scales[:, 0], scales[:, 1], scales[:, 2]),
        quat=mi.Quaternion4f([0, 0, 0, 1]), rot=mi.Matrix3f([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        extent=mi.Float(3.0), beta=mi.Float(omega), opacity=mi.Float(1.0))
    w_dr = continuous_lod_weight(ell, mi.Float(threshold), band_sigma, mi.Bool(True))
    assert np.allclose(w_dr.numpy(), w_np, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. LODSelector config + steerable visibility-mask dispatch
# ---------------------------------------------------------------------------

@needs_cuda
def test_lod_selector_config_and_masks():
    """The LODSelector behind the steerable integrators: orientation-set sizes,
    the sub-band bit layout (bit 7 = Gaussian, descending per level/orientation),
    invalid-config rejection, and that every freq x ori mode combination loads."""
    mi = _setup_mi()
    import gfields  # noqa: F401  (registers gfields_prb)

    def load(**kw):
        base = {"type": "gfields_prb", "asset_encoding": "steerable",
                "pyramid_level_count": 3, "num_orientations": 3}
        return mi.load_dict({**base, **kw})

    # Orientation-set sizes
    for num_ori in (2, 3, 4, 6, 7):
        integ = load(pyramid_level_count=2, num_orientations=num_ori)
        assert len(integ.lod.orientations) == num_ori

    # Sub-band bit layout: Gaussian at bit 7, then level 1 -> 6,5,4; level 2 -> 3,2,1
    integ = load()
    assert integ.lod.subband_bit(0) == 7
    assert [integ.lod.subband_bit(1, o) for o in range(3)] == [6, 5, 4]
    assert [integ.lod.subband_bit(2, o) for o in range(3)] == [3, 2, 1]

    # Invalid configurations must be rejected
    for bad in ({"orientation_selection": "invalid_mode"},
                {"stochastic_selection_criteria": "invalid_mode"},
                {"pyramid_level_count": 8}):  # 1 + 7*3 = 22 sub-bands > 8-bit mask
        with pytest.raises(Exception):
            load(**bad)

    # Every freq x orientation selection mode must construct
    freq_modes = ("deterministic", "power_law",
                  "control_variate_uniform", "control_variate_power_law")
    ori_modes = ("deterministic", "threshold", "importance",
                 "control_variate_threshold")
    for fm in freq_modes:
        for om in ori_modes:
            load(stochastic_selection_criteria=fm, orientation_selection=om)


# ---------------------------------------------------------------------------
# 5. gfields_prb_stochastic vs analytic tomography and gfields_prb
# ---------------------------------------------------------------------------

def _prb_prim_cloud(mi, n=200, gabor=False, seed=3):
    """Random primitive cloud. Gabor mode pairs each gabor with a nearby,
    slightly larger gaussian base so the mixture density stays non-negative
    (the structure of trained gabor-field assets). Pairs are offset because
    exactly co-located shells are degenerate for hit-walking integrators."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32)
    scales = rng.uniform(0.05, 0.15, (n, 3)).astype(np.float32)
    quats = np.tile([0, 0, 0, 1.0], (n, 1)).astype(np.float32)
    opac = rng.uniform(0.2, 1.2, (n, 1)).astype(np.float32)
    alb = np.full((n, 3), 0.8, np.float32)
    if gabor:
        omega = np.concatenate([rng.uniform(1.0, 4.0, (n, 1)),
                                np.zeros((n, 1))]).astype(np.float32)
        centers = np.concatenate([centers + 0.15 * scales, centers])
        scales = np.concatenate([0.95 * scales, scales])
        quats = np.tile(quats, (2, 1))
        opac = np.concatenate([0.85 * opac, opac])
        alb = np.tile(alb, (2, 1))
    else:
        omega = np.zeros((n, 1), np.float32)
    return {
        "type": "ellipsoidsmesh", "shell": "uv_sphere",
        "centers": mi.TensorXf(centers), "scales": mi.TensorXf(scales),
        "quaternions": mi.TensorXf(quats), "opacities": mi.TensorXf(opac),
        "albedo": mi.TensorXf(alb), "omega": mi.TensorXf(omega),
        "extent": 3.0,
    }


@needs_cuda
def test_prb_stochastic_vs_references():
    """gfields_prb_stochastic (GAUSSIAN-only) must match the deterministic
    analytic tomography reference on transmittance and gfields_prb on full
    transport, and assert finiteness everywhere. (The Gabor case is removed:
    the integrator errors on non-Gaussian kernels -- its signed-density sampler
    is biased for Gabor mixtures; use gfields_prb there.)"""
    mi = _setup_mi()
    import gfields  # noqa: F401
    T = mi.ScalarTransform4f

    KERNEL = {"kernel_type": "gaussian", "kernel_normalized": True}
    BASE = {**KERNEL, "max_overlaps": 64, "rr_depth_primitive": 1024,
            "max_depth_primitive": 2048}
    # Transport compares against gfields_prb, which truncates transport at the
    # extent shells while this integrator samples the kernels' full support
    # (matching full-range tomography). On this synthetic cloud the shells are
    # fat relative to the domain, so the deliberate convention gap reaches a
    # few percent; on well-clipped trained assets it is marginal.
    TOL_TRANSMITTANCE, TOL_TRANSPORT = 0.005, 0.06

    sensor = mi.load_dict({
        "type": "perspective", "fov": 45,
        "to_world": T().look_at(origin=[0, 0.4, 2.2], target=[0, 0, 0], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 128, "height": 128,
                 "rfilter": {"type": "box"}}})

    def render_mean(integ, spp):
        scene = mi.load_dict({"type": "scene", "integrator": integ,
                              "prims": _prb_prim_cloud(mi, gabor=False),
                              "env": {"type": "constant"}})
        img = np.array(mi.render(scene, sensor=sensor, spp=spp, seed=0))
        assert np.all(np.isfinite(img)), f"non-finite pixels ({integ['type']})"
        return float(img.mean())

    # Full-range tomography integrates each primitive over its full support --
    # exactly the stochastic integrator's unbounded free-flight convention.
    tomo = render_mean({"type": "gfields_tomography", "max_depth": 512,
                        "kernel_full_range": True, **KERNEL}, spp=16)
    st_t = render_mean({"type": "gfields_prb_stochastic", "max_depth": 1,
                        "use_nee": False, **BASE}, spp=1024)
    rel = abs(st_t - tomo) / max(abs(tomo), 1e-9)
    assert rel < TOL_TRANSMITTANCE, f"escape transmittance rel {rel:.4f}"

    prb = render_mean({"type": "gfields_prb", "max_depth": 8,
                       "solver_type": "bisection", "solver_max_iterations": 4,
                       **BASE}, spp=512)
    st = render_mean({"type": "gfields_prb_stochastic", "max_depth": 8,
                      **BASE}, spp=512)
    rel = abs(st - prb) / max(abs(prb), 1e-9)
    assert rel < TOL_TRANSPORT, f"transport vs gfields_prb rel {rel:.4f}"


@needs_cuda
def test_prb_stochastic_truncate_matches_prb():
    """truncate_at_shell=True draws each free flight from the density restricted
    to the extent shell -- the same estimand gfields_prb integrates -- so it must
    match gfields_prb far more tightly than the unbounded default (which differs
    by the clipped-tail mass) and land within a tight transport tolerance."""
    mi = _setup_mi()
    import gfields  # noqa: F401
    T = mi.ScalarTransform4f

    BASE = {"kernel_type": "gaussian", "kernel_normalized": True,
            "max_overlaps": 64, "rr_depth_primitive": 1024,
            "max_depth_primitive": 2048}
    sensor = mi.load_dict({
        "type": "perspective", "fov": 45,
        "to_world": T().look_at(origin=[0, 0.4, 2.2], target=[0, 0, 0], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 128, "height": 128,
                 "rfilter": {"type": "box"}}})

    def render_mean(integ, spp):
        scene = mi.load_dict({"type": "scene", "integrator": integ,
                              "prims": _prb_prim_cloud(mi, gabor=False),
                              "env": {"type": "constant"}})
        img = np.array(mi.render(scene, sensor=sensor, spp=spp, seed=0))
        assert np.all(np.isfinite(img)), f"non-finite pixels ({integ['type']})"
        return float(img.mean())

    prb = render_mean({"type": "gfields_prb", "max_depth": 8,
                       "solver_type": "bisection", "solver_max_iterations": 4,
                       **BASE}, spp=512)
    st_unb = render_mean({"type": "gfields_prb_stochastic", "max_depth": 8,
                          **BASE}, spp=512)
    st_trunc = render_mean({"type": "gfields_prb_stochastic", "max_depth": 8,
                            "truncate_at_shell": True, **BASE}, spp=512)

    rel_unb = abs(st_unb - prb) / max(abs(prb), 1e-9)
    rel_trunc = abs(st_trunc - prb) / max(abs(prb), 1e-9)
    assert rel_trunc < 0.02, f"truncated vs gfields_prb rel {rel_trunc:.4f}"
    assert rel_trunc < rel_unb, (
        f"truncation should tighten the gap: trunc {rel_trunc:.4f} "
        f"!< unbounded {rel_unb:.4f}")


@needs_cuda
def test_nee_truncation_option():
    """NEE origin-overlap truncation is optional (nee_truncate_at_shell), for
    both integrators: (a) prb_stochastic's flag defaults to `truncate_at_shell`;
    gfields_prb defaults unbounded; (b) the any-hit NEE fast path reproduces the
    reference walk in BOTH modes (toggling use_anyhit must not change results)."""
    mi = _setup_mi()
    import gfields  # noqa: F401
    T = mi.ScalarTransform4f

    # (a) default coupling
    st = mi.load_dict({"type": "gfields_prb_stochastic", "kernel_type": "gaussian"})
    st_tr = mi.load_dict({"type": "gfields_prb_stochastic", "kernel_type": "gaussian",
                          "truncate_at_shell": True})
    prb = mi.load_dict({"type": "gfields_prb", "kernel_type": "gaussian"})
    assert st.nee_truncate_at_shell is False
    assert st_tr.nee_truncate_at_shell is True          # follows truncate_at_shell
    assert prb.nee_truncate_at_shell is False

    # (b) any-hit == walk in each NEE mode, on an area-lit cloud (NEE carries it)
    sensor = mi.load_dict({
        "type": "perspective", "fov": 45,
        "to_world": T().look_at(origin=[0, 0.4, 2.2], target=[0, 0, 0], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 96, "height": 96, "rfilter": {"type": "box"}}})
    scene = mi.load_dict({
        "type": "scene", "prims": _prb_prim_cloud(mi, gabor=False),
        "light": {"type": "rectangle",
                  "to_world": T().look_at(origin=[1.5, 1.5, 1.0], target=[0, 0, 0],
                                          up=[0, 1, 0]).scale(0.5),
                  "emitter": {"type": "area", "radiance": {"type": "rgb", "value": 30.0}}},
        "env": {"type": "constant", "radiance": 0.05}})

    def mean(nee_trunc, use_anyhit):
        integ = mi.load_dict({"type": "gfields_prb_stochastic", "max_depth": 8,
                              "kernel_type": "gaussian", "use_nee": True,
                              "nee_truncate_at_shell": nee_trunc,
                              "use_anyhit": use_anyhit})
        img = np.array(mi.render(scene, integrator=integ, sensor=sensor, spp=256, seed=0))
        assert np.all(np.isfinite(img))
        return float(img.mean())

    for nee_trunc in (False, True):
        a = mean(nee_trunc, True)
        w = mean(nee_trunc, False)
        rel = abs(a - w) / max(abs(w), 1e-9)
        assert rel < 0.01, f"anyhit != walk (nee_truncate={nee_trunc}): rel {rel:.4f}"


@needs_cuda
def test_prb_stochastic_rejects_gabor():
    """gfields_prb_stochastic must refuse non-Gaussian kernels."""
    mi = _setup_mi()
    import gfields  # noqa: F401
    with pytest.raises(Exception):
        mi.load_dict({"type": "gfields_prb_stochastic",
                      "kernel_type": "gabor_gaussian_mixture"})


# ---------------------------------------------------------------------------
# 6. OptiX any-hit transmittance query (Scene.ray_transmittance_ellipsoids)
# ---------------------------------------------------------------------------

@needs_cuda
def test_anyhit_transmittance_query():
    """The any-hit transmittance query (docs/prb_stochastic.md section 5):
    the deterministic mode must reproduce a same-ray sequential hit walk's
    signed optical depth; the stochastic mode's mean must converge to the
    same transmittance; a non-ellipsoid blocker must return +inf depth."""
    mi = _setup_mi()
    import drjit as dr
    import gfields  # noqa: F401
    from gfields.integrators.common import Ellipsoid
    T = mi.ScalarTransform4f

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "gfields_tomography"},
        "prims": _prb_prim_cloud(mi, n=120, gabor=False, seed=9),
        "env": {"type": "constant"}})

    res = 128
    xs = (np.arange(res, dtype=np.float32) + 0.5) / res * 1.6 - 0.8
    px, py = np.meshgrid(xs, -xs)
    ray = mi.Ray3f(mi.Point3f(px.ravel(), py.ravel(),
                              np.full(res * res, 3.0, np.float32)),
                   mi.Vector3f(0, 0, -1))
    ray.mask = mi.UInt32(255)

    # Same-ray sequential walk reference (full-line signed tau per hit)
    r = mi.Ray3f(ray)
    tau_ref = mi.Float(0.0)
    for _ in range(400):
        si = scene.ray_intersect(r, coherent=False,
                                 ray_flags=mi.RayFlags.All, active=mi.Bool(True))
        hit = si.is_valid()
        if not bool(dr.any(hit)):
            break
        ell = Ellipsoid.gather(si.shape, si.prim_index, hit)
        w = ell.rot.T * r.d / ell.scale
        p = ell.rot.T * (r.o - ell.center) / ell.scale
        J = dr.rcp(dr.norm(w))
        wn = w * J
        B, Cq = dr.dot(wn, p), dr.dot(p, p)
        om, D = ell.beta * dr.sum(wn), ell.beta * dr.sum(p)
        t_i = ell.opacity * J * dr.exp(-0.5 * (Cq - B * B + om * om)) \
            * dr.cos(D - om * B) * dr.rcp(2.0 * dr.pi * dr.prod(ell.scale))
        tau_ref[hit] = tau_ref + t_i
        r.o = dr.select(hit, si.p + r.d * 5e-6, r.o)
        r.maxt = dr.select(hit, r.maxt, mi.Float(0.0))
    tau_ref = np.array(tau_ref).astype(np.float64)

    # Deterministic accumulation
    tau = np.array(scene.ray_transmittance_ellipsoids(
        ray, mi.UInt32(0),
        int(mi.RayFlags.TransmittanceEllipsoids))).astype(np.float64)
    T_ref, T_any = np.exp(-tau_ref), np.exp(-tau)
    mean_rel = abs(T_any.mean() - T_ref.mean()) / max(T_ref.mean(), 1e-9)
    frac_off = float((np.abs(T_any - T_ref) > 1e-3).mean())
    assert mean_rel < 2e-3, f"deterministic any-hit mean rel {mean_rel:.2e}"
    # isolated pixels may differ where the walk's epsilon respawn skips
    # near-coincident entries (the any-hit result is the more complete one)
    assert frac_off < 0.01, f"deterministic any-hit pixels off {frac_off:.4f}"

    # Stochastic Bernoulli mode converges to the same transmittance
    rng = np.random.default_rng(3)
    acc = np.zeros(res * res)
    K = 64
    for _ in range(K):
        acc += np.array(scene.ray_transmittance_ellipsoids(
            ray, mi.UInt32(rng.integers(0, 2**31, res * res, dtype=np.uint32)),
            int(mi.RayFlags.StochasticEllipsoids)))
    st_rel = abs(acc.mean() / K - T_ref.mean()) / max(T_ref.mean(), 1e-9)
    assert st_rel < 0.02, f"stochastic any-hit mean rel {st_rel:.2e}"

    # A blocking surface yields infinite depth
    scene_b = mi.load_dict({
        "type": "scene", "integrator": {"type": "gfields_tomography"},
        "prims": _prb_prim_cloud(mi, n=8, gabor=False, seed=9),
        "wall": {"type": "rectangle",
                 "to_world": T().translate([0, 0, 0]) @ T().scale([2, 2, 1])},
        "env": {"type": "constant"}})
    tau_b = np.array(scene_b.ray_transmittance_ellipsoids(
        ray, mi.UInt32(0), int(mi.RayFlags.TransmittanceEllipsoids)))
    assert np.isinf(tau_b).all(), "blocker must yield +inf optical depth"


# ---------------------------------------------------------------------------
# 7. OptiX any-hit distance sampling / coverage collection
#    (Scene.ray_sample_ellipsoids / ray_collect_ellipsoids)
# ---------------------------------------------------------------------------

@needs_cuda
def test_anyhit_distance_sampling():
    """The any-hit distance-sampling query must reproduce the analytic
    min-of-free-flights law P(t > s) = prod_i exp(-tau_i([0, s])) over the
    unbounded kernels (extent shells only act as BVH discovery proxies;
    primitives covering the ray origin are visible via interior-hit
    reporting), clip flights at a committed blocking surface, and the
    collect query must return exactly the primitives containing a probe
    point. Finally, gfields_prb_stochastic with the any-hit fast paths must
    match its own hit-walk estimator."""
    mi = _setup_mi()
    import drjit as dr
    import gfields  # noqa: F401
    from math import erf
    T = mi.ScalarTransform4f

    n = 60
    prims = _prb_prim_cloud(mi, n=n, gabor=False, seed=5)
    centers = np.array(prims["centers"]).reshape(n, 3).astype(np.float64)
    scales = np.array(prims["scales"]).reshape(n, 3).astype(np.float64)
    sigma = np.array(prims["opacities"]).reshape(n).astype(np.float64)
    EXT = 3.0
    plane_z = -1.5

    scene = mi.load_dict({
        "type": "scene", "integrator": {"type": "gfields_tomography"},
        "prims": prims,
        "wall": {"type": "rectangle",
                 "to_world": T().translate([0, 0, plane_z]) @ T().scale(8.0)},
        "env": {"type": "constant"}})

    # --- collect: covering set at probe points == brute-force containment
    K = 32
    probes = np.vstack([centers[:6], [[5.0, 5.0, 5.0]]]).astype(np.float32)
    w = len(probes)
    buf = dr.zeros(mi.UInt32, 2 * K * w)
    dr.eval(buf)
    slot_addr = (mi.UInt64(mi.device_address(buf))
                 + mi.UInt64(dr.arange(mi.UInt32, w)) * mi.UInt64(2 * K * 4))
    cray = dr.zeros(mi.Ray3f, w)
    cray.o = mi.Point3f(probes[:, 0], probes[:, 1], probes[:, 2])
    cray.d = mi.Vector3f(0, 0, -1)
    cray.maxt = mi.Float(10.0)
    cray.mask = mi.UInt32(255)
    cnt = np.array(scene.ray_collect_ellipsoids(cray, slot_addr, K,
                                                mi.Bool(True)))
    pairs = np.array(buf).reshape(w, K, 2)
    for i, x in enumerate(probes.astype(np.float64)):
        got = set(pairs[i, :cnt[i], 1].tolist())
        u = (x[None, :] - centers) / (scales * EXT)
        ref = set(np.nonzero((u ** 2).sum(1) <= 1.0)[0].tolist())
        assert got == ref, f"collect probe {i}: {sorted(got)} != {sorted(ref)}"

    # --- sample: empirical survival matches the analytic truncated law,
    #     from a ray origin inside the cloud (covering primitives exercised)
    o = np.array([0.05, -0.02, 0.9])
    d = np.array([0.0, 0.0, -1.0])

    def survival_ref(t):
        Tr = 1.0
        for c, s, sg in zip(centers, scales, sigma):
            ww = d / s
            p = (o - c) / s
            J = 1.0 / np.linalg.norm(ww)
            ww = ww * J
            B, C = float(ww @ p), float(p @ p)
            # Discovery proxy: the traversal only sees primitives whose
            # extent shell the ray intersects (the mass of the skipped
            # off-shell primitives is negligible by construction).
            if B * B - C + EXT ** 2 <= 0.0:
                continue
            # Unbounded flight over [0, t]: no shell clipping.
            t1 = t / J
            if t1 <= 0.0:
                continue
            A = sg * J * np.exp(-0.5 * (C - B * B)) / (2 * np.pi * np.prod(s))
            Tr *= np.exp(-0.5 * A * (erf((t1 + B) / np.sqrt(2))
                                     - erf(B / np.sqrt(2))))
        return Tr

    ns = 200000
    ray = dr.zeros(mi.Ray3f, ns)
    ray.o = mi.Point3f(*[float(v) for v in o])
    ray.d = mi.Vector3f(*[float(v) for v in d])
    ray.maxt = mi.Float(dr.inf)
    ray.mask = mi.UInt32(255)
    rng = np.random.default_rng(11)
    seeds = mi.UInt32(rng.integers(0, 2**31, ns, dtype=np.uint32))
    t_s, _wsh, _wpr, pi = scene.ray_sample_ellipsoids(
        ray, seeds, mi.Float(0.0), mi.Bool(True))
    ts = np.array(t_s)
    for t_chk in (0.2, 0.6, 1.2, 2.0):
        emp = float((ts > t_chk).mean())
        ref = survival_ref(t_chk)
        assert abs(emp - ref) < 6e-3, \
            f"sample survival at t={t_chk}: emp {emp:.4f} vs ref {ref:.4f}"
    plane_t = o[2] - plane_z
    assert bool((ts[np.isfinite(ts)] < plane_t + 1e-3).all()), \
        "flights must clip at the committed blocking surface"
    pi_t = np.array(pi.t)
    assert bool(np.isfinite(pi_t).all()
                and (np.abs(pi_t - plane_t) < 1e-3).all()), \
        "the committed preliminary intersection must be the blocker"

    # --- full transport: any-hit fast paths vs the hit walk (same estimand).
    # gfields_prb_stochastic is Gaussian-only: the any-hit distance-sampling
    # and NEE fast paths must match its own hit-walk estimator.
    sensor = mi.load_dict({
        "type": "perspective", "fov": 45,
        "to_world": T().look_at(origin=[0, 0.4, 2.2], target=[0, 0, 0],
                                up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 96, "height": 96,
                 "rfilter": {"type": "box"}}})
    scene_r = mi.load_dict({
        "type": "scene",
        "prims": _prb_prim_cloud(mi, n=120, gabor=False, seed=7),
        "env": {"type": "constant"}})
    means = {}
    for ah in (False, True):
        integ = mi.load_dict({
            "type": "gfields_prb_stochastic", "max_depth": 6,
            "kernel_type": "gaussian",
            "max_overlaps": 64, "use_anyhit": ah,
            "use_anyhit_sampling": ah})
        img = np.array(mi.render(scene_r, integrator=integ,
                                 sensor=sensor, spp=256, seed=3))
        assert np.all(np.isfinite(img))
        means[ah] = float(img.mean())
    rel = abs(means[True] - means[False]) / max(means[False], 1e-9)
    assert rel < 0.01, f"any-hit vs walk transport rel {rel:.4f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
