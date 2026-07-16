# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fast absorption-only reference renderer for tomography optimization.

Generates the ground-truth transmittance images the Gabor-field tomography
optimiser regresses against. A small, self-contained residual-ratio tracker (Novak et al. 2014)
over supervoxel local majorants (Szirmay-Kalos et al. 2011) on top of Mitsuba 3.

integrator = { 'type': 'gfields_reference',
                'sigma_t': { 'type': 'gridvolume', 'data': ..., 'to_world': ... },
                'supervoxel_factor': 8, 'background': 1.0 }

`render_absorption_references(grid, to_world, sensors, ...)` wraps this into a
one-call helper that returns a list of `mi.TensorXf` images.

Just adding this to avoid using the distance samplers to render absorption-only references,
which is very slow.
"""
from __future__ import annotations

import numpy as np
import drjit as dr
import mitsuba as mi


# ---------------------------------------------------------------------------
# Supervoxel majorant / control grids
# ---------------------------------------------------------------------------

def _block_reduce(scalar: np.ndarray, factor: int, op: str) -> np.ndarray:
    """Block-reduce a (Z,Y,X) array by `factor` per axis (edge-padded)."""
    if factor <= 1:
        return np.ascontiguousarray(scalar, dtype=np.float32)
    pad = [(-s) % factor for s in scalar.shape]
    if any(pad):
        scalar = np.pad(scalar, [(0, p) for p in pad], mode="edge")
    z, y, x = scalar.shape
    r = scalar.reshape(z // factor, factor, y // factor, factor, x // factor, factor)
    out = r.max(axis=(1, 3, 5)) if op == "max" else r.mean(axis=(1, 3, 5))
    return np.ascontiguousarray(out, dtype=np.float32)


def build_supervoxel_grids(sigma_t: np.ndarray, factor: int, majorant_factor: float = 1.01):
    """Return `(majorant, control)` coarse (Z,Y,X,1) grids from a σ_t volume."""
    data = np.asarray(sigma_t, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0] if data.shape[-1] == 1 else data.max(axis=-1)
    maj = _block_reduce(data, factor, "max") * np.float32(majorant_factor)
    ctrl = _block_reduce(data, factor, "mean")
    return maj[..., None], ctrl[..., None]


# ---------------------------------------------------------------------------
# DDA traversal over a regular grid
# ---------------------------------------------------------------------------

def _prepare_dda(bbox_min, bbox_max, resolution, ray, tmin, tmax):
    bmin, bmax = mi.Vector3f(bbox_min), mi.Vector3f(bbox_max)
    extents = bmax - bmin
    local_o = (ray.o - bmin) / extents
    local_d = ray.d / extents
    res = mi.Vector3f(float(resolution[0]), float(resolution[1]), float(resolution[2]))
    voxel = dr.rcp(res)
    current = dr.clip(dr.floor((local_o + tmin * local_d) / voxel), 0.0, res - 1.0)
    last = dr.clip(dr.floor((local_o + tmax * local_d) / voxel), 0.0, res - 1.0)
    step = dr.select(local_d >= 0.0, mi.Vector3f(1.0), mi.Vector3f(-1.0))
    nb = (current + step) * voxel
    nb = nb + dr.select((current != last) & (local_d < 0.0), voxel, mi.Vector3f(0.0))
    nonzero = local_d != 0.0
    safe_d = dr.select(nonzero, local_d, mi.Vector3f(1.0))
    inf = mi.Vector3f(dr.inf)
    dda_tmax = dr.select(nonzero, (nb - local_o) / safe_d, inf)
    dda_tdelta = dr.select(nonzero, step * voxel / safe_d, inf)
    return mi.Float(tmin), dda_tmax, dda_tdelta


def _ray_box(ray, bmin, bmax, active):
    """Slab ray/AABB intersection -> (hit, tmin, tmax)."""
    inv = dr.rcp(ray.d)
    t0 = (mi.Vector3f(bmin) - ray.o) * inv
    t1 = (mi.Vector3f(bmax) - ray.o) * inv
    tmin = dr.max(dr.minimum(t0, t1))
    tmax = dr.min(dr.maximum(t0, t1))
    tmin = dr.maximum(tmin, 0.0)
    return active & (tmax > tmin) & (tmax > 0.0), tmin, tmax


def _eval1(vol, p, active):
    it = dr.zeros(mi.Interaction3f)
    it.p = p
    return vol.eval_1(it, active)


# ---------------------------------------------------------------------------
# Residual-ratio-tracking transmittance
# ---------------------------------------------------------------------------

@dr.syntax
def _rrt_transmittance(ray, sigma_vol, maj_vol, ctrl_vol, bmin, bmax, res,
                       sampler, tmin, tmax, active):
    """Unbiased transmittance over [tmin, tmax] via residual ratio tracking
    with supervoxel local majorants (advol.tr_rrt_local, self-contained)."""
    dda_t, dda_tmax, dda_tdelta = _prepare_dda(bmin, bmax, res, ray, tmin, tmax)
    Tr = mi.Float(1.0)
    Tc = mi.Float(1.0)
    desired_tau = -dr.log(1.0 - sampler.next_1d(active))
    tau_acc = mi.Float(0.0)
    keep = mi.Bool(active)
    redraw_next = mi.Bool(False)

    while dr.hint(keep, label="gfields_reference.rrt"):
        redraw = keep & redraw_next
        desired_tau = dr.select(redraw, -dr.log(1.0 - sampler.next_1d(redraw)), desired_tau)
        tau_acc = dr.select(redraw, mi.Float(0.0), tau_acc)
        redraw_next = mi.Bool(False)

        t_next = dr.min(dda_tmax)
        hx = dda_tmax.x <= t_next
        hy = (dda_tmax.y <= t_next) & ~hx
        hz = (dda_tmax.z <= t_next) & ~hx & ~hy
        step = mi.Vector3f(dr.select(hx, dda_tdelta.x, 0.0),
                           dr.select(hy, dda_tdelta.y, 0.0),
                           dr.select(hz, dda_tdelta.z, 0.0))

        mid = ray(0.5 * (dda_t + t_next))
        maj = _eval1(maj_vol, mid, keep)
        ctrl = _eval1(ctrl_vol, mid, keep)
        residual = dr.maximum(maj - ctrl, 1e-8)
        tau_next = tau_acc + residual * (t_next - dda_t)

        t_precise = dda_t + (desired_tau - tau_acc) / residual
        reached = keep & (tau_next >= desired_tau) & (t_precise < tmax) & (residual > 0.0)
        escaped = keep & (t_next >= tmax) & ~reached
        dist = dr.select(reached, t_precise - dda_t,
                         dr.select(escaped, tmax - dda_t, t_next - dda_t))

        sigma = _eval1(sigma_vol, ray(t_precise), reached)
        delta = dr.select(reached & (residual > 0.0), (sigma - ctrl) / residual, 0.0)
        Tr = dr.select(reached, Tr * (1.0 - delta), Tr)
        Tc = Tc * dr.exp(-ctrl * dist)

        dda_t = dr.select(reached, t_precise, dr.select(escaped, mi.Float(dr.inf), t_next))
        dda_tmax = dr.select(reached, dda_tmax, dda_tmax + step)
        tau_acc = dr.select(reached, tau_acc, tau_next)
        redraw_next = reached
        keep &= ~escaped

    return dr.clip(Tr * Tc, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

class GfieldsReferenceIntegrator(mi.SamplingIntegrator):
    """Absorption-only transmittance reference (primal); L = T * background."""

    def __init__(self, props):
        super().__init__(props)
        self.background = mi.Spectrum(float(props.get("background", 1.0)))
        factor = int(props.get("supervoxel_factor", 8))
        self.sigma_t = props.get("sigma_t")
        if self.sigma_t is None:
            raise Exception("gfields_reference requires a 'sigma_t' volume property")

        params = mi.traverse(self.sigma_t)
        grid = np.array(params["data"])
        to_world = props.get("to_world", mi.ScalarTransform4f()) if props.has_property("to_world") \
            else mi.ScalarTransform4f()
        maj, ctrl = build_supervoxel_grids(grid, factor)
        vd = lambda a: {"type": "gridvolume", "data": mi.TensorXf(a),
                        "to_world": to_world, "filter_type": "nearest"}
        self.maj_vol = mi.load_dict(vd(maj))
        self.ctrl_vol = mi.load_dict(vd(ctrl))
        # World-space AABB of the unit-cube gridvolume.
        c = [to_world @ mi.ScalarPoint3f(x, y, z)
             for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        self.bmin = mi.ScalarPoint3f(*(min(p[i] for p in c) for i in range(3)))
        self.bmax = mi.ScalarPoint3f(*(max(p[i] for p in c) for i in range(3)))
        self.res = (maj.shape[2], maj.shape[1], maj.shape[0])  # (X,Y,Z)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        active = mi.Bool(active)
        hit, tmin, tmax = _ray_box(ray, self.bmin, self.bmax, active)
        T = dr.select(hit,
                      _rrt_transmittance(ray, self.sigma_t, self.maj_vol, self.ctrl_vol,
                                         self.bmin, self.bmax, self.res, sampler, tmin, tmax, hit),
                      mi.Float(1.0))
        return self.background * T, active, []

    def to_string(self):
        return "GfieldsReferenceIntegrator[]"


mi.register_integrator("gfields_reference", lambda props: GfieldsReferenceIntegrator(props))


# ---------------------------------------------------------------------------
# One-call helper
# ---------------------------------------------------------------------------

def render_absorption_references(grid, to_world, sensors, *, spp=128, background=1.0,
                                 supervoxel_factor=8, seed=0):
    """Render absorption-only transmittance references for a σ_t voxel grid.

    `grid` is a (Z,Y,X[,C]) numpy array; `to_world` maps the unit cube to world;
    `sensors` is a list of `mi.Sensor`. Returns a list of `mi.TensorXf` images.
    """
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "gfields_reference",
                       "supervoxel_factor": supervoxel_factor, "background": background,
                       "to_world": to_world,
                       "sigma_t": {"type": "gridvolume", "data": mi.TensorXf(grid),
                                   "to_world": to_world}},
    })
    return [mi.TensorXf(mi.render(scene, sensor=s, spp=spp, seed=seed)) for s in sensors]


__all__ = ["GfieldsReferenceIntegrator", "render_absorption_references",
           "build_supervoxel_grids"]
