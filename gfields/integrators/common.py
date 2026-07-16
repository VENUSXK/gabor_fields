# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Implements kernel functions for volumetric primitives in Mitsuba scenes,
including Gaussian, Epanechnikov and Gabor kernels.
'''

from __future__ import annotations # Delayed parsing of type annotations
from dataclasses import dataclass

import drjit as dr
import mitsuba as mi

from volprim.integrators.stack import alloc_stack

#-------------------------------------------------------------------------------

def get_ellipsoids_shape(scene, requested=True):
    for i, shape in enumerate(scene.shapes()):
        if shape.is_ellipsoids():
            if i > 0:
                mi.Log(mi.LogLevel.Warn, 'For performance consideration, it is '
                                         'better to define the ellipsoids shape '
                                         'as the first shape in the scene!')
            return shape, mi.ShapePtr(scene.shapes_dr()[i])
    if requested:
        raise Exception("Couldn't find ellipsoids shape in the scene!")
    else:
        return None, mi.ShapePtr(None)

def check_ellipsoids_attribute(scene, attributes):
    for i, shape in enumerate(scene.shapes()):
        if shape.is_ellipsoids():
            for attr in attributes:
                assert shape.has_attribute(attr), f'Requested ellipsoid attribute \'{attr}\' not found on shape \'{shape.id()}\''

def has_ellipsoids_shapes(scene):
    for i, shape in enumerate(scene.shapes()):
        if shape.is_ellipsoids():
            return True
    return False


@dataclass
class Ellipsoid:
    center:  mi.Point3f = mi.Point3f(0)
    scale:   mi.Vector3f = mi.Vector3f(0)
    quat:    mi.Quaternion4f = mi.Quaternion4f(0)
    rot:     mi.Matrix3f = mi.Matrix3f(0)
    extent:  mi.Float = mi.Float(3.0)
    beta:    mi.Float = mi.Float(0) #omega
    opacity: mi.Float = mi.Float(0)

    @staticmethod
    def ravel(center, scale, quat):
        data_width = 10
        data = dr.empty(mi.Float, dr.width(center) * data_width)
        idx = dr.arange(mi.UInt32, dr.width(center))
        for i in range(3):
            dr.scatter(data, center[i], idx * data_width + i)
        for i in range(3):
            dr.scatter(data, scale[i], idx * data_width + 3 + i)
        for i in range(4):
            dr.scatter(data, quat[i], idx * data_width + 6 + i)
        return data

    @staticmethod
    def unravel(data, beta=None):
        data_width = 10
        idx = dr.arange(mi.UInt32, dr.width(data) // data_width)
        center = mi.Point3f([dr.gather(mi.Float, data, idx * data_width + 0 + i) for i in range(3)])
        scale  = mi.Vector3f([dr.gather(mi.Float, data, idx * data_width + 3 + i) for i in range(3)])
        quat   = mi.Quaternion4f([dr.gather(mi.Float, data, idx * data_width + 6 + i) for i in range(4)])
        rot    = dr.quat_to_matrix(mi.Matrix3f, quat)
        return Ellipsoid(center, scale, quat, rot, None, beta, None)

    @staticmethod
    def gather(shape, prim_index, active):
        def func(self, prim_index, active):
            if self is not None and self.is_ellipsoids():
                si = dr.zeros(mi.SurfaceInteraction3f)
                si.prim_index = prim_index
                data = self.eval_attribute_x("ellipsoid", si, active)
                center = mi.Point3f([data[i] for i in range(3)])
                scale  = mi.Vector3f([data[i + 3] for i in range(3)])
                quat   = mi.Quaternion4f([data[i + 6] for i in range(4)])
                rot    = dr.quat_to_matrix(mi.Matrix3f, quat)
                extent = self.eval_attribute_1("extent", si, active)
                beta   = self.eval_attribute_1("omega", si, active)
                opacity = self.eval_attribute_1("opacities", si, active)
                return center, scale, quat, rot, extent, beta, opacity
            else:
                return mi.Point3f(0), mi.Vector3f(0), mi.Quaternion4f(0), mi.Matrix3f(0), mi.Float(0), mi.Float(0), mi.Float(0)
        return Ellipsoid(*dr.dispatch(shape, func, prim_index, active))
    
#-------------------------------------------------------------------------------

class Kernel:
    @staticmethod
    def factory(props: mi.Properties):
        name = props.get('kernel_type', 'gaussian')

        if name == 'gaussian':
            return GaussianKernel(props)
        elif name == 'epanechnikov':
            return EpanechnikovKernel(props)
        elif name in ('gabor', 'gabor_gaussian_mixture'):
            return GaborKernel(props)
        else:
            raise Exception('Unknown kernel type! Should be one of "gaussian", "epanechnikov", "gabor", or "gabor_gaussian_mixture".')

    def __init__(self, props):
        # Whether to normalize the density integral (i.e., kernel_normalized=True means that peak value is 1.0)
        self.normalized = props.get('kernel_normalized', False)

        # Whether we should always integrate the full density kernel (used in `gaussian_sp` integrator)
        self.full_range = props.get('kernel_full_range', False)


    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        raise Exception('Not implemented!')
    
                                      

    def normalization_factor(self, ellipsoid: Ellipsoid) -> mi.Float:
        raise Exception('Not implemented!')

#-------------------------------------------------------------------------------

class GaussianKernel(Kernel):
    def __init__(self, props):
        super().__init__(props)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s
        return dr.exp(-0.5 * dr.sum(p * p))

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s

        denom = dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(s)
        pdf = dr.exp(-0.5 * dr.sum(p * p)) / denom
        return dr.select(active, pdf, 0.0)

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        s = ellipsoid.scale
        w = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        w_norm = dr.norm(w)
        w /= w_norm

        B = dr.dot(w, p)
        C = dr.dot(p, p)

        # Same normalization as density_integral, so that the solved t
        # satisfies sigmat * density_integral(ray, 0, t) == chi
        K = sigmat * dr.rcp(w_norm) * dr.exp(-0.5 * (C - B * B)) * dr.rcp(2.0 * dr.pi * dr.prod(s))
        erfinv_arg = dr.erf(B / dr.sqrt(2.0)) + 2.0 * chi / K
        t_s = dr.sqrt(2.0) * dr.erfinv(erfinv_arg) - B
        t_s *= dr.rcp(w_norm)  # whitened arc length -> world-space distance
        t_s = dr.select(dr.isnan(t_s), dr.inf, t_s)  # Handle NaNs arising from intersection points in the infinite (dr.abs(erfinv_arg)>1)
        # TODO: would be interesting that erfinv returns -inf or inf instead of nans, could we change this behavior

        return dr.select(active, t_s, 0.0)

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        s = ellipsoid.scale
        w = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        w_norm = dr.rcp(dr.norm(w))
        w *= w_norm

        if self.full_range or (tmin is None and tmax is None):
            B = dr.dot(w, p)
            C = dr.dot(p, p)
            density = w_norm * dr.exp(-0.5 * dr.fma(-B, B, C))
        else:
            active = mi.Bool(active) & (tmin < tmax) & (tmax > 0.0)  # skip degenerate segments
            p = p + (tmin / w_norm) * w
            t_limit = dr.clip((tmax - tmin) / w_norm, 0.0, ellipsoid.extent * 2.0)

            B = dr.dot(w, p)
            C = dr.dot(p, p)
            sqrt_half = dr.sqrt(0.5)
            seg = 0.5 * (dr.erf(sqrt_half * (t_limit + B)) - dr.erf(sqrt_half * B))
            density = w_norm * dr.exp(-0.5 * dr.fma(-B, B, C)) * seg

        density *= dr.rcp(2.0 * dr.pi * dr.prod(s))
        density[~active] = 0.0
        return density

    def normalization_factor(self, s: mi.Vector3f) -> mi.Float:
        return dr.rcp(dr.sqrt(8.0 * (dr.pi * dr.pi * dr.pi)) * dr.prod(s))
#-------------------------------------------------------------------------------

class EpanechnikovKernel(Kernel):
    def __init__(self, props):
        super().__init__(props)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        
        s = ellipsoid.scale
        p = ellipsoid.rot.T * (p - ellipsoid.center) / s

        dist = dr.norm(p)
        value = (1.0 - (1.0/7.0) * dist**2)
        return dr.maximum(value, 0.0)

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:

        s = ellipsoid.scale
        u = ellipsoid.rot.T * (p - ellipsoid.center) / s

        shape = dr.maximum(1.0 - (1.0 / 7.0) * dr.squared_norm(u), 0.0)
        density = self.normalization_factor(ellipsoid) * shape
        return dr.select(active, density, 0.0)

    def inv_cdf(self,
                ray: mi.Ray3f,
                ellipsoid: Ellipsoid,
                sigmat: mi.Float,
                chi: mi.Float,
                active: mi.Bool) -> mi.Float:
        raise Exception('EpanechnikovKernel.inv_cdf(): Not implemented!')

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:
        s = ellipsoid.scale
        v = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        v_norm = dr.norm(v)
        w = v * dr.rcp(v_norm) 

        B = dr.dot(w, p)
        C = dr.dot(p, p)
        disc = dr.fma(B, B, 7.0 - C) 
        active = mi.Bool(active) & (disc > 0.0)
        sq = dr.sqrt(dr.maximum(disc, 0.0))
        tau_in, tau_out = -B - sq, -B + sq 

        if self.full_range or (tmin is None and tmax is None):
            a, b = tau_in, tau_out                   # whole effective support
        else:
            active = mi.Bool(active) & (tmin < tmax) & (tmax > 0.0)  # skip degenerate segments
            a = dr.maximum(tmin * v_norm, tau_in)
            b = dr.minimum(tmax * v_norm, tau_out)
        active = mi.Bool(active) & (a < b)

        G = lambda t: t - dr.fma(C, t, dr.fma(B, t * t, t * t * t * (1.0 / 3.0))) * (1.0 / 7.0)
        density = (G(b) - G(a)) * dr.rcp(v_norm)

        if self.normalized:
            density *= self.normalization_factor(ellipsoid)

        density = dr.maximum(density, 0.0)
        density[~active] = 0.0
        density[~dr.isfinite(density)] = 0.0

        return density

    def normalization_factor(self, ellipsoid: Ellipsoid) -> mi.Float:
        s = ellipsoid.scale
        return 15.0 / (56.0 * dr.sqrt(7.0) * dr.pi * dr.prod(s))

class GaborKernel(Kernel):
    """
    Gabor kernel: a frequency-modulated Gaussian. A plain Gaussian is simply
    the omega = 0 special case -- the cosine term becomes cos(0) = 1 and the
    segment integral's complex erf reduces to the real erf -- so Gaussians and
    Gabors are rendered by the *same* code path with no per-primitive branching.
    The per-primitive frequency is `ellipsoid.beta` (the `omega` attribute);
    Gaussians must carry omega = 0.
    """
    def __init__(self, props):
        super().__init__(props)

    def eval(self,
             p: mi.Point3f,
             ellipsoid: Ellipsoid,
             active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s
        β = ellipsoid.beta

        value = dr.exp(-0.5 * dr.sum(p * p)) * dr.cos(β * dr.sum(p))

        return dr.select(active, value, 0.0)

    def pdf(self,
            p: mi.Point3f,
            ellipsoid: Ellipsoid,
            active: mi.Bool) -> mi.Float:
        p = ellipsoid.rot.T * (p - ellipsoid.center)
        s = ellipsoid.scale
        p = p / s
        β = ellipsoid.beta

        denom = dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(s)
        pdf = dr.exp(-0.5 * dr.sum(p * p)) / denom * dr.cos(β * dr.sum(p))

        return dr.select(active, pdf, 0.0)

    def density_integral(self,
                         ray: mi.Ray3f,
                         ellipsoid: Ellipsoid,
                         tmin: mi.Float,
                         tmax: mi.Float,
                         active: mi.Bool) -> mi.Float:

        s = ellipsoid.scale
        w = ellipsoid.rot.T * ray.d / s
        p = ellipsoid.rot.T * (ray.o - ellipsoid.center) / s
        w_norm = dr.rcp(dr.norm(w))
        w *= w_norm

        if self.full_range or (tmin is None and tmax is None):
            B = dr.dot(w, p)
            C = dr.dot(p, p)
            ω = ellipsoid.beta * dr.sum(w)
            D = ellipsoid.beta * dr.sum(p)

            density = w_norm * dr.exp(-0.5 * (dr.fma(-B, B, C) + ω * ω)) * dr.cos(dr.fma(-ω, B, D))
        else:
            active = mi.Bool(active) & (tmin < tmax) & (tmax > 0.0)
            p = p + (tmin / w_norm) * w
            t_limit = dr.clip((tmax - tmin) / w_norm, 0.0, ellipsoid.extent * 2.0)
            tiny = mi.Mask(active & (t_limit < 1e-4))

            B = dr.dot(w, p)
            C = dr.dot(p, p)
            ω = ellipsoid.beta * dr.sum(w)
            D = ellipsoid.beta * dr.sum(p)

            sqrt_half = dr.sqrt(0.5)
            ω_erf = sqrt_half * ω
            re_diff, im_diff = faddeeva_erf_pair_difference_complex(
                mi.Complex2f(sqrt_half * (t_limit + B), -ω_erf),
                mi.Complex2f(sqrt_half * B, -ω_erf))
            phase = dr.fma(-ω, B, D)
            seg = dr.exp(-0.5 * ω * ω) * (dr.cos(phase) * re_diff
                                          - dr.sin(phase) * im_diff) * dr.rcp(dr.sqrt(dr.pi))

            density = w_norm * dr.exp(-0.5 * dr.fma(-B, B, C)) * seg
            # Midpoint rule for tiny segments
            p_mid = p + t_limit * 0.5 * w
            mid = dr.exp(-0.5 * dr.sum(p_mid * p_mid)) * dr.cos(ellipsoid.beta * dr.sum(p_mid))
            density = dr.select(tiny, w_norm * t_limit * mid, density)

        density *= dr.rcp(2.0 * dr.pi * dr.prod(s))
        density[~active] = 0.0
        return density

    def normalization_factor(self, ellipsoid: Ellipsoid) -> mi.Float:
        return dr.rcp(dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(ellipsoid.scale))

#-------------------------------------------------------------------------------
# Faddeeva w(z) complex-erf for the Gabor segment line integral
# We use Weideman's N=16 single-rational approximation

# Weideman N=16 constants (generated offline; Weideman 1994).
_WEIDEMAN_L = 3.3635856610148585
_WEIDEMAN_A = (
    9.939322536485169e-07, 3.9812875836769905e-06, -5.584233405904333e-06,
    -2.734640461595305e-05, 2.170986793497348e-05, 2.107105644503535e-04,
    8.703158429902436e-05, -1.527659738752297e-03, -3.881015190968905e-03,
    3.682567321738441e-03, 5.182240244085487e-02, 1.912417266995708e-01,
    4.692909009399138e-01, 8.864478296072821e-01, 1.362240823992100e+00,
    1.748395894058383e+00,
)
_INV_SQRT_PI = 0.5641895835477563
_SQRT_PI_2 = 0.8862269254527580


def _cmul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def _crcp(br, bi):
    inv = dr.rcp(dr.fma(br, br, bi * bi))
    return br * inv, -bi * inv


def faddeeva_w(zr: mi.Float, zi: mi.Float):
    '''Weideman rational approximation of the Faddeeva function w(z), valid for
    Im(z) >= 0. Returns (real, imag). z = zr + i*zi.'''
    # iz = i*z = -zi + i*zr
    izr, izi = -zi, zr
    lpr, lpi = _WEIDEMAN_L + izr, izi          # L + i*z
    lmr, lmi = _WEIDEMAN_L - izr, -izi         # L - i*z
    rlmr, rlmi = _crcp(lmr, lmi)
    Zr, Zi = _cmul(lpr, lpi, rlmr, rlmi)       # Z = (L + i*z) / (L - i*z)
    # Horner evaluation of the real-coefficient polynomial p(Z).
    pr, pi = mi.Float(_WEIDEMAN_A[0]), mi.Float(0.0)
    for c in _WEIDEMAN_A[1:]:
        pr, pi = _cmul(pr, pi, Zr, Zi)
        pr = pr + c
    # w = 2 p / (L - i*z)^2 + (1/sqrt(pi)) / (L - i*z)
    lm2r, lm2i = _cmul(lmr, lmi, lmr, lmi)
    rlm2r, rlm2i = _crcp(lm2r, lm2i)
    t1r, t1i = _cmul(2.0 * pr, 2.0 * pi, rlm2r, rlm2i)
    return t1r + _INV_SQRT_PI * rlmr, t1i + _INV_SQRT_PI * rlmi


def _erf_folded_complex(x: mi.Float, w: mi.Float, e_max: float = 1e30):
    '''erf(x - i*w) via the folded Faddeeva function, for signed w.

    erf(x - i*w) = 1 - E * e^{2 i x w} * w_f(zeta), E = e^{w^2 - x^2}, with the
    argument reflected into the upper half-plane: zeta = (w + i x) for x >= 0,
    zeta = -(w + i x) for x < 0, giving erf = -+1 -+ E * e^{2 i x w} w_f(zeta).
    E is clamped to `e_max`; this may be an issue if |w| >~ 8.3, but the segment
    contribution is already very small at that point.'''
    E = dr.minimum(dr.exp(dr.fma(w, w, -x * x)), e_max)
    pos = x >= 0.0
    zr = dr.select(pos, w, -w)
    zi = dr.select(pos, x, -x)
    wr, wi = faddeeva_w(zr, zi)
    c, s = dr.cos(2.0 * x * w), dr.sin(2.0 * x * w)
    # P = e^{2 i x w} * w_f(zeta)
    p_re = c * wr - s * wi
    p_im = c * wi + s * wr
    s1 = dr.select(pos, 1.0, -1.0)        # +-1 constant
    sp = dr.select(pos, -1.0, 1.0)        # sign of the E*P term
    re = dr.fma(sp * E, p_re, s1)
    im = sp * E * p_im
    return re, im


def faddeeva_erf_pair_difference_complex(z1: mi.Complex2f, z2: mi.Complex2f):
    '''Returns ((sqrt(pi)/2) * Re[erf(z1) - erf(z2)],
                (sqrt(pi)/2) * Im[erf(z1) - erf(z2)]),  z = x - i*w.

    The asymmetric Gabor segment integral is
        Re[ e^{i*phase} * (sqrt(pi)/2) * (erf(z1) - erf(z2)) ]
      = cos(phase) * re_diff - sin(phase) * im_diff'''
    re1, im1 = _erf_folded_complex(dr.real(z1), -dr.imag(z1))
    re2, im2 = _erf_folded_complex(dr.real(z2), -dr.imag(z2))
    return _SQRT_PI_2 * (re1 - re2), _SQRT_PI_2 * (im1 - im2)


#-------------------------------------------------------------------------------

def safe_rcp(x):
    return dr.select((x != 0.0), dr.rcp(x), 0.0)

def spawn_ray_origin(si, ray, epsilon = 5e-6):
    '''
    Spawn ray from volumetric primitive and return new ray origin
    '''
    return si.p + ray.d * epsilon

def ray_ellipsoid_intersection(ray: mi.Ray3f,
                               ellipsoid: Ellipsoid,
                               active: mi.Bool):
    
    scale = ellipsoid.scale * ellipsoid.extent

    d = ellipsoid.rot.T * ray.d / scale
    o = ellipsoid.rot.T * (ray.o - ellipsoid.center) / scale

    # Plane intersection approach for improved numerical robustness.
    # We define a plane perpendicular to the ray direction that contains
    # the ellipsoid center and intersect it. We then solve the ray-sphere
    # intersection as if the ray origin was this new intersection point.
    plane_t = dr.dot(-o, d) / dr.norm(d)
    plane_p = o + plane_t * d

    a = dr.squared_norm(d)
    b = 2.0 * dr.dot(plane_p, d)
    c = dr.squared_norm(plane_p) - 1.0
    valid, near_t, far_t = mi.math.solve_quadratic(a, b, c)

    # Adjust distances for plane intersection
    near_t += plane_t
    far_t += plane_t

    return valid & active, near_t, far_t

@dataclass
class PrimitiveID:
    shape: mi.ShapePtr = mi.ShapePtr(None)
    index: mi.UInt32 = mi.UInt32(0)

@dr.syntax
def primitive_tracing(scene :mi.Scene,
                      sampler: mi.Sampler,
                      ray: mi.Ray3f,
                      primitives: dr.Local[PrimitiveID],
                      depth: mi.UInt32,
                      callback,
                      payload,
                      active: mi.Bool,
                      max_depth_primitive: int=-1,
                      rr_depth_primitive: int=-1,
                      stochastic_selection_criteria: str='deterministic',
                      orientation_selection: str='deterministic',
                      select_mask: callable=None):
    '''
    Implements the Primitive Tracing algorithm that iterates over segments
    defined by kernel intersections along a ray, accounting for overlaps of
    primitives. A specified callback is called to process each segment. This
    routine is used in the `volprim_prb` integrator to implement volume
    interaction sampling.
    '''
    @dataclass
    class PrimitiveVertex:
        shape: mi.ShapePtr = mi.ShapePtr(None)
        index: mi.UInt32 = mi.UInt32(0)
        t: mi.Float = mi.Float(dr.inf)

    active    = mi.Bool(active)
    ray       = mi.Ray3f(ray)
    accum_t   = mi.Float(0.0)
    weight_rr = mi.Float(1.0)

    if stochastic_selection_criteria == 'deterministic' and orientation_selection == 'deterministic':
        ray.mask = mi.UInt32(255)
    else:
        pyrs = sampler.next_1d()
        needs_ori = orientation_selection not in ['deterministic', 'threshold']
        ori_s = sampler.next_1d() if needs_ori else mi.Float(0)
        ray.mask = select_mask(pyrs, ori_s, ray.d, active, stochastic_selection_criteria)

    # Early exit if the scene doesn't contains volumetric primitives
    if dr.hint(not has_ellipsoids_shapes(scene), mode='scalar'):
        si = scene.ray_intersect(
            ray=ray,
            coherent=(depth == 0),
            ray_flags=mi.RayFlags.All,
            active=active
        )
        return primitives, si, mi.Float(1.0), *payload

    # Stack of vertices to process as segments (allocate one more than max number of primitives)
    vertices = alloc_stack(PrimitiveVertex, mi.UInt32, primitives.alloc_size()+1)

    # Count primitives we have interacted with in the algorithm so far
    primitive_depth = mi.UInt32(0)

    # Surface interaction when intersecting other shapes
    surface_si = dr.zeros(mi.SurfaceInteraction3f)

    active1 = mi.Bool(active)
    while active1:
        # ----------------------------------------------------------------------
        # Create stack of vertices to be processed and remove invalid primitives
        # ----------------------------------------------------------------------

        vertices.clear(active1)
        active2 = mi.Bool(active1) & ~primitives.is_empty()
        count = mi.UInt32(0)
        it2   = mi.UInt32(0)
        while active2:
            prim_id = primitives.value(it2, active2)
            ellipsoid = Ellipsoid.gather(prim_id.shape, prim_id.index, active2)
            valid, t0, t1 = ray_ellipsoid_intersection(ray, ellipsoid, active2) # TODO should primitives be processed every time? Jorge: I tested using the cached exit ts and it is significantly slower
            primitives.write(prim_id, count, valid & (count != it2))
            vertices.push(PrimitiveVertex(prim_id.shape, prim_id.index, t1), valid)
            count[valid] += 1
            it2 += 1
            active2 &= (it2 < primitives.size())
        primitives.resize(count, active1)

        # ----------------------------------------------------------------------
        # Find next intersection
        # ----------------------------------------------------------------------

        # Perform Russian Roulette
        if rr_depth_primitive > 0:
            q = dr.minimum(weight_rr, 0.9)
            perform_rr = (primitive_depth > rr_depth_primitive)
            valid_rr = ((sampler.next_1d() < q) | ~perform_rr)
            weight_rr[active1 & perform_rr] *= dr.rcp(q)
            weight_rr[active1 & ~valid_rr] = 0.0
            active1 &= valid_rr

        # Kill paths that are too long
        if max_depth_primitive > 0:
            weight_rr[active1 & (primitive_depth >= max_depth_primitive)] = 0.0
            active1 &= (primitive_depth < max_depth_primitive)

        si = scene.ray_intersect(
            ray=ray,
            coherent=(primitive_depth == 0),
            ray_flags=mi.RayFlags.All,
            active=active1
        )

        is_ellipsoids = si.shape.is_ellipsoids()
        hit_surface   = active & si.is_valid() & ~is_ellipsoids
        hit_primitive = active & si.is_valid() &  is_ellipsoids

        surface_si[active1] = si
        surface_t = dr.select(hit_surface, si.t, dr.inf)

        vertices.push(PrimitiveVertex(si.shape, si.prim_index, si.t), hit_primitive)


        # ----------------------------------------------------------------------
        # Process segments up to this intersection
        # ----------------------------------------------------------------------

        seg_t0 = mi.Float(0.0)
        it3 = mi.UInt32(0)
        active3 = mi.Bool(active1) & ~primitives.is_empty()
        while active3:
            # ------------------------------------------------------------------
            # Find the closest vertex greater than the segment t0
            # ------------------------------------------------------------------

            it4 = mi.UInt32(0)
            active4 = mi.Bool(active3)
            vertex = PrimitiveVertex(mi.ShapePtr(None), mi.UInt32(0), mi.Float(dr.inf))
            while active4:
                v = vertices.value(it4, active4)
                is_closer = active4 & (v.t > seg_t0) & (v.t < vertex.t)
                vertex.shape[is_closer] = v.shape
                vertex.index[is_closer] = v.index
                vertex.t[is_closer] = v.t
                it4 += 1
                active4 &= (it4 < vertices.size())

            # This next vertex is the segment farther bound
            reached_surface = vertex.t > surface_t
            seg_t1 = dr.select(reached_surface, surface_t, vertex.t)

            # ------------------------------------------------------------------
            # Call segment callback
            # ------------------------------------------------------------------

            active_c, payload = callback(payload, primitives, ray, accum_t, seg_t0, seg_t1, active3)
            # TODO remove accum_t and set seg_t0/seg_t1 relative to original ray

            # Check whether the callback requested stopping here
            active3 &= active_c
            active1 &= active_c

            # ------------------------------------------------------------------
            # Remove vertex.index from primitives if necessary
            # ------------------------------------------------------------------

            # Check whether we have reach the primitive/surface we intersected last
            reached_si = si.is_valid() & (((si.prim_index == vertex.index) & (si.shape == vertex.shape)) | reached_surface)

            should_remove = mi.Bool(active3) & ~reached_si
            active6 = mi.Bool(should_remove) & (primitives.size() > 1)
            shift = mi.Bool(False)
            it6 = mi.UInt32(0)
            while active6:
                p = primitives.value(it6, active6)
                shift |= active6 & (p.shape == vertex.shape) & (p.index == vertex.index)
                primitives.write(primitives.value(it6 + 1, shift), it6, shift)
                it6 += 1
                active6 &= (it6 < (primitives.size() - 1))
            primitives.pop(should_remove)

            # ------------------------------------------------------------------
            # Continue with next segment
            # ------------------------------------------------------------------

            # Break if we are entering the primitive that we intersected last
            active3 &= ~reached_si

            # Otherwise continue with the next segment
            seg_t0[active3] = mi.Float(seg_t1)

            # Check whether we have already visited all the vertices
            it3 += 1
            active3 &= (it3 < vertices.size())

        # Stop if we processed all vertices and ray escaped or hit a surface
        active1 &= hit_primitive

        # Add primitive we are entering into to `primitives`
        valid_alloc = active1 & (primitives.size() < (primitives.alloc_size()))

        primitives.push(PrimitiveID(si.shape, si.prim_index), valid_alloc)
        primitive_depth[active1] += 1

        ray.o = spawn_ray_origin(si, ray)
        accum_t += si.t

    return primitives, surface_si, weight_rr, *payload

#-------------------------------------------------------------------------------

class EllipsoidsFactory:
    '''
    Helper class to build Ellipsoids datasets for testing purposes
    '''
    def __init__(self):
        self.centers = []
        self.scales = []
        self.quaternions = []
        self.sigmats = []
        self.albedos = []

    def add(self, mean, scale, sigmat=1.0, albedo=1.0, euler=[0.0, 0.0, 0.0]):
        self.centers.append(mi.ScalarPoint3f(mean))
        self.scales.append(mi.ScalarVector3f(scale))
        quaternion = dr.slice(dr.euler_to_quat(dr.deg2rad(mi.ScalarPoint3f(euler))))
        self.quaternions.append(mi.ScalarQuaternion4f(quaternion))
        self.sigmats.append(sigmat)
        if isinstance(albedo, float):
            albedo = mi.ScalarColor3f(albedo)
        self.albedos.append(albedo)

    def build(self):
        import numpy as np
        num_gaussians = len(self.centers)
        centers = mi.TensorXf(np.ravel(np.array(self.centers)), shape=(num_gaussians, 3))
        scales  = mi.TensorXf(np.ravel(np.array(self.scales)), shape=(num_gaussians, 3))
        quats   = mi.TensorXf(np.ravel(np.array(self.quaternions)), shape=(num_gaussians, 4))
        sigmats = mi.TensorXf(self.sigmats, shape=(num_gaussians, 1))
        self.albedos = np.array(self.albedos).reshape((num_gaussians, -1))
        albedos = mi.TensorXf(np.array(self.albedos))
        return centers, scales, quats, sigmats, albedos

class GaborFactory:
    '''
    Helper class to build Gabor datasets for testing purposes
    '''
    def __init__(self):
        self.centers = []
        self.scales = []
        self.quaternions = []
        self.sigmats = []
        self.albedos = []
        self.omega = []

    def add(self, mean, scale, sigmat=1.0, albedo=1.0, euler=[0.0, 0.0, 0.0], omega=1.0):
        self.centers.append(mi.ScalarPoint3f(mean))
        self.scales.append(mi.ScalarVector3f(scale))
        quaternion = dr.slice(dr.euler_to_quat(dr.deg2rad(mi.ScalarPoint3f(euler))))
        self.quaternions.append(mi.ScalarQuaternion4f(quaternion))
        self.sigmats.append(sigmat)
        if isinstance(albedo, float):
            albedo = mi.ScalarColor3f(albedo)
        self.albedos.append(albedo)
        self.omega.append(omega)

    def build(self):
        import numpy as np
        num_gaussians = len(self.centers)
        centers = mi.TensorXf(np.ravel(np.array(self.centers)), shape=(num_gaussians, 3))
        scales  = mi.TensorXf(np.ravel(np.array(self.scales)), shape=(num_gaussians, 3))
        quats   = mi.TensorXf(np.ravel(np.array(self.quaternions)), shape=(num_gaussians, 4))
        sigmats = mi.TensorXf(self.sigmats, shape=(num_gaussians, 1))
        self.albedos = np.array(self.albedos).reshape((num_gaussians, -1))
        albedos = mi.TensorXf(np.array(self.albedos))
        omega = mi.TensorXf(self.omega, shape=(num_gaussians, 1))
        return centers, scales, quats, sigmats, albedos, omega