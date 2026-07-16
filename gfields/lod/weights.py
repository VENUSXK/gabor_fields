# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Level-of-detail: the single home for all frequency-band / LOD math.

Two runtimes, one source of truth for the Gabor Fourier band:

* **numpy (offline)** -- :func:`gabor_frequency_band`, :func:`gabor_band_weights`,
  :func:`build_freq_threshold_lut`, :func:`frequency_threshold_fraction`. Used by
  the interactive viewer and the offline LOD-video schedule.
* **Dr.Jit (runtime)** -- :func:`continuous_lod_weight` (the twin of the numpy
  band weight, evaluated per primitive inside the integrators) and
  :class:`LODSelector` (stochastic pyramid-level / orientation selection via
  per-ray visibility masks). Consumed by ``gfields_prb`` / ``gfields_tomography``.

Visibility-mask encodings
-------------------------
``simple``     bit ``l`` = pyramid level ``l`` (Gaussian base at bit 0).
``steerable``  bit 7 = Gaussian base; bits 6..0 = Gabor sub-bands laid out
               as ``bit = 7 - (1 + (level-1) * num_orientations + ori)``.
"""

from __future__ import annotations

import numpy as np

import drjit as dr
import mitsuba as mi

from gfields.lod.utils import steerable_subband_bit


# ===========================================================================
# numpy band math (offline: 3D Gabor Fourier band = peak +/- envelope spread)
# ===========================================================================

def gabor_frequency_band(
    omega: np.ndarray, scales: np.ndarray, *, band_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-primitive Gabor frequency band ``[f_lo, f_hi]`` from the analytic 3D
    Fourier transform.

    The kernel ``exp(-||p||^2 / 2) * cos(omega . p)`` is a unit-variance
    Gaussian envelope modulating a cosine: its spectrum is not a delta at
    the peak frequency but a pair of Gaussian lobes of standard deviation
    ``||1/scale||`` centered at ``||omega/scale||``. Treating the primitive
    as a single peak frequency ignores that extent, so a hard threshold on
    the peak makes primitives pop in/out discretely. ``band_sigma`` sets how
    many envelope standard deviations of extent define the significant band
    edges around the peak.
    """
    omega = np.asarray(omega).reshape(-1, 1)
    scales = np.asarray(scales).reshape(-1, 3)
    inv_scales = 1.0 / np.maximum(scales, 1e-8)
    f_peak = np.linalg.norm(omega * inv_scales, axis=-1)
    f_spread = band_sigma * np.linalg.norm(inv_scales, axis=-1)
    f_lo = np.maximum(f_peak - f_spread, 0.0)
    f_hi = f_peak + f_spread
    return f_lo, f_hi


def gabor_band_weights(
    f_lo: np.ndarray, f_hi: np.ndarray, threshold: float,
) -> np.ndarray:
    """
    Continuous per-primitive visibility weight in ``[0, 1]``, smoothstep-shaped.

    Maps the fraction of each primitive's frequency band ``[f_lo, f_hi]``
    lying at or below ``threshold`` through a cubic smoothstep (3t^2-2t^3).
    Compared to a linear ramp the smoothstep has zero derivative at both band
    edges, so primitives ease into and out of visibility rather than starting
    or ending their fade with an abrupt velocity change. This matches the
    cosine-shaped temporal schedule in ``frequency_threshold_fraction``:
    both curves apply the same S-shaping principle, one over the whole
    timeline and one over each primitive's individual spectral band.
    """
    span = np.maximum(f_hi - f_lo, 1e-8)
    t = np.clip((threshold - f_lo) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_freq_threshold_lut(
    gabor_bands: dict[int, tuple[np.ndarray, np.ndarray]],
    freq_threshold_max: float,
    *,
    n_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a lookup table that maps visibility-fraction -> frequency threshold.

    The total weighted opacity (sum of gabor_band_weights over all primitives)
    is not linear in threshold when primitives are non-uniformly distributed
    across frequency. A plain linear sweep in threshold causes no visible
    change for most of the video and then a rapid collapse at the end once the
    threshold finally hits the dense frequency cluster.  Inverting the CDF of
    total visibility vs. threshold remaps the sweep so that equal visual weight
    changes per unit time, keeping the fade progressive throughout.

    This is just to render smoother continuous LOD videos for the slides.
    """
    if not gabor_bands or freq_threshold_max < 1e-12:
        return np.array([0.0, 1.0]), np.array([0.0, max(freq_threshold_max, 0.0)])

    all_f_lo = np.concatenate([f_lo for f_lo, _ in gabor_bands.values()])
    all_f_hi = np.concatenate([f_hi for _, f_hi in gabor_bands.values()])

    thresholds = np.linspace(0.0, freq_threshold_max, n_samples)
    total_w = np.fromiter(
        (gabor_band_weights(all_f_lo, all_f_hi, float(t)).sum() for t in thresholds),
        dtype=np.float64,
        count=n_samples,
    )

    w_max = float(total_w[-1])
    if w_max < 1e-12:
        return np.array([0.0, 1.0]), np.array([0.0, freq_threshold_max])
    fractions = total_w / w_max

    # interp requires strictly increasing x; deduplicate flat regions
    _, keep = np.unique(fractions, return_index=True)
    return fractions[keep], thresholds[keep]


def frequency_threshold_fraction(u: float) -> float:
    """
    Symmetric video schedule over normalized time u in [0, 1].

    Returns a visibility fraction in [0, 1] passed to the LUT built by
    ``build_freq_threshold_lut`` to obtain the actual frequency threshold.
    Linear (triangle-wave) so the fraction decreases at a constant rate:
      u=0   -> fraction 1 (full Gabor spectrum visible)
      u=0.5 -> fraction 0 (Gaussians only)
      u=1   -> fraction 1 again
    The LUT's CDF inversion is what ensures equal visual change per frame.

    This is also just to render pretty LOD videos for the slides.
    """
    return 1.0 - 2.0 * min(u, 1.0 - u)


# ===========================================================================
# Dr.Jit runtime: continuous weight + stochastic selection
# ===========================================================================

#: Selection criteria that need no importance-sampling weights
DETERMINISTIC_CRITERIA = ('deterministic', 'gaussian_ms')

STOCHASTIC_CRITERIA = (
    'deterministic', 'gaussian_ms', 'power_law',
    'control_variate_uniform', 'control_variate_power_law',
    'control_variate_power_law_accumulating', 'biased_zero',
)

ORIENTATION_CRITERIA = (
    'deterministic', 'threshold', 'importance',
    'control_variate_threshold',
)


def continuous_lod_weight(ellipsoid, freq_threshold, band_sigma,
                          active) -> mi.Float:
    """Smooth visibility weight for continuous frequency-domain filtering.

    Mirrors the numpy implementation (:func:`gabor_frequency_band` /
    :func:`gabor_band_weights`) used for offline LOD videos: each Gabor
    primitive occupies the spectral band ``f_peak +/- band_sigma * f_spread``
    with ``f_peak = ||beta / scale||`` and ``f_spread = ||1 / scale||``
    (analytic Fourier extent of the Gabor envelope). The weight ramps
    smoothly from 1 (band fully below the threshold) to 0 (band fully
    above), so primitives fade instead of popping. Gaussians (``omega == 0``)
    always have weight 1.

    A negative ``freq_threshold`` disables filtering (weight 1 everywhere).
    """
    inv_scale = dr.rcp(dr.maximum(ellipsoid.scale, 1e-12))
    f_peak = dr.norm(ellipsoid.beta * inv_scale)
    f_spread = band_sigma * dr.norm(inv_scale)
    f_lo = dr.maximum(f_peak - f_spread, 0.0)
    f_hi = f_peak + f_spread

    # Fraction of the band below the threshold, smoothstepped.
    t = dr.clip((freq_threshold - f_lo) / dr.maximum(f_hi - f_lo, 1e-12), 0.0, 1.0)
    w = t * t * dr.fma(-2.0, t, 3.0)

    # A Gaussian is a Gabor with omega == 0; it is never frequency-filtered.
    is_gabor = ellipsoid.beta != 0.0
    w = dr.select(is_gabor, w, mi.Float(1.0))
    w = dr.select(freq_threshold < 0.0, mi.Float(1.0), w)
    return dr.select(active, w, mi.Float(1.0))


class LODSelector:
    """Stochastic pyramid-level / orientation selection via visibility masks.

    Construct from an ``mi.Properties`` (consumes the LOD-related plugin
    properties) and query per-ray masks with :meth:`select_mask` and
    per-shape importance weights with :meth:`weight_for_shape`. The main
    path and NEE run independent criteria; pass ``for_nee=True`` to use the
    NEE configuration.
    """

    def __init__(self, props, allow_dynamic_orientation=False):
        """
        Args:
            props: ``mi.Properties`` holding the LOD plugin properties.
            allow_dynamic_orientation: permit the ray-direction-dependent
                orientation modes ('importance', 'control_variate_threshold').
                Valid whenever the mask is selected with the same ``ray.d`` used
                to evaluate ``weight_for_shape`` along that segment: tomography
                (straight rays) and ``gfields_prb`` (the per-bounce mask is
                re-selected at each scatter vertex with the new direction, so
                within a segment the direction is constant). Leave False only for
                integrators that cannot guarantee this; the modes then fall back
                to 'deterministic'.
        """
        self.pyramid_level_count = int(props.get('pyramid_level_count', 1))
        self.asset_encoding = str(props.get('asset_encoding', 'simple'))
        assert self.asset_encoding in ('simple', 'steerable')

        self.num_orientations = int(props.get('num_orientations', 3))
        if self.asset_encoding == 'steerable':
            # Steerable bit layout: bit 7 = Gaussian, bits 6..0 = Gabor
            # sub-bands -> 1 + (levels-1) * orientations must fit in 8 bits.
            n_subbands = 1 + (self.pyramid_level_count - 1) * self.num_orientations
            if n_subbands > 8:
                raise Exception(
                    f"Steerable encoding overflows the 8-bit visibility mask: "
                    f"1 + ({self.pyramid_level_count}-1) * {self.num_orientations} "
                    f"= {n_subbands} sub-bands > 8.")
        elif self.pyramid_level_count > 8:
            raise Exception(
                f"pyramid_level_count must be <= 8 (one visibility-mask bit "
                f"per level), got {self.pyramid_level_count}.")

        self.criteria = str(props.get('stochastic_selection_criteria', 'deterministic'))
        self.criteria_nee = str(props.get('stochastic_selection_criteria_nee', 'deterministic'))
        for c, name in ((self.criteria, 'stochastic_selection_criteria'),
                        (self.criteria_nee, 'stochastic_selection_criteria_nee')):
            if c not in STOCHASTIC_CRITERIA:
                raise Exception(f"Invalid {name}: '{c}' (one of {STOCHASTIC_CRITERIA})")
        if self.criteria_nee == 'gaussian_ms':
            raise Exception(
                "'gaussian_ms' is a primary-path-only criterion and is not "
                "valid for stochastic_selection_criteria_nee.")
        if self.criteria == 'biased_zero':
            raise Exception("'biased_zero' is an NEE-only criterion.")

        self.power_law_beta = float(props.get('power_law_beta', 0.75))
        self.power_law_beta_nee = float(props.get('power_law_beta_nee', 0.75))

        # Steerable orientation parameters
        ori_sel = str(props.get('orientation_selection', 'deterministic'))
        ori_sel_nee = str(props.get('orientation_selection_nee', 'deterministic'))

        for attr, val in (('orientation_selection', ori_sel),
                          ('orientation_selection_nee', ori_sel_nee)):
            if val not in ORIENTATION_CRITERIA:
                raise Exception(f"Invalid {attr}: '{val}' (one of {ORIENTATION_CRITERIA})")
            if (val in ('importance', 'control_variate_threshold')
                    and not allow_dynamic_orientation):
                mi.Log(mi.LogLevel.Warn,
                       f"[LODSelector] {attr}='{val}' requires straight rays "
                       "(tomography); falling back to 'deterministic'.")
                val = 'deterministic'
            setattr(self, attr, val)

        self.orientation_threshold = float(props.get('orientation_threshold', 0.7))
        self.orientation_decay_rate = float(props.get('orientation_decay_rate', 1.5))
        self.importance_mix_weight = float(props.get('importance_mix_weight', 0.5))

        # Bin rotation: quaternion rotating bin directions for rotated assets
        self.bin_rotation = None
        if props.has_property('bin_rotation'):
            quat_data = props.get('bin_rotation')
            if isinstance(quat_data, (mi.Quaternion4f, mi.ScalarQuaternion4f)):
                self.bin_rotation = mi.Quaternion4f(quat_data)
            elif hasattr(quat_data, '__len__') and len(quat_data) == 4:
                self.bin_rotation = mi.Quaternion4f(quat_data[0], quat_data[1],
                                                    quat_data[2], quat_data[3])

        self.is_gauss_ms = (self.criteria == 'gaussian_ms')
        self._gauss_only_active = mi.Bool(False)

        # Power-law warps
        self._plf = lambda x: dr.power(x, 1.0 / (1.0 - self.power_law_beta))
        self._plf_nee = lambda x: dr.power(x, 1.0 / (1.0 - self.power_law_beta_nee))

        all_pure_deterministic = (
            self.orientation_selection == 'deterministic'
            and self.orientation_selection_nee == 'deterministic'
            and self.criteria == 'deterministic'
            and self.criteria_nee == 'deterministic')
        if not all_pure_deterministic or self.asset_encoding == 'steerable':
            self._init_orientations()
        if self.main_needs_weights() or self.nee_needs_weights():
            self._precompute_level_weights()

    # ------------------------------------------------------------- queries

    def main_needs_weights(self):
        """True when the primary path needs importance-sampling weights."""
        return (self.criteria not in DETERMINISTIC_CRITERIA
                or self.orientation_selection != 'deterministic')

    def nee_needs_weights(self):
        """True when the NEE path needs importance-sampling weights."""
        return (self.criteria_nee not in ('deterministic', 'biased_zero')
                or self.orientation_selection_nee != 'deterministic')

    def is_deterministic(self):
        """True when neither path performs stochastic selection."""
        return self.main_is_deterministic() and self.nee_is_deterministic()

    def main_is_deterministic(self):
        """True when the primary path performs no stochastic selection."""
        return (self.criteria == 'deterministic'
                and self.orientation_selection == 'deterministic')

    def nee_is_deterministic(self):
        """True when the NEE path performs no stochastic selection."""
        return (self.criteria_nee == 'deterministic'
                and self.orientation_selection_nee == 'deterministic')

    def set_gauss_only(self, lanes: mi.Bool):
        """Set the per-lane gaussian_ms override (no-op in other modes)."""
        self._gauss_only_active = mi.Bool(lanes)

    def gauss_mask_value(self):
        """Visibility-mask bit matching only the Gaussian (pyr0) layer."""
        return mi.UInt32(1 << 7) if self.asset_encoding == 'steerable' else mi.UInt32(1)

    def apply_gauss_only(self, mask, for_nee=False):
        """Collapse `mask` to the Gaussian layer on lanes flagged via
        :meth:`set_gauss_only` (no-op unless criterion is 'gaussian_ms', or
        for NEE, which never applies the primary-path override)."""
        if for_nee or not self.is_gauss_ms:
            return mask
        return dr.select(self._gauss_only_active, self.gauss_mask_value(), mask)

    def subband_bit(self, level, ori=-1):
        """Visibility-mask bit index of a (level, orientation) sub-band in
        the steerable encoding (bit 7 = Gaussian base; orientation bins of
        level l >= 1 descend from bit 6)."""
        return steerable_subband_bit(level, ori, self.num_orientations)

    # ------------------------------------------------------ initialisation

    def _init_orientations(self):
        """Dr.Jit orientation vectors from the canonical table (must match the
        offline decomposition in ``gfields.lod.decompose``)."""
        from gfields.lod.utils import orientation_vectors

        base = [mi.Vector3f(float(v[0]), float(v[1]), float(v[2]))
                for v in orientation_vectors(self.num_orientations)]
        if self.bin_rotation is not None:
            rot = dr.quat_to_matrix(mi.Matrix3f, self.bin_rotation)
            self.orientations = [rot @ ori for ori in base]
        else:
            self.orientations = base

    def _precompute_level_weights(self):
        """Per-visibility-mask importance weights for each criterion."""
        n = self.pyramid_level_count
        n_ori = self.num_orientations
        steerable = self.asset_encoding == 'steerable'

        def compute_freq_weights(criteria, b):
            fw = [1.0] * n
            if criteria == 'power_law':
                for i in range(n):
                    lo, hi = i / n, (i + 1) / n
                    fw[i] = 1.0 / (hi**(1 - b) - lo**(1 - b))
            elif criteria == 'control_variate_uniform':
                fw[0] = 1.0
                for i in range(1, n):
                    fw[i] = float(n - 1)
            elif criteria == 'control_variate_power_law':
                fw[0] = 1.0
                if n > 1:
                    for i in range(1, n):
                        lo, hi = (i - 1) / (n - 1), i / (n - 1)
                        fw[i] = 1.0 / (hi**(1 - b) - lo**(1 - b))
            elif criteria == 'control_variate_power_law_accumulating':
                for i in range(n):
                    fw[i] = 1.0 if i == 0 else 1.0 / max(1.0 - (i / n)**(1 - b), 1e-8)
            return fw

        def populate_map(freq_weights, ori_w):
            wmap = {}
            if steerable:
                wmap[1 << 7] = freq_weights[0]
                for lvl in range(1, n):
                    for oi in range(n_ori):
                        bit = steerable_subband_bit(lvl, oi, n_ori)
                        if bit >= 0:
                            wmap[1 << bit] = freq_weights[lvl] * ori_w
            else:
                for lvl in range(n):
                    wmap[1 << lvl] = freq_weights[lvl]
            return wmap

        fw = compute_freq_weights(self.criteria, self.power_law_beta)
        fw_nee = compute_freq_weights(self.criteria_nee, self.power_law_beta_nee)
        # Static orientation modes ('deterministic', 'threshold') carry no
        # importance weight; only the frequency-level weight applies.
        self.level_weight_map = populate_map(fw, 1.0)
        self.nee_level_weight_map = populate_map(fw_nee, 1.0)

        # Freq-only weights for dynamic orientation modes (steerable only)
        if steerable:
            self.freq_weight_map = {1 << 7: fw[0]}
            self.freq_weight_map_nee = {1 << 7: fw_nee[0]}
            for lvl in range(1, n):
                for oi in range(n_ori):
                    bit = steerable_subband_bit(lvl, oi, n_ori)
                    if bit >= 0:
                        self.freq_weight_map[1 << bit] = fw[lvl]
                        self.freq_weight_map_nee[1 << bit] = fw_nee[lvl]

    # ------------------------------------------------- orientation helpers

    def _compute_alignments(self, ray_d):
        """Alignment factors |dot(ray, ori)| for each orientation."""
        ray_d_n = ray_d / dr.maximum(dr.norm(ray_d), 1e-8)
        return [dr.abs(dr.dot(ray_d_n, ori)) for ori in self.orientations]

    def _compute_ori_weights(self, alignments):
        """Contribution weights exp(-decay * alignment^2)."""
        return [dr.exp(-self.orientation_decay_rate * a * a) for a in alignments]

    def _select_orientation(self, ray_d, freq_level, ori_sample, active, for_nee=False):
        """Orientation mask for one Gabor level (steerable encoding)."""
        n_ori = self.num_orientations
        ori_sel = self.orientation_selection_nee if for_nee else self.orientation_selection

        def bit_for_ori(oi):
            return steerable_subband_bit(freq_level, oi, n_ori)

        if ori_sel == 'deterministic':
            mask = mi.UInt32(0)
            for oi in range(n_ori):
                bit = bit_for_ori(oi)
                if bit >= 0:
                    mask |= mi.UInt32(1 << bit)
            return mask

        if ori_sel == 'threshold':
            alignments = self._compute_alignments(ray_d)
            thresh = mi.Float(self.orientation_threshold)
            mask = mi.UInt32(0)
            for oi, a in enumerate(alignments):
                bit = bit_for_ori(oi)
                if bit >= 0:
                    mask = dr.select(active & (a <= thresh),
                                     mask | mi.UInt32(1 << bit), mask)
            return mask

        if ori_sel == 'importance':
            # Sample one orientation weighted by its expected contribution
            alignments = self._compute_alignments(ray_d)
            weights = self._compute_ori_weights(alignments)
            total = dr.maximum(sum(weights), 1e-8)
            alpha, unif_p = self.importance_mix_weight, 1.0 / n_ori
            mixed = [alpha * (w / total) + (1 - alpha) * unif_p for w in weights]
            cdf, cumsum = [], mi.Float(0.0)
            for p in mixed:
                cumsum = cumsum + p
                cdf.append(cumsum)
            mask = mi.UInt32(0)
            for oi in range(n_ori):
                lower = mi.Float(0.0) if oi == 0 else cdf[oi - 1]
                in_rng = active & (ori_sample >= lower) & (ori_sample < cdf[oi])
                bit = bit_for_ori(oi)
                if bit >= 0:
                    mask = dr.select(in_rng, mi.UInt32(1 << bit), mask)
            return mask

        if ori_sel == 'control_variate_threshold':
            # Below-threshold orientations deterministic + sample one above
            alignments = self._compute_alignments(ray_d)
            thresh = mi.Float(self.orientation_threshold)
            base_mask, num_above = mi.UInt32(0), mi.Float(0.0)
            above_info = []
            for oi, a in enumerate(alignments):
                bit = bit_for_ori(oi)
                below, above = a <= thresh, a > thresh
                if bit >= 0:
                    base_mask = dr.select(active & below,
                                          base_mask | mi.UInt32(1 << bit), base_mask)
                num_above = dr.select(active & above, num_above + 1.0, num_above)
                above_info.append((oi, above))
            stoch_mask = mi.UInt32(0)
            cumcount = mi.Float(0.0)
            for oi, above in above_info:
                prev = cumcount
                cumcount = dr.select(above, cumcount + 1.0, cumcount)
                safe_n = dr.maximum(num_above, 1.0)
                lower, upper = prev / safe_n, cumcount / safe_n
                in_rng = active & above & (ori_sample >= lower) & (ori_sample < upper)
                bit = bit_for_ori(oi)
                if bit >= 0:
                    stoch_mask = dr.select(in_rng, mi.UInt32(1 << bit), stoch_mask)
            return base_mask | stoch_mask

        raise Exception(f"Invalid orientation_selection: '{ori_sel}'")

    # ------------------------------------------------------ mask selection

    def select_mask(self, pyrs, ori_sample, ray_d, active, for_nee=False) -> mi.UInt32:
        """Per-ray visibility mask for the configured criterion.

        Args:
            pyrs: uniform [0,1) sample driving level selection.
            ori_sample: uniform [0,1) sample driving orientation selection.
            ray_d: ray direction (orientation-dependent modes).
            active: lane mask.
            for_nee: use the NEE criterion instead of the primary one.
        """
        criteria = self.criteria_nee if for_nee else self.criteria
        n = self.pyramid_level_count
        steerable = self.asset_encoding == 'steerable'

        def gauss_mask():
            return self.gauss_mask_value()

        def level_mask(lvl, lanes):
            if steerable:
                if lvl == 0:
                    return gauss_mask()
                return self._select_orientation(ray_d, lvl, ori_sample, lanes, for_nee)
            return mi.UInt32(1 << lvl)

        def apply_gauss_only(mask):
            return self.apply_gauss_only(mask, for_nee)

        if criteria == 'biased_zero':
            return gauss_mask()

        if criteria in ('deterministic', 'gaussian_ms'):
            mask = gauss_mask()
            for lvl in range(1, n):
                mask = mask | level_mask(lvl, active)
            return apply_gauss_only(mask)

        plf = self._plf_nee if for_nee else self._plf
        sample = plf(pyrs) if 'power_law' in criteria else pyrs

        if criteria == 'power_law':
            level = mi.UInt32(dr.clamp(sample * n, 0, n - 1))
            mask = mi.UInt32(0)
            for lvl in range(n):
                is_lvl = level == mi.UInt32(lvl)
                mask = dr.select(is_lvl, level_mask(lvl, active & is_lvl), mask)
            return apply_gauss_only(mask)

        if criteria in ('control_variate_uniform', 'control_variate_power_law'):
            num_high = n - 1
            high_idx = mi.UInt32(dr.clamp(sample * num_high, 0, num_high - 1))
            selected_lvl = high_idx + 1
            mask = gauss_mask()
            for lvl in range(1, n):
                is_lvl = selected_lvl == mi.UInt32(lvl)
                mask = dr.select(is_lvl, mask | level_mask(lvl, active & is_lvl), mask)
            return apply_gauss_only(mask)

        if criteria == 'control_variate_power_law_accumulating':
            level_k = mi.UInt32(dr.clamp(sample * n, 0, n - 1))
            mask = gauss_mask()
            for lvl in range(1, n):
                inc = level_k >= mi.UInt32(lvl)
                mask = dr.select(inc, mask | level_mask(lvl, active & inc), mask)
            return apply_gauss_only(mask)

        raise Exception(f"Invalid selection criteria: {criteria}")

    # ----------------------------------------------------- shape weighting

    def weight_for_shape(self, shape, ray_d, active, for_nee=False) -> mi.Float:
        """Importance-sampling weight for a shape's visibility mask."""
        criteria = self.criteria_nee if for_nee else self.criteria
        ori_sel = self.orientation_selection_nee if for_nee else self.orientation_selection
        steerable = self.asset_encoding == 'steerable'

        if criteria in DETERMINISTIC_CRITERIA and ori_sel == 'deterministic':
            return mi.Float(1.0)
        if criteria == 'biased_zero' and for_nee:
            return mi.Float(1.0)

        vis_mask = shape.visibility_mask()
        weight_map = self.nee_level_weight_map if for_nee else self.level_weight_map

        # Static modes: one precomputed weight per visibility mask
        if not steerable or ori_sel not in ('importance', 'control_variate_threshold'):
            weight = mi.Float(1.0)
            for mask_val, w in weight_map.items():
                weight = dr.select(active & (vis_mask == mi.UInt32(mask_val)),
                                   mi.Float(w), weight)
            return weight

        # Dynamic orientation modes: freq_weight x ori_weight
        freq_map = self.freq_weight_map_nee if for_nee else self.freq_weight_map
        freq_weight = mi.Float(1.0)
        for mask_val, w in freq_map.items():
            freq_weight = dr.select(active & (vis_mask == mi.UInt32(mask_val)),
                                    mi.Float(w), freq_weight)
        return freq_weight * self._ori_weight_for_shape(shape, ray_d, active, for_nee)

    def _ori_weight_for_shape(self, shape, ray_d, active, for_nee):
        """Orientation weight for steerable assets (dynamic ori modes only)."""
        ori_sel = self.orientation_selection_nee if for_nee else self.orientation_selection
        n_ori = self.num_orientations
        vis_mask = shape.visibility_mask()
        is_gaussian = vis_mask == mi.UInt32(1 << 7)

        def is_shape_ori(oi):
            m = mi.Bool(False)
            for lvl in range(1, self.pyramid_level_count):
                bit = steerable_subband_bit(lvl, oi, n_ori)
                if bit >= 0:
                    m = m | (vis_mask == mi.UInt32(1 << bit))
            return m

        if ori_sel == 'importance':
            alignments = self._compute_alignments(ray_d)
            weights = self._compute_ori_weights(alignments)
            total = dr.maximum(sum(weights), 1e-8)
            alpha, unif_p = self.importance_mix_weight, 1.0 / n_ori
            ori_weight = mi.Float(1.0)
            for oi in range(n_ori):
                p_mixed = alpha * (weights[oi] / total) + (1 - alpha) * unif_p
                ori_weight = dr.select(active & is_shape_ori(oi),
                                       dr.rcp(dr.maximum(p_mixed, 1e-8)), ori_weight)
            return dr.select(is_gaussian, mi.Float(1.0), ori_weight)

        if ori_sel == 'control_variate_threshold':
            alignments = self._compute_alignments(ray_d)
            thresh = mi.Float(self.orientation_threshold)
            num_above = mi.Float(0.0)
            for a in alignments:
                num_above = dr.select(a > thresh, num_above + 1.0, num_above)
            ori_weight = mi.Float(1.0)
            for oi in range(n_ori):
                is_above = alignments[oi] > thresh
                ori_weight = dr.select(active & is_shape_ori(oi) & is_above,
                                       num_above, ori_weight)
            return dr.select(is_gaussian, mi.Float(1.0), ori_weight)

        return mi.Float(1.0)
