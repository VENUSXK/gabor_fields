# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Interactive polyscope viewer for volumetric-primitive assets.

Features:
- progressive sample accumulation (one sample per frame) while the camera and
  parameters are untouched; instant reset on interaction,
- live per-sample cost readout (GPU kernel execution time via Dr.Jit's kernel
  history) so the price of each acceleration setting is visible,
- continuous frequency-domain LOD slider (perceptually equalized through the
  asset's visibility CDF), driven through ``mi.traverse`` without scene
  rebuilds,
- tomography / path-traced (prb / prb stochastic) integrator switch with
  **all** the exposed performance options (stochastic LOD criteria for
  primary and NEE paths, power-law betas, Russian-roulette depths, solver,
  primitive overlap budget, phase function, kernel options, the stochastic
  sampler's NEE-transmittance / shell-truncation / residual-clamp modes, and
  the any-hit fast paths -- on by default wherever an integrator supports
  them),
- intuitive environment-map rotation (azimuth/elevation) and intensity,
- real-time exposure tonemapping, and a single **Save frame** capture that
  writes EXR + tonemapped PNG + a JSON sidecar (accumulated samples, full
  integrator configuration, camera and environment) plus full Mitsuba
  XML + per-level PLY scene export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

import drjit as dr
import mitsuba as mi

from gfields import lod as L  # numpy band math (gabor_frequency_band, LUT)
from gfields.rendering import pipeline as P  # asset detection + path resolution
from gfields.lod import utils as A  # pyramid-level numpy plumbing
from gfields.rendering.pipeline import (
    MS_ALBEDO_DEFAULT,
    MS_INTEGRATOR_DEFAULTS,
    TOMOGRAPHY_MAX_DEPTH_DEFAULT,
    detect_asset_info,
    load_steerable_asset,
)
from gfields.io import is_steerable_asset
from gfields.training.adaptive import clamp_extents
from . import exporters
from .render_session import RenderSession


from .sample_accum_helpers import (
    buffer_is_linear_tau,
    interpret_tomography_buffer,
    needs_mean_tau_accumulation,
    tau_to_radiance,
    tomography_criteria_choices,
    tomography_integrator_props,
)

# Kernel / solver / phase choices exposed in the UI. 'gabor' is the canonical
# kernel name (a Gaussian is its omega=0 special case; the plugins still accept
# the legacy 'gabor_gaussian_mixture' alias). The stochastic criteria lists are
# pulled from ``gfields.lod`` at launch so they can never drift from what the
# integrators actually accept.
KERNEL_TYPES = ['gabor', 'gaussian']
SOLVER_TYPES = ['bisection', 'newton', 'disabled']
PHASE_TYPES = ['isotropic', 'hg']
INTEGRATOR_MODES = ['tomography', 'path tracing (prb)',
                    'path tracing (prb stochastic)']
# 'analytic' first: the plugin default and the better estimator at equal
# variance (docs/prb_stochastic.md); 'stochastic' is kept for comparison.
NEE_TRANSMITTANCE_MODES = ['analytic', 'stochastic']


def _input_int(psim, label: str, value: int, *, vmin: int | None = None,
               vmax: int | None = None) -> tuple[bool, int]:
    """Type-in integer field (``step=0`` hides ImGui's +/- steppers)."""
    changed, v = psim.InputInt(label, int(value), 0, 0)
    v = int(v)
    if vmin is not None:
        v = max(int(vmin), v)
    if vmax is not None:
        v = min(int(vmax), v)
    return v != int(value) or changed, v


def _input_float(psim, label: str, value: float, *, vmin: float | None = None,
                vmax: float | None = None, fmt: str = '%.4g') -> tuple[bool, float]:
    """Type-in float field (``step=0`` hides ImGui's +/- steppers)."""
    changed, v = psim.InputFloat(label, float(value), 0.0, 0.0, fmt)
    v = float(v)
    if vmin is not None:
        v = max(float(vmin), v)
    if vmax is not None:
        v = min(float(vmax), v)
    return abs(v - float(value)) > 1e-12 or changed, v


# ---------------------------------------------------------------------------
# Asset state
# ---------------------------------------------------------------------------

class AssetState:
    """Numpy pyramid levels plus frequency-band metadata for the slider."""

    def __init__(self, asset_path: str, *, asset_pyr: Optional[int],
                 opacity_scale: float, albedo_scale: float,
                 extent: float, world_scale: float, use_asset_albedo: bool,
                 gaussian_only: Optional[bool]):
        asset_root = Path(asset_path)
        asset_dir = P.resolve_asset_path(asset_path, asset_pyr=asset_pyr)
        self.pyramid_meta = A.load_pyramid_metadata(asset_root)
        gabor_target = self.pyramid_meta['gabor_target_level']
        gauss_target = self.pyramid_meta['gauss_target_level']
        self.is_steerable = is_steerable_asset(asset_dir)
        # ``detect_asset_info`` requires the pyramid layout
        # (``data/root.primitives_pyr*.ply``); a steerable-only asset instead
        # stores a single ``data/root.primitives_steerable.ply`` and has no
        # per-level files. Load it through ``steerable_info`` alone
        # (``self.levels`` stays empty, which ``centers_all`` /
        # ``build_scene`` already handle).
        try:
            self.info = detect_asset_info(str(asset_dir))
        except (ValueError, FileNotFoundError):
            if not self.is_steerable:
                raise
            self.info = None
        self.steerable_info = None
        if self.is_steerable:
            self.steerable_info = load_steerable_asset(
                str(asset_dir),
                opacity_scale=1.0,
                albedo_scale=albedo_scale if use_asset_albedo else 1.0,
                default_extent=extent,
                world_scale=world_scale,
            )
        self.gaussian_only = (
            gaussian_only if gaussian_only is not None
            else 'gaussian' in asset_root.as_posix().lower())

        self.levels: list[dict[str, Any]] = []
        self.meta: list[dict[str, Any]] = []
        for level in range(self.info.num_levels if self.info is not None else 0):
            data = A.load_level_numpy(
                self.info, level,
                albedo_scale=albedo_scale, opacity_scale=opacity_scale,
                extent=extent, world_scale=world_scale,
                use_asset_albedo=use_asset_albedo)
            # Gabor-ness is carried by omega: to view a band as Gaussian
            # (gaussian_only mode, or an LOD policy that maps this band to the
            # Gaussian target) we zero its omega up front, then everything
            # downstream derives it via is_gabor_from_omega.
            render_as_gabor = (not self.gaussian_only) and A.is_gabor_for_level(
                level, gabor_target, gauss_target,
                num_levels=self.info.num_levels,
                skip_gabor_optim=self.pyramid_meta['skip_gabor_optim'],
                skip_gaussian_optim=self.pyramid_meta['skip_gaussian_optim'])
            if not render_as_gabor:
                data = {**data, 'omega': np.zeros_like(
                    np.asarray(data['omega'], dtype=np.float32))}
            self.levels.append(data)
            self.meta.append({
                'is_gabor': A.is_gabor_from_omega(data['omega']),
                'visibility_mask': 1 << min(level, 7),
            })

        # Frequency bands of the gabor levels -> slider LUT
        bands = {}
        f_max = 0.0
        for i, (data, meta) in enumerate(zip(self.levels, self.meta)):
            if not meta['is_gabor']:
                continue
            f_lo, f_hi = L.gabor_frequency_band(
                data['omega'].reshape(-1), data['scales'])
            bands[i] = (f_lo, f_hi)
            if len(f_hi):
                f_max = max(f_max, float(f_hi.max()))
        self.has_gabors = bool(bands)
        self.freq_max = f_max if f_max > 0 else 1.0
        self.lut_fractions, self.lut_thresholds = L.build_freq_threshold_lut(
            bands, self.freq_max)
        # True when the rendered scene will contain Gabor primitives
        # (omega != 0). Gates the Gaussian-only gfields_prb_stochastic mode,
        # whose free-flight inversion / delta tracking is biased for signed
        # Gabor densities (steerable sub-bands above frequency level 0 are
        # Gabor by construction).
        self.contains_gabors = self.has_gabors or (
            self.steerable_info is not None
            and self.steerable_info.num_frequency_levels > 1)

    def threshold_for_fraction(self, fraction: float) -> float:
        return float(np.interp(fraction, self.lut_fractions, self.lut_thresholds))

    def centers_all(self) -> np.ndarray:
        if self.levels:
            return np.concatenate([d['centers'] for d in self.levels], axis=0)
        if self.steerable_info is None:
            return np.zeros((0, 3), dtype=np.float64)
        chunks = []
        for key in sorted(self.steerable_info.primitives_dict.keys()):
            centers = self.steerable_info.primitives_dict[key]['centers']
            chunks.append(np.array(centers, dtype=np.float64).reshape(-1, 3))
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))

    def opacity_base_levels(self) -> list[dict[str, Any]]:
        """Per-mesh unscaled opacities (sorted mesh order matches ``RenderSession``)."""
        if self.levels:
            return self.levels
        if self.steerable_info is None:
            return []
        prims = self.steerable_info.primitives_dict
        return [{'opacities': np.array(prims[key]['opacities']).reshape(-1)}
                for key in sorted(prims.keys())]

    def albedo_base_levels(self) -> list[dict[str, Any]]:
        """Per-mesh unscaled albedo (sorted mesh order matches ``RenderSession``)."""
        if self.levels:
            return [{'albedo': d['albedo']} for d in self.levels]
        if self.steerable_info is None:
            return []
        prims = self.steerable_info.primitives_dict
        return [{'albedo': np.array(prims[key]['albedo']).reshape(-1, 3)}
                for key in sorted(prims.keys())]


def _default_ortho_scale(levels: list[dict[str, Any]]) -> float:
    """Half-extent of the orthographic film in world units (training uses ~0.6–1.3)."""
    aabb = A.compute_visual_aabb_from_levels(levels)
    return max(0.15, 0.45 * float(np.max(aabb['size'])))


_ORTHO_SCALE_MIN = 0.05
_ORTHO_SCALE_MAX = 20.0


def _clip_ortho_scale(scale: float) -> float:
    return float(np.clip(scale, _ORTHO_SCALE_MIN, _ORTHO_SCALE_MAX))


def _sync_polyscope_projection(ps, ui, state) -> None:
    """Match polyscope navigation to the Mitsuba sensor projection."""
    mode = 'orthographic' if ui.camera_mode_idx == 0 else 'perspective'
    if state.get('polyscope_projection') == mode:
        return
    ps.set_view_projection_mode(mode)
    state['polyscope_projection'] = mode
    state['polyscope_vert_fov'] = float(ps.get_vertical_fov_degrees())
    state['ortho_scale_track'] = float(ui.ortho_scale)
    state['ortho_slider_track'] = float(ui.ortho_scale)


def _bind_ortho_scroll_to_scale(ps, ui, state, *,
                                 ortho_slider_changed: bool) -> bool:
    """In orthographic mode, route mouse-wheel zoom to ``ui.ortho_scale``.

   Polyscope's orthographic projection adjusts vertical FOV on scroll

    Returns True when the ortho scale changed (resets accumulation).
    """
    if ui.camera_mode_idx != 0:
        return False

    fov = float(ps.get_vertical_fov_degrees())
    prev_fov = float(state.get('polyscope_vert_fov', fov))
    prev_scale = float(state.get('ortho_scale_track', ui.ortho_scale))
    prev_slider = float(state.get('ortho_slider_track', ui.ortho_scale))
    changed = False

    if ortho_slider_changed and abs(ui.ortho_scale - prev_slider) > 1e-8:
        ui.ortho_scale = _clip_ortho_scale(ui.ortho_scale)
        if prev_scale > 1e-8 and prev_fov > 1e-6:
            fov = float(np.clip(
                prev_fov * (ui.ortho_scale / prev_scale), 5.0, 120.0))
            ps.set_vertical_fov_degrees(fov)
        prev_scale = ui.ortho_scale
        changed = True
    elif abs(fov - prev_fov) > 1e-4 and prev_fov > 1e-6:
        ui.ortho_scale = _clip_ortho_scale(prev_scale * (fov / prev_fov))
        prev_scale = ui.ortho_scale
        changed = True

    state['polyscope_vert_fov'] = fov
    state['ortho_scale_track'] = prev_scale
    state['ortho_slider_track'] = float(ui.ortho_scale)
    return changed


def tomography_accumulates_optical_depth(ui) -> bool:
    """True when tomography must average optical depth before exponentiating."""
    if ui.integrator_idx != 0:
        return False
    return buffer_is_linear_tau(integrator=build_integrator_dict(ui))


def tomography_radiance_for_display(
    image: np.ndarray,
    ui,
    *,
    integrator_props: dict[str, Any] | None = None,
    envmap_path: Optional[str] = None,
    accumulator_is_tau: bool = False,
) -> np.ndarray:
    """Convert a raw / mean optical-depth buffer to radiance for tonemapping."""
    if ui.integrator_idx != 0:
        return np.asarray(image, dtype=np.float64)[..., :3]
    emitter_scale = ui.env_scale if envmap_path else 1.0
    if accumulator_is_tau:
        return tau_to_radiance(image, emitter_scale=emitter_scale)
    props = integrator_props or build_integrator_dict(ui)
    return interpret_tomography_buffer(
        image,
        criteria=props.get('stochastic_selection_criteria', 'deterministic'),
        orientation_selection=props.get('orientation_selection', 'deterministic'),
        steerable=props.get('asset_encoding') == 'steerable',
        emitter_scale=emitter_scale,
        integrator=props,
    )


def viewer_tau_accumulation_config(ui, envmap_path: Optional[str]) -> dict[str, Any]:
    """RenderSession optical-depth accumulation settings for tomography."""
    if ui.integrator_idx != 0:
        return {
            'accumulate_tau': False,
            'linear_tau_samples': True,
            'emitter_scale': 1.0,
        }
    props = build_integrator_dict(ui)
    linear = buffer_is_linear_tau(integrator=props)
    needs_tau = needs_mean_tau_accumulation(
        criteria=props.get('stochastic_selection_criteria', 'deterministic'),
        orientation_selection=props.get('orientation_selection', 'deterministic'),
        steerable=props.get('asset_encoding') == 'steerable',
    )
    return {
        'accumulate_tau': linear or needs_tau,
        'linear_tau_samples': linear,
        'emitter_scale': ui.env_scale if envmap_path else 1.0,
    }


def build_integrator_dict(ui) -> dict[str, Any]:
    # gfields_prb_stochastic is Gaussian-only (it raises on Gabor kernels) and
    # never consults the LOD selector (single-level min-of-free-flights
    # sampler), so pin its kernel and criteria here; the UI hides those
    # controls in that mode.
    stochastic_sampler = ui.integrator_idx == 2
    common: dict[str, Any] = {
        'kernel_type': ('gaussian' if stochastic_sampler
                        else KERNEL_TYPES[ui.kernel_idx]),
        'kernel_normalized': ui.kernel_normalized,
        'kernel_full_range': ui.kernel_full_range,
        'pyramid_level_count': ui.level_count,
        'stochastic_selection_criteria': (
            'deterministic' if stochastic_sampler
            else ui.primary_criteria()[ui.criteria_idx]),
        'power_law_beta': ui.power_law_beta,
    }
    # Training and the integrator default use freq_threshold=-1 (filter off).
    # Only compile the continuous LOD path when the user lowers the slider.
    if ui.freq_filter_active:
        common['freq_threshold'] = ui.freq_threshold
        common['freq_band_sigma'] = ui.freq_band_sigma
    if ui.is_steerable and ui.asset_encoding_steerable:
        common.update({
            'asset_encoding': 'steerable',
            'num_orientations': ui.num_orientations,
            'orientation_selection': ui.orientation_criteria[ui.orientation_idx],
            'orientation_threshold': ui.orientation_threshold,
            'orientation_decay_rate': ui.orientation_decay_rate,
            'importance_mix_weight': ui.importance_mix_weight,
        })
        if ui.integrator_idx == 1:  # NEE exists only on the prb path
            common['orientation_selection_nee'] = (
                ui.orientation_criteria[ui.orientation_nee_idx])

    if ui.integrator_idx == 0:  # tomography
        d = tomography_integrator_props(
            {
                'type': 'gfields_tomography',
                'max_depth': ui.max_depth,
                # Internal @dr.freeze (records cleanly, ~2.4x replay); volprim_prb
                # has no equivalent, so this is tomography-only.
                'use_kernel_freeze': ui.use_kernel_freeze,
                'use_anyhit': ui.use_anyhit,
                **common,
            },
            criteria=ui.primary_criteria()[ui.criteria_idx],
            orientation_selection=(
                ui.orientation_criteria[ui.orientation_idx]
                if ui.is_steerable and ui.asset_encoding_steerable
                else 'deterministic'),
            steerable=ui.is_steerable and ui.asset_encoding_steerable,
        )
        return d

    d: dict[str, Any] = {
        'type': ('gfields_prb_stochastic' if stochastic_sampler
                 else 'gfields_prb'),
        'max_depth': ui.max_depth,
        'rr_depth': ui.rr_depth,
        'rr_nee_depth': ui.rr_nee_depth,
        'use_nee': ui.use_nee,
        'use_indirect': ui.use_indirect,
        'max_depth_primitive': ui.max_depth_primitive,
        'rr_depth_primitive': ui.rr_depth_primitive,
        'max_overlaps': ui.max_overlaps,
        'solver_type': SOLVER_TYPES[ui.solver_idx],
        'solver_max_iterations': ui.solver_max_iterations,
        'num_bounces_gaussian_ms': ui.num_bounces_gaussian_ms,
        'stochastic_selection_criteria_nee': (
            'deterministic' if stochastic_sampler
            else ui.nee_criteria[ui.criteria_nee_idx]),
        'power_law_beta_nee': ui.power_law_beta_nee,
        'nee_truncate_at_shell': ui.nee_truncate_at_shell,
        'phasefunction': ({'type': 'hg', 'g': ui.hg_g} if ui.phase_idx == 1
                          else {'type': 'isotropic'}),
        **common,
    }
    if stochastic_sampler:
        d['nee_transmittance'] = NEE_TRANSMITTANCE_MODES[ui.nee_transmittance_idx]
        d['use_anyhit'] = ui.use_anyhit_nee
        d['use_anyhit_sampling'] = ui.use_anyhit_sampling
        d['residual_weight_clamp'] = ui.residual_weight_clamp
        d['truncate_at_shell'] = ui.truncate_at_shell
    return d


def build_emitter_dict(ui, envmap_path: Optional[str]) -> dict[str, Any]:
    if envmap_path:
        return {'type': 'envmap', 'filename': envmap_path,
                'scale': ui.env_scale, 'to_world': env_to_world(ui)}
    # Match training: unit constant emitter (implicit radiance = 1).
    return {'type': 'constant'}


def env_to_world(ui):
    T = mi.ScalarTransform4f
    return T().rotate([0, 1, 0], ui.env_azimuth) @ T().rotate([1, 0, 0], ui.env_elevation)


def _deterministic_criteria_idx(criteria: list[str]) -> int:
    try:
        return criteria.index('deterministic')
    except ValueError:
        return 0


def _apply_ms_integrator_preset(ui) -> None:
    ui.max_depth = MS_INTEGRATOR_DEFAULTS['max_depth']
    ui.rr_depth = MS_INTEGRATOR_DEFAULTS['rr_depth']
    ui.rr_nee_depth = MS_INTEGRATOR_DEFAULTS['rr_nee_depth']
    ui.max_overlaps = MS_INTEGRATOR_DEFAULTS['max_overlaps']
    # Multiple scattering integrates kernel transmittance over the SEGMENT
    # between scatter events (sample_segment / eval_transmittance pass real
    # seg_t0,seg_t1 / 0,maxt bounds). full_range=True makes density_integral
    # ignore those bounds and return the full infinite-line integral, which
    # overcounts every segment (up to ~3.6x for narrow segments) -> density is
    # massively overestimated. Tomography (single-pass) genuinely wants the full
    # integral; MS must not.
    ui.kernel_full_range = False
    ui.solver_idx = SOLVER_TYPES.index(
        MS_INTEGRATOR_DEFAULTS['solver_type'])
    ui.solver_max_iterations = MS_INTEGRATOR_DEFAULTS['solver_max_iterations']
    ui.criteria_idx = _deterministic_criteria_idx(ui.primary_criteria())
    ui.criteria_nee_idx = _deterministic_criteria_idx(ui.nee_criteria)
    ui.albedo_scale = MS_ALBEDO_DEFAULT


def _apply_tomography_integrator_preset(ui) -> None:
    ui.max_depth = TOMOGRAPHY_MAX_DEPTH_DEFAULT
    # Tomography integrates each kernel's full contribution along the ray in one
    # pass -> the full-line integral is the intended quantity.
    ui.kernel_full_range = True
    # The tomography criteria list is shorter than the prb one (no gaussian_ms),
    # so reset the selection to keep the index from dangling past the new list.
    ui.criteria_idx = _deterministic_criteria_idx(ui.primary_criteria())


def _scaled_level_data(data: dict[str, Any], opacity_weight: float,
                       albedo_scale: float) -> dict[str, Any]:
    """Return a shallow copy of *data* with opacities/albedo scaled."""
    if abs(opacity_weight - 1.0) < 1e-12 and abs(albedo_scale - 1.0) < 1e-12:
        return data
    out = dict(data)
    if abs(opacity_weight - 1.0) >= 1e-12:
        out['opacities'] = np.ascontiguousarray(
            data['opacities'] * opacity_weight, dtype=np.float32)
    if abs(albedo_scale - 1.0) >= 1e-12:
        out['albedo'] = np.ascontiguousarray(
            data['albedo'] * albedo_scale, dtype=np.float32)
    return out


def _level_with_extents(data: dict[str, Any], meta: dict[str, Any],
                        ui) -> dict[str, Any]:
    """Apply fixed or adaptive per-primitive bounding extents."""
    data = dict(data)
    n = int(data['centers'].shape[0])
    if ui.adaptive_extent:
        ext = clamp_extents(
            data['scales'],
            omega=data['omega'] if meta['is_gabor'] else None,
            opacities=data['opacities'],
            base_extent=ui.base_extent,
            min_extent=ui.min_extent,
            threshold=ui.min_contribution,
            envelope_only=not meta['is_gabor'],
        )
        data['extent'] = np.ascontiguousarray(ext.reshape(-1, 1), dtype=np.float32)
    elif np.asarray(data['extent']).reshape(-1).size == n:
        pass
    else:
        data['extent'] = np.full((n, 1), float(ui.base_extent), dtype=np.float32)
    return data


def _prune_level_by_band(data: dict[str, Any], freq_threshold: float,
                         band_sigma: float, drop_eps: float = 0.02) -> dict[str, Any]:
    """Hard-cull gabor primitives whose `continuous_lod_weight` has dropped below
    `drop_eps` so they are removed from the scene/BVH (never traced) instead of
    merely scaled to ~0. The smooth weight still covers the transition band
    (kept primitives near the cutoff).

    The weight is the same smoothstep on the analytic Gabor band
    `[f_lo, f_hi] = ‖β/s‖ ∓ band_sigma·‖1/s‖` used by `continuous_lod_weight`.
    Pruning on `f_lo > threshold` alone never triggers — the Gaussian envelope's
    spectrum always reaches DC (`f_lo` clamps to 0) — so we threshold the weight.
    Returns a filtered shallow copy; per-primitive arrays are sliced.
    """
    if freq_threshold < 0.0:
        return data
    omega = np.asarray(data["omega"], dtype=np.float64).reshape(-1)
    scales = np.asarray(data["scales"], dtype=np.float64)
    f_lo, f_hi = L.gabor_frequency_band(omega, scales, band_sigma=band_sigma)
    t = np.clip((float(freq_threshold) - f_lo) / np.maximum(f_hi - f_lo, 1e-12), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)            # smoothstep, matches continuous_lod_weight
    keep = w > drop_eps
    n = keep.shape[0]
    if keep.all():
        return data
    out: dict[str, Any] = {}
    for k, v in data.items():
        arr = np.asarray(v)
        out[k] = arr[keep] if (arr.ndim >= 1 and arr.shape[0] == n) else v
    return out


def build_scene(asset: AssetState, ui, envmap_path: Optional[str]) -> mi.Scene:
    scene: dict[str, Any] = {
        'type': 'scene',
        'integrator': build_integrator_dict(ui),
        'environment': build_emitter_dict(ui, envmap_path),
    }
    level_cap = min(int(ui.level_count), 8)
    if asset.steerable_info is not None and ui.asset_encoding_steerable:
        for key, prim in asset.steerable_info.primitives_dict.items():
            freq_level = int(key.split('_')[0][1:])
            if freq_level >= level_cap:
                continue
            scene[f'primitives_{key}'] = dict(prim)
        return mi.load_dict(scene)

    mesh_idx = 0
    for data, meta in zip(asset.levels[:level_cap], asset.meta[:level_cap]):
        prepared = _level_with_extents(
            _scaled_level_data(data, ui.opacity_weight, 1.0), meta, ui)
        # Continuous-LOD hard prune: when the frequency filter is active, drop
        # gabor kernels whose band is fully above the cutoff so they cost nothing
        # to trace (the smooth weight covers the transition band). This is what
        # turns the band-limit slider into an actual speedup.
        if ui.freq_filter_active and meta['is_gabor']:
            prepared = _prune_level_by_band(
                prepared, ui.freq_threshold, ui.freq_band_sigma)
        if int(np.asarray(prepared['centers']).shape[0]) == 0:
            continue  # whole level filtered out
        scene[f'primitives_{mesh_idx}'] = A.numpy_level_to_mesh_entry(
            prepared, visibility_mask=meta['visibility_mask'])
        mesh_idx += 1
    return mi.load_dict(scene)


# ---------------------------------------------------------------------------
# UI state
# ---------------------------------------------------------------------------

class UIState:
    def __init__(self, asset: AssetState, args,
                 tomography_criteria, prb_criteria, orientation_criteria):
        self.tomography_criteria = list(tomography_criteria)
        self.prb_criteria = list(prb_criteria)
        from gfields.lod import STOCHASTIC_CRITERIA
        # NEE list is path-tracing only; includes ``biased_zero``, excludes ``gaussian_ms``.
        self.nee_criteria = [c for c in STOCHASTIC_CRITERIA if c != 'gaussian_ms']
        self.orientation_criteria = list(orientation_criteria)
        self.is_steerable = asset.is_steerable

        # Integrator / level-of-detail (0 = tomography, 1 = prb,
        # 2 = prb stochastic; INTEGRATOR_MODES)
        self.integrator_idx = {'tomography': 0, 'prb': 1,
                               'prb_stochastic': 2}.get(args.integrator, 1)
        # ``--freeze`` drives the tomography integrator's *internal* @dr.freeze
        # (use_kernel_freeze): it passes the integrator as traversable state via
        # state_fn so it records cleanly. Frozen replay is ~2.4x faster per frame;
        # scene params (opacity/albedo) and freq_threshold still propagate on replay.
        self.use_kernel_freeze = bool(getattr(args, 'freeze', False))
        # Any-hit fast paths default ON everywhere an integrator supports one:
        # tomography's primal-pass traversal (falls back to the sequential hit
        # walk on unsupported scene configs) and prb_stochastic's NEE any-hit.
        # ``--no-use_anyhit`` disables both at launch; each stays toggleable in
        # the Integrator panel.
        self.use_anyhit = bool(getattr(args, 'use_anyhit', True))
        self.use_anyhit_nee = self.use_anyhit
        # Distance-sampling any-hit stays opt-in (validated but currently
        # slower than the hit walk -- prb_stochastic.md §5); the clamp caps the
        # envelope delta-tracking null-weight amplification.
        self.use_anyhit_sampling = False
        self.residual_weight_clamp = 1.0
        # prb_stochastic shell-truncation conventions (docs/prb_stochastic.md):
        # kernels are unbounded by default (shells are BVH discovery proxies);
        # truncate_at_shell reproduces gfields_prb's shell-truncated transport,
        # nee_truncate_at_shell the shadow-ray origin-overlap convention
        # (honored by both prb integrators).
        self.truncate_at_shell = False
        self.nee_truncate_at_shell = False
        if asset.steerable_info is not None:
            self.level_count = max(1, min(
                8, asset.steerable_info.num_frequency_levels))
        else:
            self.level_count = max(1, min(8, asset.info.num_levels))
        self.criteria_idx = 0
        self.criteria_nee_idx = 0
        self.power_law_beta = 0.75
        self.power_law_beta_nee = 0.75
        self.num_bounces_gaussian_ms = 0

        # Path tracing (prb) — initialize every field rebuild_key() reads, then
        # apply tomography-specific overrides when launching in that mode.
        _apply_ms_integrator_preset(self)
        if args.integrator == 'tomography':
            self.max_depth = TOMOGRAPHY_MAX_DEPTH_DEFAULT
            self.albedo_scale = 0.0
        self.use_nee = True
        self.use_indirect = True
        self.max_depth_primitive = 256
        self.rr_depth_primitive = 256
        self.phase_idx = 0
        self.hg_g = 0.857
        # prb stochastic only; 0 = 'analytic' (plugin default, best at equal
        # variance).
        self.nee_transmittance_idx = 0

        # Kernel (match training: normalized + full-range tomography).
        # full_range is mode-dependent: tomography integrates each kernel fully
        # in one pass (True); MS needs per-segment transmittance (False). The
        # integrator presets above already set this, but pin the startup default
        # to the launched mode so MS does not start with the overcounting path.
        self.kernel_idx = 0
        self.kernel_normalized = True
        self.kernel_full_range = (self.integrator_idx == 0)

        # Opacity weight (base opacities are stored unscaled in the asset)
        self.opacity_weight = float(args.opacity_scale)

        # Continuous frequency-domain LOD (off by default; matches integrator default)
        self.freq_fraction = 1.0
        self.freq_band_sigma = 2.0
        self.freq_threshold = -1.0

        # Steerable orientation selectivity (auto-enabled for steerable assets)
        self.asset_encoding_steerable = asset.is_steerable
        self.num_orientations = (
            asset.steerable_info.num_orientations
            if asset.steerable_info is not None else 3)
        self.orientation_idx = 0
        self.orientation_nee_idx = 0
        self.orientation_threshold = 0.7
        self.orientation_decay_rate = 1.5
        self.importance_mix_weight = 0.5

        # Environment
        self.env_azimuth = 0.0
        self.env_elevation = 0.0
        self.env_scale = 1.0

        # Display / budget
        self.exposure = 0.0
        self.max_spp = args.max_spp
        self.resolution = int(args.resolution)
        self.preview_resolution = int(args.preview_resolution)
        self.still_frames_threshold = 12

        # Camera (orthographic default; scale set from asset bounds, not polyscope fov)
        self.camera_mode_idx = 0  # 0 = orthographic, 1 = perspective
        self.perspective_fov = 45.0
        if getattr(args, 'ortho_scale', None) is not None:
            self.ortho_scale = float(args.ortho_scale)
        else:
            if asset.levels:
                levels_for_scale = asset.levels
            elif asset.steerable_info is not None:
                # Steerable assets keep no simple ``levels``; build the AABB
                # input from the per-sub-band primitive dicts (centers+scales).
                levels_for_scale = [
                    {'centers': np.array(p['centers'], dtype=np.float64).reshape(-1, 3),
                     'scales': np.array(p['scales'], dtype=np.float64).reshape(-1, 3)}
                    for p in asset.steerable_info.primitives_dict.values()]
            else:
                levels_for_scale = [{'centers': asset.centers_all()}]
            self.ortho_scale = _default_ortho_scale(levels_for_scale)

        # Adaptive per-primitive extents
        self.adaptive_extent = False
        self.min_contribution = 1e-8
        self.min_extent = 0.1
        self.base_extent = float(args.kernel_extent)

    @property
    def freq_filter_active(self) -> bool:
        """True when the continuous frequency filter is enabled (slider < 1)."""
        return self.freq_fraction < 1.0 - 1e-6

    def primary_criteria(self) -> list[str]:
        """Stochastic LOD criteria for the active integrator mode."""
        if self.integrator_idx == 0:
            return self.tomography_criteria
        return self.prb_criteria

    def rebuild_key(self):
        """Anything requiring an integrator/scene rebuild (constructor props)."""
        return (self.integrator_idx, self.criteria_idx, self.criteria_nee_idx,
                round(self.power_law_beta, 4), round(self.power_law_beta_nee, 4),
                self.num_bounces_gaussian_ms,
                self.max_depth, self.rr_depth, self.rr_nee_depth, self.use_nee,
                self.use_indirect, self.max_depth_primitive,
                self.rr_depth_primitive, self.max_overlaps, self.solver_idx,
                self.solver_max_iterations, self.phase_idx, round(self.hg_g, 4),
                self.nee_transmittance_idx,
                self.use_anyhit, self.use_anyhit_nee, self.use_anyhit_sampling,
                round(self.residual_weight_clamp, 4),
                self.truncate_at_shell, self.nee_truncate_at_shell,
                self.kernel_idx, self.kernel_normalized, self.kernel_full_range,
                self.freq_filter_active, round(self.freq_band_sigma, 4),
                # Quantized freq so the hard prune (build_scene) re-applies as the
                # slider sweeps, while fine moves within a bin stay on the smooth
                # runtime weight (no rebuild).
                round(self.freq_fraction, 2),
                self.level_count,
                self.asset_encoding_steerable, self.num_orientations,
                self.orientation_idx, self.orientation_nee_idx,
                round(self.orientation_threshold, 4),
                round(self.orientation_decay_rate, 4),
                round(self.importance_mix_weight, 4),
                self.adaptive_extent, round(self.min_contribution, 8),
                round(self.min_extent, 4), round(self.base_extent, 4))

    def runtime_key(self):
        """Anything handled through mi.traverse (no rebuild)."""
        key: list[Any] = [
            round(self.opacity_weight, 6),
            round(self.albedo_scale, 6),
            round(self.env_azimuth, 2),
            round(self.env_elevation, 2),
            round(self.env_scale, 4),
            self.max_spp,
            self.camera_mode_idx,
            round(self.perspective_fov, 2),
            round(self.ortho_scale, 5),
        ]
        if self.freq_filter_active:
            key.insert(1, round(self.freq_threshold, 5))
        return tuple(key)


# ---------------------------------------------------------------------------
# Viewer main
# ---------------------------------------------------------------------------

def run_viewer(args) -> None:
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ModuleNotFoundError as e:
        raise SystemExit(
            "The interactive viewer needs polyscope, which is not installed.\n"
            "  Install it with:  uv pip install -e \".[viewer]\"   (or: pip install polyscope)\n"
            "  (re-running install.sh / install.ps1 also pulls it in.)"
        ) from e

    if mi.variant() is None:
        mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
    import gfields  # noqa: F401  (registers the unified integrators)
    from gfields.lod import ORIENTATION_CRITERIA, STOCHASTIC_CRITERIA

    print(f'Loading asset: {args.asset}')
    asset = AssetState(
        args.asset, asset_pyr=args.asset_pyr,
        opacity_scale=1.0, albedo_scale=1.0,
        extent=args.kernel_extent, world_scale=args.world_scale,
        use_asset_albedo=not args.ignore_asset_albedo,
        gaussian_only=args.gaussian_only)
    n_prims = int(asset.centers_all().shape[0])
    n_levels = (
        len(asset.levels)
        if asset.levels
        else (asset.steerable_info.num_frequency_levels
              if asset.steerable_info is not None else 0))
    print(f'  {n_levels} levels, {n_prims} primitives, '
          f'f_max={asset.freq_max:.1f}'
          + ('  [steerable]' if asset.is_steerable else ''))
    if asset.steerable_info is not None:
        print(f'    steerable sub-bands: {asset.steerable_info.total_subbands} '
              f'({asset.steerable_info.num_orientations} orientations)')
    for i, (data, meta) in enumerate(zip(asset.levels, asset.meta)):
        kind = 'Gabor' if meta['is_gabor'] else 'Gaussian'
        print(f'    level {i}: {data["centers"].shape[0]} primitives ({kind})')

    prb_criteria = [c for c in STOCHASTIC_CRITERIA if c != 'biased_zero']
    ui = UIState(asset, args, tomography_criteria_choices(STOCHASTIC_CRITERIA),
                 prb_criteria, ORIENTATION_CRITERIA)
    if args.integrator == 'prb' and args.albedo_scale != MS_ALBEDO_DEFAULT:
        ui.albedo_scale = float(args.albedo_scale)
    if ui.integrator_idx == 2 and asset.contains_gabors:
        print('[viewer] --integrator prb_stochastic is Gaussian-only and this '
              'asset contains Gabor kernels (its distance sampler is biased '
              'for signed Gabor densities, see docs/prb_stochastic.md); '
              'falling back to prb.')
        ui.integrator_idx = 1
    print(f'  ortho scale (default): {ui.ortho_scale:.4g}')
    out_dir = Path(args.output)

    state = {
        'scene': None, 'session': None, 'rebuild_key': None,
        'prev_cam_key': None, 'still_frames': 0, 'camera_moving': True,
        'polyscope_projection': None, 'prev_integrator_idx': ui.integrator_idx,
    }

    def active_envmap() -> Optional[str]:
        """Env-map path for the *current* integrator mode.

        An explicitly passed ``--envmap`` applies to every mode; the launch
        default only activates in the multiple-scattering modes -- tomography
        then uses the constant emitter (radiance 1.0), matching training. The
        emitter is part of the scene, so mode switches pick this up through
        the integrator rebuild.
        """
        if not args.envmap:
            return None
        if getattr(args, 'envmap_explicit', True):
            return args.envmap
        return args.envmap if ui.integrator_idx >= 1 else None

    def rebuild():
        """Rebuild the scene/session; on failure keep the previous one so a bad
        parameter combination cannot crash the interactive loop."""
        try:
            envmap_path = active_envmap()
            scene = build_scene(asset, ui, envmap_path)
            tau_cfg = viewer_tau_accumulation_config(ui, envmap_path)
            # Freeze is driven by the integrator's internal use_kernel_freeze
            # (see UIState.use_kernel_freeze), so keep the outer
            # dr.freeze(mi.render) path off.
            session = RenderSession(
                scene, max_spp=ui.max_spp, try_freeze=False,
                **tau_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f'[viewer] scene rebuild failed ({type(exc).__name__}: {exc}); '
                  f'keeping previous settings.')
            state['rebuild_key'] = ui.rebuild_key()  # accept so we do not spin
            return
        state['scene'] = scene
        state['session'] = session
        state['rebuild_key'] = ui.rebuild_key()
        session.seed_opacity_weight(ui.opacity_weight, asset.opacity_base_levels())
        session.seed_albedo_scale(1.0, asset.albedo_base_levels())
        session.set_albedo_scale(ui.albedo_scale, asset.albedo_base_levels(),
                                 absolute=(ui.integrator_idx >= 1))

    rebuild()
    if state['session'] is None:
        raise RuntimeError('initial scene build failed; see message above')

    ps.set_program_name('gfields viewer')
    ps.init()
    ps.set_up_dir('y_up')
    ps.set_ground_plane_mode('none')

    # Register the primitive centers so polyscope has something to frame
    # and orbit around (display itself is the fullscreen render).
    cloud = ps.register_point_cloud('primitives', asset.centers_all(),
                                    point_render_mode='quad')
    cloud.set_radius(0.0005, relative=True)
    cloud.set_transparency(0.15)
    ps.reset_camera_to_home_view()
    if ui.camera_mode_idx == 0:
        ps.set_view_projection_mode('orthographic')
        state['polyscope_projection'] = 'orthographic'
        state['polyscope_vert_fov'] = float(ps.get_vertical_fov_degrees())
        state['ortho_scale_track'] = float(ui.ortho_scale)
        state['ortho_slider_track'] = float(ui.ortho_scale)

    def viewer_callback():
        session = state['session']
        ortho_slider_changed = False

        # ---------------------------------------------------------- panel
        psim.PushItemWidth(220)
        psim.TextUnformatted(
            f'spp {session.spp_done}/{ui.max_spp}'
            + ('  [converged]' if session.converged else ''))
        psim.TextUnformatted(
            f'cost/sample {session.ms_per_sample:6.2f} ms'
            f'   (frame {session.last_exec_ms:6.2f} ms)')
        eff_res = (ui.preview_resolution if state['camera_moving']
                   else ui.resolution)
        psim.TextUnformatted(
            f'render {eff_res}x{eff_res}'
            + ('  [preview]' if state['camera_moving'] else ''))

        if psim.TreeNodeEx('Camera', psim.ImGuiTreeNodeFlags_DefaultOpen):
            _, ui.camera_mode_idx = psim.Combo(
                'projection', ui.camera_mode_idx, ['orthographic', 'perspective'])
            if ui.camera_mode_idx == 0:
                ortho_slider_changed, ui.ortho_scale = _input_float(
                    psim, 'ortho scale', ui.ortho_scale,
                    vmin=_ORTHO_SCALE_MIN, vmax=_ORTHO_SCALE_MAX, fmt='%.4g')
                psim.TextUnformatted('scroll wheel adjusts ortho scale')
            if ui.camera_mode_idx == 1:
                _, ui.perspective_fov = _input_float(
                    psim, 'fov (deg)', ui.perspective_fov,
                    vmin=5.0, vmax=120.0, fmt='%.1f')
            _, ui.resolution = _input_int(
                psim, 'resolution (static)', ui.resolution, vmin=512, vmax=2048)
            _, ui.preview_resolution = _input_int(
                psim, 'resolution (moving)', ui.preview_resolution,
                vmin=128, vmax=1024)
            _, ui.still_frames_threshold = _input_int(
                psim, 'still frames', ui.still_frames_threshold, vmin=1, vmax=120)
            psim.TreePop()

        if psim.TreeNodeEx('Level of detail',
                           psim.ImGuiTreeNodeFlags_DefaultOpen):
            if asset.has_gabors:
                changed, ui.freq_fraction = psim.SliderFloat(
                    'frequency filter', ui.freq_fraction, 0.0, 1.0)
                if changed:
                    if ui.freq_filter_active:
                        ui.freq_threshold = asset.threshold_for_fraction(
                            ui.freq_fraction)
                    else:
                        ui.freq_threshold = -1.0
                if ui.freq_filter_active:
                    psim.TextUnformatted(f'cutoff: {ui.freq_threshold:.1f} '
                                         f'(0 = Gaussians only)')
                else:
                    psim.TextUnformatted('filter off (1.0 = full spectrum)')
                if ui.freq_filter_active:
                    _, ui.freq_band_sigma = psim.SliderFloat(
                        'band sigma', ui.freq_band_sigma, 0.5, 4.0)
            _, ui.level_count = _input_int(
                psim, 'pyramid levels', ui.level_count, vmin=1,
                vmax=max(1, min(8, (
                    asset.steerable_info.num_frequency_levels
                    if asset.steerable_info is not None and ui.asset_encoding_steerable
                    else (asset.info.num_levels if asset.info is not None else 8)))))
            if ui.integrator_idx == 2:
                # prb_stochastic never consults the LOD selector (single-level
                # Gaussian min-of-free-flights sampler); build_integrator_dict
                # pins the criteria to 'deterministic'.
                psim.TextUnformatted(
                    'prb stochastic: LOD criteria not used (single-level)')
            else:
                _, ui.criteria_idx = psim.Combo(
                    'criteria (primary)', ui.criteria_idx, ui.primary_criteria())
                crit = ui.primary_criteria()[ui.criteria_idx]
                if 'power_law' in crit:
                    _, ui.power_law_beta = _input_float(
                        psim, 'power-law beta', ui.power_law_beta,
                        vmin=0.05, vmax=0.95)
                if ui.integrator_idx == 1 and crit == 'gaussian_ms':
                    _, ui.num_bounces_gaussian_ms = _input_int(
                        psim, 'gaussian_ms bounces', ui.num_bounces_gaussian_ms,
                        vmin=0)
                if ui.integrator_idx == 1:
                    _, ui.criteria_nee_idx = psim.Combo(
                        'criteria (NEE)', ui.criteria_nee_idx, ui.nee_criteria)
                    if 'power_law' in ui.nee_criteria[ui.criteria_nee_idx]:
                        _, ui.power_law_beta_nee = _input_float(
                            psim, 'power-law beta (NEE)', ui.power_law_beta_nee,
                            vmin=0.05, vmax=0.95)
            psim.TreePop()

        if ui.is_steerable and psim.TreeNodeEx('Steerable orientations'):
            _, ui.asset_encoding_steerable = psim.Checkbox(
                'steerable encoding', ui.asset_encoding_steerable)
            if ui.asset_encoding_steerable:
                _, ui.num_orientations = _input_int(
                    psim, 'orientations', ui.num_orientations, vmin=1, vmax=8)
                _, ui.orientation_idx = psim.Combo(
                    'orientation selection', ui.orientation_idx,
                    ui.orientation_criteria)
                if ui.integrator_idx == 1:
                    _, ui.orientation_nee_idx = psim.Combo(
                        'orientation (NEE)', ui.orientation_nee_idx,
                        ui.orientation_criteria)
                _, ui.orientation_threshold = _input_float(
                    psim, 'orientation threshold', ui.orientation_threshold,
                    vmin=0.0, vmax=1.0)
                _, ui.orientation_decay_rate = _input_float(
                    psim, 'orientation decay', ui.orientation_decay_rate,
                    vmin=0.1, vmax=4.0)
                ori_modes = {ui.orientation_criteria[ui.orientation_idx]}
                if ui.integrator_idx == 1:
                    ori_modes.add(ui.orientation_criteria[ui.orientation_nee_idx])
                if ori_modes & {'importance', 'control_variate_threshold'}:
                    _, ui.importance_mix_weight = _input_float(
                        psim, 'importance mix', ui.importance_mix_weight,
                        vmin=0.0, vmax=1.0)
            psim.TreePop()

        if psim.TreeNodeEx('Integrator', psim.ImGuiTreeNodeFlags_DefaultOpen):
            _, ui.integrator_idx = psim.Combo(
                'mode', ui.integrator_idx, INTEGRATOR_MODES)
            if asset.contains_gabors and ui.integrator_idx == 2:
                # gfields_prb_stochastic is Gaussian-only: refuse the switch
                # (before the preset check below, so nothing else changes) and
                # raise the alarm text.
                prev = state['prev_integrator_idx']
                ui.integrator_idx = prev if prev != 2 else 1
                state['stochastic_denied'] = True
            if asset.contains_gabors:
                psim.TextColored(
                    (1.0, 0.35, 0.35, 1.0) if state.get('stochastic_denied')
                    else (0.6, 0.6, 0.6, 1.0),
                    'prb stochastic unavailable: asset has Gabor kernels')
            if ui.integrator_idx != state['prev_integrator_idx']:
                if ui.integrator_idx >= 1:
                    _apply_ms_integrator_preset(ui)
                else:
                    _apply_tomography_integrator_preset(ui)
                state['prev_integrator_idx'] = ui.integrator_idx
            if ui.integrator_idx == 0 and session.accumulate_tau:
                psim.TextUnformatted(
                    'progressive: mean optical depth (1 spp), exp(-τ) for display')
            if ui.integrator_idx == 0:
                _, ui.use_anyhit = psim.Checkbox(
                    'any-hit primal pass', ui.use_anyhit)
            _, ui.max_depth = _input_int(psim, 'max depth', ui.max_depth, vmin=-1)
            if ui.integrator_idx >= 1:
                _, ui.rr_depth = _input_int(psim, 'rr depth', ui.rr_depth, vmin=0)
                _, ui.rr_nee_depth = _input_int(
                    psim, 'rr depth (NEE)', ui.rr_nee_depth, vmin=0)
                _, ui.use_nee = psim.Checkbox('next-event estimation', ui.use_nee)
                _, ui.use_indirect = psim.Checkbox('indirect', ui.use_indirect)
                _, ui.max_depth_primitive = _input_int(
                    psim, 'max primitive depth', ui.max_depth_primitive, vmin=1)
                _, ui.rr_depth_primitive = _input_int(
                    psim, 'rr primitive depth', ui.rr_depth_primitive, vmin=1)
                _, ui.max_overlaps = _input_int(
                    psim, 'max overlaps', ui.max_overlaps, vmin=1)
                if ui.integrator_idx == 1:
                    _, ui.solver_idx = psim.Combo('solver', ui.solver_idx, SOLVER_TYPES)
                    _, ui.solver_max_iterations = _input_int(
                        psim, 'solver iterations', ui.solver_max_iterations, vmin=1)
                else:
                    _, ui.nee_transmittance_idx = psim.Combo(
                        'NEE transmittance', ui.nee_transmittance_idx,
                        NEE_TRANSMITTANCE_MODES)
                    _, ui.use_anyhit_nee = psim.Checkbox(
                        'NEE any-hit', ui.use_anyhit_nee)
                    _, ui.use_anyhit_sampling = psim.Checkbox(
                        'distance-sampling any-hit', ui.use_anyhit_sampling)
                    _, ui.residual_weight_clamp = _input_float(
                        psim, 'residual weight clamp',
                        ui.residual_weight_clamp, vmin=0.0, vmax=100.0)
                    _, ui.truncate_at_shell = psim.Checkbox(
                        'truncate at shell', ui.truncate_at_shell)
                _, ui.nee_truncate_at_shell = psim.Checkbox(
                    'NEE truncate at shell', ui.nee_truncate_at_shell)
                _, ui.phase_idx = psim.Combo('phase', ui.phase_idx, PHASE_TYPES)
                if ui.phase_idx == 1:
                    _, ui.hg_g = _input_float(
                        psim, 'hg g', ui.hg_g, vmin=-0.99, vmax=0.99)
            _, ui.max_spp = _input_int(psim, 'spp budget', ui.max_spp, vmin=1)
            psim.TreePop()

        if psim.TreeNodeEx('Kernel'):
            if ui.integrator_idx == 2:
                psim.TextUnformatted(
                    'kernel: gaussian (prb stochastic is Gaussian-only)')
            else:
                _, ui.kernel_idx = psim.Combo(
                    'kernel', ui.kernel_idx, KERNEL_TYPES)
            _, ui.kernel_normalized = psim.Checkbox('normalized', ui.kernel_normalized)
            _, ui.kernel_full_range = psim.Checkbox('full range', ui.kernel_full_range)
            psim.TreePop()

        if active_envmap() and psim.TreeNodeEx('Environment'):
            _, ui.env_azimuth = _input_float(
                psim, 'azimuth', ui.env_azimuth, vmin=-180.0, vmax=180.0, fmt='%.2f')
            _, ui.env_elevation = _input_float(
                psim, 'elevation', ui.env_elevation, vmin=-90.0, vmax=90.0, fmt='%.2f')
            _, ui.env_scale = _input_float(
                psim, 'intensity', ui.env_scale, vmin=0.0, vmax=10.0)
            psim.TreePop()

        if psim.TreeNodeEx('Asset', psim.ImGuiTreeNodeFlags_DefaultOpen):
            _, ui.opacity_weight = _input_float(
                psim, 'opacity weight', ui.opacity_weight, vmin=0.0, vmax=1000.0)
            if ui.integrator_idx >= 1:
                _, ui.albedo_scale = _input_float(
                    psim, 'albedo', ui.albedo_scale, vmin=0.0, vmax=1.0)
            _, ui.adaptive_extent = psim.Checkbox(
                'adaptive extents', ui.adaptive_extent)
            if ui.adaptive_extent:
                _, ui.min_contribution = _input_float(
                    psim, 'min contribution', ui.min_contribution,
                    vmin=1e-8, vmax=1e-1, fmt='%.2e')
                _, ui.base_extent = _input_float(
                    psim, 'max extent (sigma)', ui.base_extent,
                    vmin=0.5, vmax=8.0)
                _, ui.min_extent = _input_float(
                    psim, 'min extent (sigma)', ui.min_extent,
                    vmin=0.01, vmax=2.0)
            else:
                _, ui.base_extent = _input_float(
                    psim, 'kernel extent', ui.base_extent,
                    vmin=0.5, vmax=8.0)
            psim.TreePop()

        if psim.TreeNodeEx('Display & export', psim.ImGuiTreeNodeFlags_DefaultOpen):
            _, ui.exposure = _input_float(
                psim, 'exposure (display)', ui.exposure, vmin=-6.0, vmax=6.0)
            if psim.Button('Save frame (EXR+PNG+JSON)'):
                if session.image is not None:
                    hdr = session.image
                    display = tomography_radiance_for_display(
                        hdr, ui, envmap_path=active_envmap(),
                        accumulator_is_tau=session.accumulate_tau)
                    paths = exporters.save_frame_bundle(
                        hdr, out_dir, exposure=ui.exposure,
                        display_image=display,
                        metadata=capture_metadata(
                            ps, ui, session, active_envmap(),
                            camera_moving=state['camera_moving']))
                    print('Saved frame bundle:', paths['exr'].name,
                          '/', paths['png'].name, '/', paths['json'].name)
            if psim.Button('Export Mitsuba scene'):
                xml = export_current_scene(ps, ui, asset, out_dir,
                                           envmap_path=active_envmap())
                print('Exported scene:', xml)
            psim.TreePop()
        psim.PopItemWidth()

        _sync_polyscope_projection(ps, ui, state)
        if _bind_ortho_scroll_to_scale(
                ps, ui, state, ortho_slider_changed=ortho_slider_changed):
            state['still_frames'] = 0
            state['camera_moving'] = True

        # --------------------------------------------------- state updates
        if ui.rebuild_key() != state['rebuild_key']:
            rebuild()
            session = state['session']

        session.max_spp = max(1, ui.max_spp)
        if ui.freq_filter_active:
            session.set_runtime_param(session.freq_key, mi.Float(ui.freq_threshold))
        session.set_opacity_weight(ui.opacity_weight, asset.opacity_base_levels())
        session.set_albedo_scale(ui.albedo_scale, asset.albedo_base_levels(),
                                 absolute=(ui.integrator_idx >= 1))
        session.emitter_scale = (
            ui.env_scale if active_envmap() else 1.0)
        if active_envmap():
            session.set_runtime_param(session.envmap_to_world_key, env_to_world(ui))
            session.set_runtime_param(session.envmap_scale_key, mi.Float(ui.env_scale))

        cam = ps.get_view_camera_parameters()
        origin = np.asarray(cam.get_position())
        target = origin + np.asarray(cam.get_look_dir())
        up = np.asarray(cam.get_up_dir())
        fov = float(cam.get_fov_vertical_deg())
        cam_key = (tuple(np.round(origin, 5)), tuple(np.round(target, 5)),
                   tuple(np.round(up, 5)))
        if state['prev_cam_key'] is not None and cam_key != state['prev_cam_key']:
            state['still_frames'] = 0
            state['camera_moving'] = True
        else:
            state['still_frames'] += 1
            if state['still_frames'] >= ui.still_frames_threshold:
                state['camera_moving'] = False
        state['prev_cam_key'] = cam_key

        eff_res = (ui.preview_resolution if state['camera_moving']
                   else ui.resolution)
        camera_mode = ('perspective' if ui.camera_mode_idx == 1
                       else 'orthographic')
        session.set_camera(
            origin, target, up, fov, eff_res, eff_res,
            camera_mode=camera_mode,
            perspective_fov=ui.perspective_fov,
            ortho_scale=ui.ortho_scale)
        session.update_state(ui.runtime_key())

        img = session.step()
        if img is not None:
            rad = tomography_radiance_for_display(
                img, ui, envmap_path=active_envmap(),
                accumulator_is_tau=session.accumulate_tau)
            disp = np.clip(rad * (2.0 ** ui.exposure), 0.0, None) ** (1.0 / 2.2)
            ps.add_color_image_quantity(
                'render', np.clip(disp, 0.0, 1.0),
                enabled=True, show_fullscreen=True, show_in_imgui_window=False)

    ps.set_user_callback(viewer_callback)
    ps.show()


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------

def _camera_record(ps, ui, *, camera_moving: bool = False) -> dict[str, Any]:
    cam = ps.get_view_camera_parameters()
    origin = np.asarray(cam.get_position())
    target = origin + np.asarray(cam.get_look_dir())
    eff_res = ui.preview_resolution if camera_moving else ui.resolution
    return {
        'projection': ('perspective' if ui.camera_mode_idx == 1
                       else 'orthographic'),
        'origin': [float(c) for c in origin],
        'target': [float(c) for c in target],
        'up': [float(c) for c in cam.get_up_dir()],
        'fov_y_deg': float(ui.perspective_fov if ui.camera_mode_idx == 1
                           else cam.get_fov_vertical_deg()),
        'ortho_scale': float(ui.ortho_scale),
        'resolution': [eff_res, eff_res],
    }


def capture_metadata(ps, ui, session,
                     envmap_path: Optional[str] = None,
                     *, camera_moving: bool = False) -> dict[str, Any]:
    env: dict[str, Any] = {'type': 'constant', 'radiance': 1.0}
    if envmap_path:
        env = {
            'type': 'envmap',
            'azimuth_deg': ui.env_azimuth,
            'elevation_deg': ui.env_elevation,
            'intensity': ui.env_scale,
        }
    return {
        'accumulated_spp': session.spp_done,
        'max_spp': ui.max_spp,
        'ms_per_sample': round(session.ms_per_sample, 4),
        'integrator': build_integrator_dict(ui),
        'accumulation_space': (
            'optical_depth' if session.accumulate_tau else 'radiance'),
        'display_radiance': (
            'exp(-mean_optical_depth)'
            if session.accumulate_tau else 'identity'),
        'linear_tau_samples': session.linear_tau_samples,
        'opacity_weight': ui.opacity_weight,
        'albedo_scale': ui.albedo_scale,
        'environment': env,
        'camera': _camera_record(ps, ui, camera_moving=camera_moving),
    }


def export_current_scene(ps, ui, asset, out_dir, *,
                         envmap_path: Optional[str] = None):
    cam = ps.get_view_camera_parameters()
    origin = np.asarray(cam.get_position())
    target = origin + np.asarray(cam.get_look_dir())
    up = [float(c) for c in cam.get_up_dir()]
    T = mi.ScalarTransform4f
    to_world = T().look_at(
        origin=[float(c) for c in origin],
        target=[float(c) for c in target],
        up=up)
    if ui.camera_mode_idx == 1:
        sensor_dict: dict[str, Any] = {
            'type': 'perspective',
            'fov': float(ui.perspective_fov),
            'fov_axis': 'y',
            'to_world': to_world,
        }
    else:
        sensor_dict = {
            'type': 'orthographic',
            'to_world': to_world @ T().scale(
                [ui.ortho_scale, ui.ortho_scale, 1.0]),
        }
    sensor_dict.update({
        'film': {'type': 'hdrfilm', 'width': ui.resolution,
                 'height': ui.resolution},
        'sampler': {'type': 'independent', 'sample_count': ui.max_spp},
    })
    levels = []
    for data, meta in zip(asset.levels, asset.meta):
        # Albedo stays unscaled here, matching build_scene (the live albedo
        # slider is applied through mi.traverse, not baked into the export).
        levels.append(_level_with_extents(
            _scaled_level_data(data, ui.opacity_weight, 1.0), meta, ui))
    return exporters.export_scene(
        out_dir,
        levels=levels,
        level_meta=asset.meta,
        integrator=build_integrator_dict(ui),
        sensor=sensor_dict,
        emitter=build_emitter_dict(ui, envmap_path))
