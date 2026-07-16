# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Viewer-side stochastic-tomography accumulation helpers.

Display logic for the interactive viewer's progressive tomography accumulation
(the only consumer). Unbiased multi-sample tomography with stochastic
level-of-detail must average *optical depth* per sample and apply ``exp(-τ)``
(times emitter radiance) only for display. Averaging transmittance after
per-sample exponentiation is biased because ``E[exp(-τ)] ≠ exp(-E[τ])``.

``gfields_tomography`` accumulates linear optical depth via
``render_marginalization=True`` (equivalently ``output='optical_depth'``);
both paths are handled here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gfields.lod import DETERMINISTIC_CRITERIA

# Orientation modes that sample stochastically (tomography / straight rays).
_STOCHASTIC_ORIENTATION = frozenset(
    {'importance', 'control_variate_threshold'})

# Criteria that must not appear on tomography (NEE / path-tracing only).
_TOMOGRAPHY_INVALID_CRITERIA = frozenset({'biased_zero', 'gaussian_ms'})


def tomography_criteria_choices(criteria: list[str] | tuple[str, ...]) -> list[str]:
    """Criteria exposed for tomography UIs / CLI (drops NEE-only entries)."""
    return [c for c in criteria if c not in _TOMOGRAPHY_INVALID_CRITERIA]


def needs_mean_tau_accumulation(
    *,
    criteria: str,
    orientation_selection: str = 'deterministic',
    steerable: bool = False,
) -> bool:
    """True when per-sample optical depth must be averaged before ``exp``."""
    if criteria not in DETERMINISTIC_CRITERIA:
        return True
    if steerable and orientation_selection in _STOCHASTIC_ORIENTATION:
        return True
    return False


def tomography_integrator_props(
    base: dict[str, Any],
    *,
    criteria: str,
    orientation_selection: str = 'deterministic',
    steerable: bool = False,
) -> dict[str, Any]:
    """Return *base* with linear-depth integrator flags when required."""
    props = dict(base)
    if needs_mean_tau_accumulation(
            criteria=criteria,
            orientation_selection=orientation_selection,
            steerable=steerable):
        # Match production renders / the hierarchical alias: accumulate mean τ
        # per sample (radiance output with a constant emitter equals τ) and
        # apply exp(-τ) only after averaging.
        props['render_marginalization'] = True
        props['output'] = 'radiance'
    else:
        props.setdefault('output', 'radiance')
        props.setdefault('render_marginalization', False)
    return props


def buffer_is_linear_tau(
    *,
    integrator: dict[str, Any] | None = None,
    criteria: str = 'deterministic',
    orientation_selection: str = 'deterministic',
    steerable: bool = False,
    hierarchical_alias: bool = False,
) -> bool:
    """True when the film holds mean optical depth (not transmittance)."""
    if integrator is not None:
        return bool(integrator.get('render_marginalization')) or (
            integrator.get('output') == 'optical_depth')
    return hierarchical_alias or needs_mean_tau_accumulation(
        criteria=criteria,
        orientation_selection=orientation_selection,
        steerable=steerable,
    )


def tau_to_radiance(
    tau: np.ndarray,
    *,
    emitter_scale: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Map mean optical depth to emitted radiance (constant or per-pixel scale)."""
    tau = np.asarray(tau, dtype=np.float64)
    scale = np.asarray(emitter_scale, dtype=np.float64)
    return scale * np.exp(-np.clip(tau, 0.0, None))


def film_sample_to_tau(
    sample: np.ndarray,
    *,
    linear_tau: bool,
    emitter_scale: float = 1.0,
) -> np.ndarray:
    """Map one per-sample film value to optical depth for accumulation.

    When the integrator uses ``render_marginalization`` (or ``output=
    optical_depth``), each sample is already linear τ. Otherwise each sample
    holds transmittance ``T = exp(-τ)`` (times emitter radiance for non-constant
    emitters); recover τ via ``-log(T / emitter_scale)``.
    """
    rgb = np.asarray(sample, dtype=np.float64)[..., :3]
    if linear_tau:
        return rgb
    scale = max(float(emitter_scale), 1e-8)
    transmittance = np.clip(rgb / scale, 1e-8, 1.0)
    return -np.log(transmittance)


def interpret_tomography_buffer(
    buffer: np.ndarray,
    *,
    criteria: str = 'deterministic',
    orientation_selection: str = 'deterministic',
    steerable: bool = False,
    emitter_scale: float | np.ndarray = 1.0,
    hierarchical_alias: bool = False,
    integrator: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert a raw integrator film to display radiance."""
    rgb = np.asarray(buffer, dtype=np.float64)[..., :3]
    linear = buffer_is_linear_tau(
        integrator=integrator,
        criteria=criteria,
        orientation_selection=orientation_selection,
        steerable=steerable,
        hierarchical_alias=hierarchical_alias,
    )
    if linear:
        return tau_to_radiance(rgb, emitter_scale=emitter_scale)
    return rgb
