# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Frame and scene exports for the interactive viewer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import mitsuba as mi

from gfields.lod.utils import write_numpy_level_ply


def _timestamped(out_dir: Path, stem: str, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{suffix}"


def _tonemap(image: np.ndarray, exposure: float = 0.0, gamma: float = 2.2) -> np.ndarray:
    """Exposure + gamma tonemap to display-referred [0, 1] (matches viewer)."""
    ldr = np.clip(image * (2.0 ** exposure), 0.0, None) ** (1.0 / gamma)
    return np.clip(ldr, 0.0, 1.0)


def save_exr(image: np.ndarray, out_dir: Path, stem: str = 'frame') -> Path:
    path = _timestamped(out_dir, stem, '.exr')
    mi.util.write_bitmap(str(path), image)
    return path


def save_png(image: np.ndarray, out_dir: Path, stem: str = 'frame',
             exposure: float = 0.0, gamma: float = 2.2) -> Path:
    path = _timestamped(out_dir, stem, '.png')
    mi.util.write_bitmap(str(path), _tonemap(image, exposure, gamma))
    return path


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of capture metadata to JSON-serializable types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'numpy'):  # drjit / mitsuba tensors
        return np.asarray(value).tolist()
    return str(value)


def save_frame_bundle(image: np.ndarray, out_dir: Path, *,
                      stem: str = 'frame', exposure: float = 0.0,
                      gamma: float = 2.2,
                      display_image: Optional[np.ndarray] = None,
                      metadata: Optional[dict[str, Any]] = None) -> dict[str, Path]:
    """Write a capture bundle sharing one timestamped stem.

    Emits the linear-HDR ``.exr``, the tonemapped display ``.png`` (same
    exposure/gamma as the live viewport) and a ``.json`` sidecar recording the
    number of accumulated samples, the full integrator configuration, the
    camera pose and the environment / display settings — everything needed to
    reproduce the captured frame. Returns the written paths keyed by extension.

    *image* is written to EXR as-is (e.g. mean optical depth). *display_image*,
    when given, is tonemapped for the PNG (e.g. ``exp(-τ)`` radiance).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    exr_path = out_dir / f'{base}.exr'
    png_path = out_dir / f'{base}.png'
    json_path = out_dir / f'{base}.json'

    mi.util.write_bitmap(str(exr_path), image)
    mi.util.write_bitmap(
        str(png_path), _tonemap(display_image if display_image is not None else image,
                                exposure, gamma))

    record: dict[str, Any] = {
        'timestamp': base.split('_', 1)[1] if '_' in base else base,
        'exposure': float(exposure),
        'gamma': float(gamma),
        'resolution': [int(image.shape[1]), int(image.shape[0])],
    }
    if metadata:
        record.update(_jsonable(metadata))
    json_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    return {'exr': exr_path, 'png': png_path, 'json': json_path}


def _xml_value(key: str, value: Any, indent: str) -> str:
    """Serialize one property to Mitsuba scene-XML (mitsuba >= 3.8 removed
    mi.xml.dict_to_xml; this covers the value types our exports use)."""
    if isinstance(value, bool):
        return f'{indent}<boolean name="{key}" value="{str(value).lower()}"/>'
    if isinstance(value, int):
        return f'{indent}<integer name="{key}" value="{value}"/>'
    if isinstance(value, float):
        return f'{indent}<float name="{key}" value="{value}"/>'
    if isinstance(value, str):
        return f'{indent}<string name="{key}" value="{value}"/>'
    if isinstance(value, mi.ScalarTransform4f):
        rows = np.asarray(value.matrix)
        flat = ' '.join(f'{v:.9g}' for v in rows.reshape(-1))
        return (f'{indent}<transform name="{key}">\n'
                f'{indent}    <matrix value="{flat}"/>\n'
                f'{indent}</transform>')
    if isinstance(value, dict):
        if value.get('type') == 'rgb':
            v = value['value']
            v = ', '.join(f'{c:.9g}' for c in v) if isinstance(v, (list, tuple)) else f'{v:.9g}'
            return f'{indent}<rgb name="{key}" value="{v}"/>'
        inner = ''.join('\n' + _xml_value(k, v, indent + '    ')
                        for k, v in value.items() if k != 'type')
        tag = _TAG_FOR_KEY.get(key, key)
        return (f'{indent}<{tag} type="{value["type"]}" name="{key}">'
                f'{inner}\n{indent}</{tag}>')
    raise TypeError(f'cannot serialize {key!r} of type {type(value)} to XML')


_TAG_FOR_KEY = {
    'integrator': 'integrator', 'sensor': 'sensor', 'emitter': 'emitter',
    'film': 'film', 'sampler': 'sampler', 'radiance': 'spectrum',
}


def _dict_to_xml(scene: dict[str, Any], path: Path) -> None:
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<scene version="3.0.0">']
    for key, value in scene.items():
        if key == 'type':
            continue
        if isinstance(value, dict) and key not in _TAG_FOR_KEY:
            tag = 'shape' if 'filename' in value or 'mesh' in value.get('type', '') else key
            inner = ''.join('\n' + _xml_value(k, v, '        ')
                            for k, v in value.items() if k != 'type')
            lines.append(f'    <{tag} type="{value["type"]}" id="{key}">'
                         f'{inner}\n    </{tag}>')
        else:
            lines.append(_xml_value(key, value, '    '))
    lines.append('</scene>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def export_scene(out_dir: Path, *,
                 levels: list[dict[str, Any]],
                 level_meta: list[dict[str, Any]],
                 integrator: dict[str, Any],
                 sensor: dict[str, Any],
                 emitter: dict[str, Any]) -> Path:
    """Write a self-contained Mitsuba scene: one PLY per pyramid level plus
    a scene.xml referencing them with the current integrator, camera and
    environment settings.

    Args:
        levels: numpy level dicts (centers/scales/quaternions/omega/...); a
            level renders as Gaussian where its omega is 0 and Gabor otherwise.
        level_meta: per-level ``{'visibility_mask': int}``.
        integrator / sensor / emitter: ``mi.load_dict``-style dicts using
            only XML-serializable values.
    """
    out_dir = Path(out_dir) / time.strftime('scene_%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    scene: dict[str, Any] = {
        'type': 'scene',
        'integrator': integrator,
        'sensor': sensor,
        'emitter': emitter,
    }
    for i, (data, meta) in enumerate(zip(levels, level_meta)):
        ply_path = out_dir / f'level_{i}.ply'
        write_numpy_level_ply(data, ply_path)
        entry: dict[str, Any] = {
            'type': 'ellipsoidsmesh',
            'shell': 'uv_sphere',
            'filename': ply_path.name,
            'visibility_mask': int(meta.get('visibility_mask', 255)),
        }
        scene[f'primitives_{i}'] = entry

    xml_path = out_dir / 'scene.xml'
    _dict_to_xml(scene, xml_path)
    return xml_path
