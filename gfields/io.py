# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
I/O routines for handling Mitsuba scene assets. Includes functions to load
"Python assets" (Mitsuba scenes stored as Python modules using dictionaries)
and convert existing scenes into dictionaries, which can be written to disk
as Python assets.
'''

import os, shutil, logging
from os.path import join, basename, splitext, exists
import numpy as np

import drjit as dr
import mitsuba as mi

from volprim.io import asset_to_dict, scale_films, ellipsoid_dict_to_ply

def get_plugin_type(obj):
    '''
    Fetch the plugin type name for a given instance.
    This function also handle the corner cases of expanded objects
    '''
    if isinstance(obj, mi.Scene):
        return 'scene'
    # Resolve the C++ class name. `type(obj).__name__` only reports the bound
    # base (e.g. 'Mesh' for an EllipsoidsMesh), so prefer `class_().alias()` /
    # `class_name()` depending on which the current Mitsuba build exposes.
    if hasattr(obj, 'class_') and callable(obj.class_):
        cname = obj.class_().alias()
    elif hasattr(obj, 'class_name'):
        cname = obj.class_name()
    else:
        cname = type(obj).__name__
    if cname in ('scene', 'Scene'):
        return 'scene'
    # Python-defined integrators share a C++ base ('SamplingIntegrator' /
    # 'CppADIntegrator'); dispatch on the concrete Python subclass name.
    if cname in ('SamplingIntegrator', 'CppADIntegrator', 'ADIntegrator', 'RBIntegrator'):
        return {
            'PRBIntegrator': 'prb',
            'VolumetricPrimitiveRadianceFieldIntegrator': 'volprim_rf',
            'GfieldsPRBIntegrator': 'gfields_prb',
            'GfieldsPRBStochasticIntegrator': 'gfields_prb_stochastic',
            'GfieldsTomographyIntegrator': 'gfields_tomography',
        }.get(type(obj).__name__, cname.lower())
    class_to_plugin = {
        'PerspectiveCamera': 'perspective', 'ProjectiveCamera': 'perspective',
        'OrthographicCamera': 'orthographic', 'ThinLensCamera': 'thinlens',
        'ConstantBackgroundEmitter': 'constant', 'EnvironmentMapEmitter': 'envmap',
        'PointLight': 'point', 'DirectionalEmitter': 'directional',
        'SpotLight': 'spot', 'AreaLight': 'area',
        'SRGBReflectanceSpectrum': 'srgb', 'SRGBEmitterSpectrum': 'srgb',
        'IndependentSampler': 'independent',
        'BitmapTextureImpl_FP16': 'bitmap', 'BitmapTextureImpl': 'bitmap',
    }
    if cname in class_to_plugin:
        return class_to_plugin[cname]
    return cname.lower()  # e.g. 'EllipsoidsMesh' -> 'ellipsoidsmesh'

def dict_to_asset(scene_dict: dict, output_folder: str, verbose=False):
    '''
    Generate a Python asset that contains a dictionary that represents a Mitsuba
    scene.
    '''
    sensor_types  = ['perspective', 'orthographic', 'thinlens']
    emitter_types = ['envmap', 'constant', 'point', 'distant', 'spot', 'directional']

    if verbose:
        print(f'Writing asset to {output_folder} ...')

    def dict_to_string(d, output_folder, path, indent=0, add_resources=False) -> str:
        '''
        Convert a Python dictionary to its string Python code representation.
        The input dictionary will only contain strings as keys, and float, int, bool
        and stings as values.
        '''
        w        = lambda x: str(''.join([' ']) * indent) + x
        sanitize = lambda x: x.replace('.', '_')
        resolve  = lambda x: str(mi.Thread.thread().file_resolver().resolve(x))

        object_type = d.get('type', None)

        s = '{\n'
        if add_resources:
            s += w("    'resources': { 'type': 'resources', 'path': dirname(__file__) },\n")

        if object_type == 'meshholder':
            object_type = d['type'] = 'ply'
            m = mi.Mesh("", len(d['vertex_positions']) // 3, len(d['faces']) // 3)
            p = mi.traverse(m)
            p['faces']            = d['faces']
            p['vertex_positions'] = d['vertex_positions']
            p['vertex_normals']   = d['vertex_normals']
            p['vertex_texcoords'] = d['vertex_texcoords']
            p.update()
            os.makedirs(join(output_folder, 'meshes'), exist_ok=True)
            d['filename'] = join(output_folder, 'meshes', f'{path}.ply')
            m.write_ply(d['filename'])
            for k in ['faces', 'vertex_positions', 'vertex_normals', 'vertex_texcoords']:
                del d[k]

        is_ellipsoid = object_type and 'ellipsoid' in object_type
        if is_ellipsoid and not 'filename' in d:
            extras_keys = list(filter(lambda k: isinstance(d[k], np.ndarray), list(d.keys())))
            extras_keys = list(filter(lambda k: k not in ['centers', 'scales', 'quaternions'], extras_keys))

            os.makedirs(join(output_folder, 'data'), exist_ok=True)
            filename = join(output_folder, 'data', f'{path}.ply')
            ellipsoid_dict_to_ply(d, extras_keys, filename)

            for k in extras_keys + ['centers', 'scales', 'quaternions']:
                del d[k]
            d['filename'] = filename

        for k, v in d.items():
            if isinstance(v, dict) and v['type'] == 'resources':
                mi.Thread.thread().file_resolver().append(v['path'])
                continue

            s += w(f"    '{sanitize(k)}': ")

            if isinstance(v, dict):
                s += dict_to_string(v, output_folder, f'{path}.{k}', indent + 4)
                s += f",\n"
            elif isinstance(v, str):
                if k == 'filename':
                    src = resolve(v)
                    base, ext = splitext(basename(src))
                    dst_folder = {
                        '.obj': 'meshes',
                        '.ply': 'data' if is_ellipsoid else 'meshes',
                        '.jpg': 'textures',
                        '.png': 'textures',
                        '.exr': 'textures',
                        '.json': 'data',
                    }[ext]
                    os.makedirs(join(output_folder, dst_folder), exist_ok=True)
                    v = join(dst_folder, f'{base}{ext}')
                    dst = join(output_folder, v)
                    if not exists(dst):
                        shutil.copy(src, dst)
                    s += f"r\'{v}\',\n"
                else:
                    if k == 'id':
                        v = sanitize(v)
                    s += f"\'{v}\',\n"
            elif k == 'to_world':
                if object_type == 'envmap':
                    T = mi.ScalarTransform4f(v)
                    R = dr.transform_decompose(v.matrix)[1]
                    rotations = dr.quat_to_euler(R)
                    if any([r != 0.0 for r in rotations]):
                        s += f"T()"
                        if rotations[0] != 0.0:
                            s += f".rotate([1, 0, 0], {rotations[0]})"
                        if rotations[1] != 0.0:
                            s += f".rotate([0, 1, 0], {rotations[1]})"
                        if rotations[2] != 0.0:
                            s += f".rotate([0, 0, 1], {rotations[2]})"
                        s += f"\n"
                elif object_type in sensor_types:
                    T = mi.ScalarTransform4f(v)
                    origin    = T @ mi.ScalarPoint3f(0, 0, 0)
                    target    = T @ mi.ScalarPoint3f(0, 0, 1)
                    up        = T @ mi.ScalarVector3f(0, 1, 0)
                    s += f"T().look_at(\n"
                    s += w(f"         origin={origin},\n")
                    s += w(f"         target={target},\n")
                    s += w(f"         up={up},\n")
                    s += w(f"     ),\n")
                else:
                    s += f"T([{v.matrix[0]}, {v.matrix[1]}, {v.matrix[2]}, {v.matrix[3]}]),\n"
            elif k == 'wrap_mode' or k == 'filter_mode':
                s += f"dr.{v},\n"
            elif isinstance(v, np.ndarray) or dr.is_tensor_v(v):
                os.makedirs(join(output_folder, 'data'), exist_ok=True)
                filename = f'data/{path}.{k}.npy'
                np.save(join(output_folder, filename), np.array(v))
                s += f"np.load(join(dirname(__file__), \'{filename}\')),\n"
            elif isinstance(v, list):
                s += f"{v},\n"
            elif isinstance(v, mi.ScalarTransform3f):
                s += f"mi.ScalarTransform3f([\n"
                for i in range(3):
                    s += w(f"         {v.matrix[i]},\n")
                s += w(f"     ]),\n")
            elif isinstance(v, mi.ScalarTransform4f):
                s += f"T([\n"
                for i in range(4):
                    s += w(f"         {v.matrix[i]},\n")
                s += w(f"     ]),\n")
            elif isinstance(v, mi.Bitmap):
                os.makedirs(join(output_folder, 'textures'), exist_ok=True)
                filename = join('textures', f'{path}.exr')
                mi.util.write_bitmap(join(output_folder, filename), v)
                s += f'r\'{filename}\',\n'
            else:
                s += f"{v},\n"
        s += w('}')

        s = s.replace('\'bitmap\': ', '\'filename\': ')
        s = s.replace('\\', '/')

        return s

    assert scene_dict['type'] == 'scene', 'can only process scene dictionary!'

    # Sort scene elements
    sensors  = {}
    emitters = {}
    objects  = {}
    for k, v in scene_dict.items():
        if not isinstance(v, dict):
            continue
        if 'type' not in v:
            continue
        if v['type'] in sensor_types:
            sensors[k] = v
        elif v['type'] in emitter_types:
            emitters[k] = v
        else:
            objects[k] = v

    os.makedirs(output_folder, exist_ok=True)
    filename = join(output_folder, '__init__.py')
    with open(filename, 'w') as f:
        f.write('import os\n')
        f.write('from os.path import join, dirname\n')
        f.write('import numpy as np\n')
        f.write('import drjit as dr\n')
        f.write('import mitsuba as mi\n')
        f.write('from mitsuba.scalar_rgb import ScalarTransform4f as T\n')

        f.write('\n')

        f.write('OBJECTS = ')
        f.write(dict_to_string(objects, output_folder, 'root', add_resources=True))
        f.write('\n')
        f.write('\n')

        f.write('SENSORS = ')
        f.write(dict_to_string(sensors, output_folder, 'root', add_resources=True))
        f.write('\n')
        f.write('\n')

        f.write('EMITTERS = ')
        f.write(dict_to_string(emitters, output_folder, 'root', add_resources=True))
        f.write('\n')


def object_to_dict(root):
    '''
    Routine to convert a Mitsuba object into a dictionary that could be passed
    directly to `mi.load_dict()` to re-load the same object.
    '''
    class ObjectToDictTraversal(mi.TraversalCallback):
        def __init__(self, node=None, name=None):
            mi.TraversalCallback.__init__(self)
            self._node = node
            self._name = name
            self._dict = {}
            if node is not None:
                self._plugin_type = get_plugin_type(node)
                self._dict = { 'type': self._plugin_type }

        def put(self, name, value, flags, cpptype=None):
            """Unified method to register both objects and values with the traversal callback."""
            if isinstance(value, mi.Object):
                self.put_object(name, value, flags)
            else:
                self.put_parameter(name, value, flags, cpptype)

        def put_parameter(self, name, value, flags, value_type=None):
            # Filter out some of the parameters
            if name in ['enabled', 'silhouette_sampling_weight', 'sampling_weight']:
                return

            if value_type is not None:
                value = mi.get_property(value, value_type, self._node)

            if isinstance(value, (mi.Float, mi.UInt32, mi.Color3f, mi.Spectrum, mi.Vector3f)) and dr.width(value) == 1:
                value = dr.slice(value)
            elif isinstance(value, mi.Transform3f):
                value = mi.ScalarTransform3f(dr.slice(value.matrix))
            elif isinstance(value, mi.Transform4f):
                value = mi.ScalarTransform4f(dr.slice(value.matrix))

            self._dict[name] = value

        def put_object(self, name, node, flags):
            # Recursively traverse other object to assemble dictionary
            cb = ObjectToDictTraversal(node, name)
            node.traverse(cb)

            plugin_type = cb._plugin_type
            
            # Get parent class name, with fallback for objects without class_()
            if hasattr(node, 'class_') and callable(node.class_):
                parent_name = node.class_().parent().name()
            else:
                # Check if it's a Mesh by type inspection
                parent_name = type(node).__mro__[1].__name__ if len(type(node).__mro__) > 1 else ''

            # Handle special cases
            d = cb._dict

            # Meshes
            if (parent_name == 'Mesh' or isinstance(node, mi.Mesh)) and 'faces' in d:
                d['type'] = 'meshholder'
            # Ellipsoids - check plugin_type, type name, d['type'], and data structure for robustness
            is_ellipsoid = ('ellipsoid' in plugin_type.lower() or 
                           'ellipsoid' in type(node).__name__.lower() or
                           ('type' in d and 'ellipsoid' in str(d['type']).lower()))
            # Also detect ellipsoids by data structure: TensorXf with width divisible by 10
            if not is_ellipsoid and 'data' in d and hasattr(d['data'], 'array'):
                data_width = dr.width(d['data'].array)
                if data_width > 0 and data_width % 10 == 0:
                    is_ellipsoid = True
            if is_ellipsoid and 'data' in d:
                prim_count = dr.width(d['data'].array) // 10
                data = np.array(d['data']).reshape(prim_count, -1)
                d['centers']     = data[:, 0:3]
                d['scales']      = data[:, 3:6]
                d['quaternions'] = data[:, 6:10]
                del d['data']
                for k2, v2 in d.items():
                    if dr.is_array_v(v2):
                        d[k2] = np.array(v2).reshape(prim_count, -1)
                # Ensure type is set correctly for ellipsoid so dict_to_asset recognizes it
                if 'ellipsoid' not in d.get('type', '').lower():
                    d['type'] = 'ellipsoidsmesh'
            # Bitmap
            elif plugin_type == 'bitmap':
                d['bitmap'] = mi.Bitmap(d['data'])
                del d['data']
                d['wrap_mode']   = dr.WrapMode(d['wrap_mode'])
                d['filter_mode'] = dr.FilterMode(d['filter_mode'])
            # Film
            elif plugin_type == 'hdrfilm':
                d['width']  = d['size'][0]
                d['height'] = d['size'][1]
                for k in ['size', 'crop_size', 'crop_offset']:
                    del d[k]
            # sRGB
            elif plugin_type == 'srgb':
                d['type'] = 'rgb'
            # Sensors
            elif plugin_type == 'perspective' or plugin_type == 'orthographic':
                if 'shutter_open_time' in d:
                    d['shutter_close'] = d['shutter_open'] + d['shutter_open_time']
                    del d['shutter_open_time']
                if plugin_type == 'perspective' and 'x_fov' in d:
                    d['fov'] = d['x_fov']
                    d['fov_axis'] = 'x'
                    del d['x_fov']

            self._dict[name] = d

    cb = ObjectToDictTraversal()
    cb.put_object('root', root, None)
    return cb._dict['root']


def load_ellipsoid_ply(filename):
    '''Inverse of `ellipsoid_dict_to_ply`: read an ellipsoid `.ply` (the 3DGS
    layout written by `dict_to_asset`) back into raw per-primitive numpy arrays,
    undoing the log-scale and (i,j,k,r)->(r,i,j,k) quaternion reordering. Returns
    a dict `{centers, scales, quaternions, <extras: opacities, omega, albedo,
    extent, ...>}` matching the per-array npy format the pipeline used to store.'''
    import plyfile
    v = plyfile.PlyData.read(str(filename))['vertex'].data
    names = set(v.dtype.names)

    def cols(prefix, n):
        return np.stack([v[f'{prefix}_{i}'] for i in range(n)], axis=-1).astype(np.float32)

    out = {
        'centers':     np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32),
        'scales':      np.exp(cols('scale', 3)),
        'quaternions': np.stack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']], axis=-1).astype(np.float32),
    }
    geom = {'x', 'y', 'z', 'nx', 'ny', 'nz',
            'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'}
    dims = {}
    for name in names - geom:
        base, _, idx = name.rpartition('_')
        dims[base] = max(dims.get(base, 0), int(idx) + 1)
    for base, dim in dims.items():
        out[base] = cols(base, dim) if dim > 1 else v[f'{base}_0'].astype(np.float32)
    return out


def ply_asset_levels(ply_dir):
    '''Sorted pyramid level indices present as ``root.primitives_pyr*.ply`` in *ply_dir*
    (the canonical ``<asset>/data`` store); the .ply counterpart of :func:`npy_asset_levels`.'''
    import re
    from pathlib import Path
    levels = set()
    for f in Path(ply_dir).glob('root.primitives_pyr*.ply'):
        m = re.search(r'pyr(\d+)', f.name)
        if m:
            levels.add(int(m.group(1)))
    return sorted(levels)


# ---------------------------------------------------------------------------
# Steerable pyramids plys
# ---------------------------------------------------------------------------

STEERABLE_PLY_BASENAME = 'root.primitives_steerable.ply'


def steerable_ply_path(asset_dir):
    '''Canonical location of a steerable decomposition's combined .ply.'''
    return join(str(asset_dir), 'data', STEERABLE_PLY_BASENAME)


def is_steerable_asset(asset_dir):
    '''True if *asset_dir* holds a combined steerable ``.ply`` decomposition.'''
    return exists(steerable_ply_path(asset_dir))


# Per-primitive sub-band flag columns (constant within a sub-band).
_STEERABLE_FLAGS = ('visibility_mask', 'freq_level', 'orientation_index')


def save_steerable_ply(subbands, filename):
    '''Write a steerable decomposition to a single combined ellipsoid `.ply`.

    ``subbands`` is a list of dicts, one per (non-empty) sub-band, each with:
      - ``centers`` (N,3), ``scales`` (N,3), ``quaternions`` (N,4)
      - ``opacities`` (N,) or (N,1); optional ``omega`` (N,d), ``albedo`` (N,3),
        ``extent`` (N,) or (N,1)
      - scalar ``visibility_mask``, ``freq_level``, ``orientation_index``
    Empty sub-bands (0 primitives) are skipped; they carry no data and are
    reconstructed as needed from the surviving (freq_level, orientation_index)
    grid. Returns the number of primitives written.
    '''
    subbands = [s for s in subbands if len(np.asarray(s['centers'])) > 0]
    if not subbands:
        raise ValueError('save_steerable_ply: no non-empty sub-bands to write')

    def _field(key, dim, default):
        parts = []
        for s in subbands:
            n = len(np.asarray(s['centers']))
            v = s.get(key)
            if v is None:
                parts.append(np.full((n, dim), default, np.float32))
            else:
                parts.append(np.asarray(v, np.float32).reshape(n, dim))
        return np.concatenate(parts, axis=0)

    def _flag(key):
        parts = [np.full((len(np.asarray(s['centers'])), 1), float(s[key]), np.float32)
                 for s in subbands]
        return np.concatenate(parts, axis=0)

    # omega width is producer-defined (scalar magnitude, or a 3D omega vector);
    # keep whatever the sub-bands carry, defaulting to a single zero column.
    omega_dim = 1
    for s in subbands:
        v = s.get('omega')
        if v is not None:
            omega_dim = max(omega_dim,
                            np.asarray(v).reshape(len(np.asarray(s['centers'])), -1).shape[1])

    d = {
        'type':        'ellipsoidsmesh',
        'centers':     _field('centers', 3, 0.0),
        'scales':      _field('scales', 3, 1.0),
        'quaternions': _field('quaternions', 4, 0.0),
    }
    extras = ['opacities', 'omega', 'albedo', 'extent', *_STEERABLE_FLAGS]
    d['opacities'] = _field('opacities', 1, 1.0)
    d['omega']     = _field('omega', omega_dim, 0.0)
    d['albedo']    = _field('albedo', 3, 1.0)
    d['extent']    = _field('extent', 1, 3.0)
    for flag in _STEERABLE_FLAGS:
        d[flag] = _flag(flag)

    os.makedirs(os.path.dirname(str(filename)), exist_ok=True)
    ellipsoid_dict_to_ply(d, extras, str(filename))
    return len(d['centers'])


def load_steerable_ply(filename):
    '''Read a combined steerable `.ply` back into per-sub-band arrays + config.

    Returns ``(subbands, config)`` where ``subbands`` is a list of dicts (one per
    stored (freq_level, orientation_index) group) with keys ``centers``,
    ``scales``, ``quaternions``, ``opacities``, ``omega``, ``albedo``,
    ``extent``, ``visibility_mask``, ``freq_level``, ``orientation_index``, and
    ``config`` is ``{num_frequency_levels, num_orientations, total_subbands}``
    reconstructed from the flag columns (no sidecar needed).
    '''
    a = load_ellipsoid_ply(filename)
    for flag in _STEERABLE_FLAGS:
        if flag not in a:
            raise ValueError(f'{filename} is not a steerable ply (missing {flag!r} column)')

    vis  = np.rint(np.ravel(a['visibility_mask'])).astype(np.int64)
    freq = np.rint(np.ravel(a['freq_level'])).astype(np.int64)
    ori  = np.rint(np.ravel(a['orientation_index'])).astype(np.int64)

    # Group primitives by (freq_level, orientation_index), preserving first-seen
    # order so sub-bands come out Gaussian-first / low-frequency-first.
    order, seen = [], set()
    for key in zip(freq.tolist(), ori.tolist()):
        if key not in seen:
            seen.add(key)
            order.append(key)

    subbands = []
    for f_lvl, o_idx in order:
        m = (freq == f_lvl) & (ori == o_idx)
        sub = {'freq_level': int(f_lvl), 'orientation_index': int(o_idx),
               'visibility_mask': int(vis[m][0])}
        for k in ('centers', 'scales', 'quaternions', 'opacities',
                  'omega', 'albedo', 'extent'):
            if k in a:
                arr = a[k]
                sub[k] = arr[m] if np.ndim(arr) > 1 else np.ravel(arr)[m]
        subbands.append(sub)

    # Config is reconstructed from the flag columns (no sidecar). This assumes
    # the highest frequency level and orientation bin are each populated by at
    # least one primitive -- true for any real decomposition, since every bin
    # receives its best-aligned Gabors.
    gabor_ori = ori[freq > 0]
    config = {
        'num_frequency_levels': int(freq.max()) + 1 if freq.size else 0,
        'num_orientations': int(gabor_ori.max()) + 1 if gabor_ori.size else 0,
        'total_subbands': len(subbands),
    }
    return subbands, config


# ---------------------------------------------------------------------------
# Legacy per-array npy asset layout
# ---------------------------------------------------------------------------
# Assets kept as one .npy per attribute per level under ``npy_data/``. These
# loaders read that layout directly so pre-.ply assets still open; new assets
# use the unified data/*.ply store written by ``dict_to_asset``.

def npy_asset_levels(npy_dir):
    """Sorted pyramid level indices present in a legacy ``npy_data/`` dir."""
    import re
    from pathlib import Path
    levels = set()
    for f in Path(npy_dir).glob("centers_pyr*.npy"):
        m = re.search(r"pyr(\d+)", f.name)
        if m:
            levels.add(int(m.group(1)))
    return sorted(levels)


def load_ellipsoid_npy_level(npy_dir, level):
    """DEPRECATED: load one pyramid level from the legacy per-array
    ``npy_data/<attr>_pyr{level}.npy`` layout into the same dict as
    :func:`load_ellipsoid_ply`. Kept only to read pre-.ply assets."""
    import warnings
    from pathlib import Path
    warnings.warn(
        "gfields.io: the legacy npy_data/ asset layout is deprecated and will "
        "be removed in a future release; re-save assets as data/*.ply.",
        DeprecationWarning, stacklevel=2)
    npy_dir = Path(npy_dir)

    def _load(attr):
        p = npy_dir / f"{attr}_pyr{level}.npy"
        return np.load(p) if p.exists() else None

    centers, scales, quats = _load("centers"), _load("scales"), _load("quaternions")
    if centers is None or scales is None or quats is None:
        raise FileNotFoundError(
            f"No npy centers/scales/quaternions for level {level} in {npy_dir}")
    n = len(centers)
    out = {
        "centers": np.asarray(centers, np.float32).reshape(n, 3),
        "scales": np.asarray(scales, np.float32).reshape(n, 3),
        "quaternions": np.asarray(quats, np.float32).reshape(n, 4),
    }
    for attr in ("opacities", "sigma_t", "sigmat"):  # opacity aliases
        v = _load(attr)
        if v is not None:
            out["opacities"] = np.asarray(v, np.float32).reshape(n, -1)
            break
    for attr in ("omega", "albedo", "extent"):
        v = _load(attr)
        if v is not None:
            arr = np.asarray(v, np.float32)
            out[attr] = arr.reshape(n, -1) if arr.size >= n else arr.reshape(1, -1)
    return out
