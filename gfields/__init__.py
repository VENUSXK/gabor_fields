# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import mitsuba as mi

if mi.variant() is None:
    for _variant in ('cuda_ad_rgb', 'llvm_ad_rgb'):
        if _variant in mi.variants():
            mi.set_variant(_variant)
            break
    else:
        raise ImportError('gfields requires a JIT Mitsuba variant '
                          '(cuda_ad_rgb or llvm_ad_rgb); none is available.')
elif not mi.variant().startswith(('cuda', 'llvm')):
    # A scalar variant lacks ShapePtr & friends: the plugin modules below would
    # die with an opaque AttributeError halfway through. Fail early and clearly.
    raise ImportError(
        f"gfields requires a JIT Mitsuba variant, but {mi.variant()!r} is "
        f"active. Call mi.set_variant('cuda_ad_rgb') (or 'llvm_ad_rgb') "
        f"before importing gfields.")

# Upstream volprim lives in submodules/volprim and is imported directly: it
# registers `volprim_prb` / `volprim_rf` / `volprim_tomography`
# plugins and provides shared, unmodified helpers reused below (cameras, utils, io, stack).
_SUBMODULE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'volprim'))
if os.path.isdir(_SUBMODULE) and _SUBMODULE not in sys.path:
    sys.path.insert(0, _SUBMODULE)
try:
    import volprim
except ImportError as e:
    raise ImportError('gfields requires the upstream volprim submodule; run '
                      '`git submodule update --init submodules/volprim`.') from e

from . import benchmark
from . import cameras     # camera factories + samplers (+ re-exports volprim.cameras)
from . import io          # 
from . import optimizers  # BoundedAdam
from . import losses      # image losses / metrics (L1/L2/PSNR/SSIM)
from . import utils       # (re-exports volprim.utils)
from . import integrators # (registers the gfields_* plugins)
from . import references   # (registers gfields_reference: fast absorption tracker)
