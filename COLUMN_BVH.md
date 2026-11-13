# Column BVH

Enable grouped CUDA acceleration with `--use_columns_representation --use_column_bvh`.
The existing default remains `ellipsoidsmesh`; the grouped mode reuses the analytic
`ellipsoids` plugin. Each consecutive group of K expanded Gaussians contributes one
conservative union AABB to OptiX, while the full Gaussian parameter arrays remain intact.
For 6144 columns and four primitives per column, this means 6144 BVH primitives for
24576 Gaussians. This is a bound-count reduction, not a guaranteed fourfold speedup.

Ordinary closest-hit queries test children and return the actual Gaussian id through
intersection attribute 0. Surface interactions decode that id before looking up geometry
or attributes. Dedicated any-hit queries report one event per column and reuse the
existing per-Gaussian optical-depth, stochastic sampling, and coverage routines inside
that event. Children missing the ray are excluded even if their union AABB intersects it.

Parameter writes continue to use the existing expand/write/update flow. The analytic
shape recomputes column bounds and marks acceleration data dirty when geometry or
support parameters change. Saved PLY assets still contain every expanded Gaussian;
they do not preserve the column grouping or shared optimizer state.

Grouped geometry currently requires the CUDA backend. Scalar, LLVM, and Metal grouped
requests fail explicitly. The analytic support differs from the existing polygonal
`uv_sphere` proxy, so isolate the grouping effect by comparing analytic K=1 versus
analytic K>1, as the tests do. Neither the Gaussian density formula nor the optimizer
parameter sharing changes when enabling this switch.

The shared analytic intersection now projects the ray origin with
`dot(-o,d) / squared_norm(d)` rather than `dot(-o,d) / norm(d)`. The latter
uses a distance instead of the ray parameter and caused cancellation for small
Gaussian scales, admitting false child hits inside a larger union bound. Both CUDA
and CPU analytic intersections use the corrected projection. A 24576-Gaussian,
2048-ray regression test guards grouped-versus-independent optical-depth parity.

## Build

This change includes native code and regenerated OptiX PTX. Updating Python alone is
insufficient. Rebuild the Mitsuba fork and refresh its Python runtime package:

```bash
./.venv/bin/python gfields/build_mitsuba.py --python ./.venv/bin/python --jobs 8
```

The checked-in PTX targets compute capability 7.5 and PTX ISA 8.2, compiled with CUDA
12.2 and OptiX 8.0 headers. To regenerate it after changing CUDA sources, use the
existing Makefile in `submodules/mitsuba3/resources/ptx` with the appropriate CUDA,
OptiX include path, and supported host compiler. Rebuild Mitsuba afterwards. On the
local Windows development build, native compilation and `copy-targets` were both run.

## Verification

```bash
./.venv/bin/python -m unittest gfields.tests.test_column_training gfields.tests.test_column_bvh
```

Tests compare analytic ungrouped and grouped traversal: hit ids/distances, short rays,
interior origins, gaps between children, deterministic and stochastic transmittance,
free-flight sampling, overlapping coverage collection, forward images, parameter
gradients, bounds updates, and complete expanded PLY export. Short training runs also
exercise the new flag with any-hit enabled and disabled.
