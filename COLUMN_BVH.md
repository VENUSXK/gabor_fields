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
support parameters change. Compact Column PLY assets preserve the shared parameters
and primitives_per_column; expanded per-Gaussian arrays are reconstructed on loading.

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


## Column mesh shells

On `exp/column-mesh-shell`, add `--use_column_mesh` together with
`--use_columns_representation` to training, static rendering, or video rendering.
The flag selects `ellipsoidsmesh` with `shell=box` and an explicit
`primitives_per_column`. Training also enables the grouped-geometry configuration.
Omitting the new flag preserves the existing analytic Column rendering path; training
still uses its existing mesh default unless `--use_column_bvh` is requested.
Primitive Y centers remain trainable on this branch.

Each column contributes one conservative axis-aligned box mesh: 24 template vertices
and 12 triangles. The box encloses the union of all child ellipsoid supports, including
rotation and per-primitive extents. With 6144 columns and four primitives per column,
there are 73728 triangles instead of the independent UV shells' 1769472 triangles.
OptiX still builds its own triangle BVH; one shell does not mean one BVH leaf.
Shared arrays, Gaussian evaluation, optimizer updates, and compact PLY export are reused.
Grouped mesh geometry requires CUDA and fails explicitly on other backends.

A shell hit is only a candidate. CUDA reuses the analytic child support test and the
existing optical-depth, free-flight, and coverage helpers with resolved child indices.
The complete closed box is queried using its exit faces, once per column, so rays
starting inside a proxy and finite segments ending before a proxy exit are supported.
Ordinary geometry is traced separately; its nearest hit is merged with the closest
actual child, rather than treating a proxy face as a Gaussian event. This preserves
opaque blockers and overlapping-column ordering. Dedicated queries preserve their
original segment clipping, activity masks, sampling seeds, and child coverage ids.

The correctness-first implementation uses an additional complete-proxy traversal.
That is expensive for the sequential closest-hit walk and is not a general speedup.
An illustrative synthetic thin-slab test on an RTX 5070 Ti Laptop GPU, 24576 Gaussians,
256 x 144 orthographic pixels, spp=1, seed=0, three warmups and six timed renders gave:

| Geometry | Warm scene construction median | Sequential tomography median | Any-hit tomography median |
| --- | ---: | ---: | ---: |
| Independent UV mesh shells | 8.80 ms | 25.57 ms | 4.02 ms |
| Analytic Column bounds | 1.97 ms | 79.97 ms | 6.90 ms |
| Column box mesh shells | 3.07 ms | 632.82 ms | 3.95 ms |

Scene construction was timed separately with two warmups and four samples, using
preconstructed parameter tensors and synchronization after each load. It includes
scene setup and acceleration construction, rather than isolated BVH builder time.
These are forward-render timings, not training or multiple-scattering benchmarks.
The analytic Column and Column mesh images matched at rtol=5e-5, atol=2e-6.
Use identical assets, integrators, cameras, and settings for an application benchmark;
prefer the analytic path for sequential traversal until that path is improved. For
training or tomography experiments, `--use_anyhit` selects the relevant any-hit path.

The regression suite also compares child ids and distances, finite segments entirely
inside a proxy, overlapping columns with opaque geometry, containing-support collection,
stochastic queries, one-child columns, tensor extents, compact PLY reload, optimization
updates, forward images and shared-parameter gradients. Both multiple-scattering
integrators are checked against analytic Column geometry. Run the verification command
above after rebuilding the native plugin and refreshing its Python runtime package.
