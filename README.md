# Gabor Fields: Orientation-Selective Level-of-Detail for Volume Rendering

![Teaser: a smoke bunny reconstructed as a Gabor Field, shown at four
continuous levels of detail (512 → 33,280 Gabor kernels) with per-level optical
depth and power spectrum.](data/figures/teaser.png)

Project Page: https://arcanous98.github.io/projectPages/gaborVolumes.html

Reference implementation of **Gabor Fields: Orientation-Selective
Level-of-Detail for Volume Rendering**
Jorge Condor*, Nicolai Hermann*, Mehmet Ata Yurtsever, Piotr Didyk (USI Lugano, Switzerland),
Transactions on Graphics, SIGGRAPH 2026.

This repo is based on
[facebookresearch/volumetric_primitives](https://github.com/facebookresearch/volumetric_primitives)
(*Don't Splat your Gaussians*, Condor et al., SIGGRAPH 2025). What `gfields`
brings on top:

- **Path tracing Gabor kernel volumes**. Continuous LOD control, higher
  quality, and many acceleration opportunities vs. Gaussians alone. See the
  paper for details.
- **A substantially improved training pipeline** for all kernel types
  (including Gaussians), replacing *Don't Splat your Gaussians*'s.
- **Any-hit shaders for Mitsuba 3**, used throughout our integrators
  (~6× faster tomographic rendering vs. *Don't Splat your Gaussians*).
- **`gfields_prb_stochastic`**, a new integrator based on decomposition
  tracking ([Kutz et al. 2017](https://doi.org/10.1145/3072959.3073665)):
  ~2× faster than regular `gfields_prb`. Gaussians only, for now.
- **Lots of convenient tools for working with primitive volumes**, like a
  voxelizer, a vdb2npy script, a fully fledged Polyscope viewer, a residual ratio tracking
  integrator with supervoxels support for fast reference generation, and tools to
  optimize performance and memory usage of your trained assets, like dynamic
  pruning and adaptive extent clamping.

---

## Install

```bash
git clone --recursive https://github.com/Arcanous98/gabor_fields
cd gabor_fields
./install.sh                 # Windows PowerShell:  .\install.ps1
#   ./install.sh --extras    # + torch / lpips / wandb / metrics (needed for the voxelizer)
```

This builds a [uv](https://docs.astral.sh/uv/) environment and
**compiles our custom Mitsuba 3 and DrJIT forks**
([Arcanous98/mitsuba3_gfields](https://github.com/Arcanous98/mitsuba3_gfields),
which adds per-ray visibility masking,
per-primitive `extent` tensors and anyhit shaders). Needs `uv`, `cmake`, a C++ compiler
(MSVC/VS 2022 on Windows; gcc/clang + Ninja on Linux), and an NVIDIA driver. 
We currently don't support LLVM and Metal backends but
we are happy to welcome contributions :)

Validate:

```bash
python -m pytest gfields/tests/test_gfields.py -q
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import gfields; print('OK')"
```

---

## Train

```bash
examples/train.sh                      # bundled smoke asset, 512 Gauss + 4096 Gabor
examples/train.sh wdas8                # a named asset (needs its volume, see below)
examples/train.sh data/volumes/grids/mine.npy --opacity_scale 12   # your own volume
examples/train.sh --gaussian           # Gaussian-only baseline
```

Optimization is **tomographic** (absorption-only) and defaults to the OptiX
**any-hit** shader. Tune primitive
budget and density from the CLI (`--gauss_count`, `--gabor_count`,
`--opacity_scale`) — see [`examples/README.md`](examples/README.md) for good
combos and how the recipe and named-asset lookup live in `gfields/presets.py`.
Named assets (`wdas8`, `embergen`, `explosion`, `fire`, `smoke`, `smokev2`,
`tornado`, `dust`) resolve to volumes under `data/volumes/grids/`, downloaded
per [`data/volumes/asset_sources.md`](data/volumes/asset_sources.md); only
`smoke` ships with the repo (as `data/volumes/smoke.vol`). These are roughly
the presets and assets used in the paper, and are kept just as a reference.

Output -> `results/checkpoints/<asset>/`.

The optional **`--post_refine`** flag (off by default; **experimental** refines
the asset once optimization finishes: it prunes very shallow primitives
(with a PSNR loss budget, `--post_prune_budget`, default 0.1 dB) then adaptively
clamps the survivors' per-primitive extents, baking the tighter extents into the
asset `.ply` so later renders use them with no extra flag. `--post_prune` and
`--post_clamp` run each half on its own (the standalone clamp writes only a
sidecar `extents_pyr*.npy`). We are still tuning this feature so use with
caution.

---

## Render

```bash
python -m gfields.render <asset>                                             # tomography (ortho)
python -m gfields.render <asset> --multiple_scattering --envmap sky.exr --spp 64
python -m gfields.render <asset> --multiple_scattering --stochastic_sampling # gfields_prb_stochastic
python -m gfields.render video <asset> --motion turntable --frames 120       # MP4 via ffmpeg
```

`--stochastic_sampling` switches to **`gfields_prb_stochastic`**, our
decomposition-tracking path tracer for Gaussian assets: instead of collecting
the overlapping primitive set and solving for the mixture optical depth
(Newton/bisection), it draws one closed-form free flight per primitive and
takes the minimum of all traversed primitives along a ray (decomposition tracking, Kutz et al. 2017). Combine with
`--stochastic_selection_criteria control_variate_power_law` or `--orientation_selection
importance` (steerable assets) for the Laplacian/orientation-selective speedups
described in the paper. 

Full flag list: `python -m gfields.render --help` (and `... video --help`).

---

## Visualize

We include several trained assets under `data/examples/`and you can interactively
visualize them with our viewer:

```bash
gfields viewer                                      # bundled wdas8 cloud (tomography, constant emitter)
gfields viewer --integrator prb                     # multiple scattering + bundled env map
gfields viewer data/examples/tornado --envmap sky.exr --integrator prb
```

Interactive polyscope viewer with progressive accumulation and an LOD slider;
all three integrators are switchable live (`tomography` / `prb` /
`prb_stochastic` — the latter Gaussian-only assets) together with their
stochastic LOD criteria, NEE transmittance / shell-truncation modes and the
any-hit fast paths (on by default; `--no_anyhit` disables them at launch). The launch defaults are
deterministic tomography with any-hit and the constant emitter (radiance 1.0,
matching training); the multiple-scattering modes default to the bundled
environment map instead — also when switching modes live (an explicit
`--envmap` applies to every mode).
Launcher: `examples/view.{sh,ps1}`. The installer pulls in its
[polyscope](https://polyscope.run/py/) dependency; if you set up the environment
before that or see `No module named 'polyscope'`, install it with
`uv pip install -e ".[viewer]"` (or `pip install polyscope`).

---

## Extra tools

```bash
# Convert an OpenVDB grid to the cached .npy format used for training
python -m gfields.references.vdb --input mine.vdb --output data/volumes/grids/

# Voxelize a trained Gabor Field back to a dense grid (needs --extras: torch)
gfields voxelize asset   --asset data/examples/wdas8 --output voxelized.npy --grid_res 256 256 256
gfields voxelize compare --asset data/examples/wdas8 --grid_res 256 256 256 --tomography_only

# LOD / post-training refinement
gfields decompose steerable --asset <asset> --output <out> --num_orientations 3
gfields decompose lod-partition --path <checkpoint> --output <out>   # frequency-band Laplacian pyramid
gfields decompose clamp <asset> --level 1                            # adaptive per-primitive extents
gfields decompose prune  -p <asset> -r opacity -b 0.5 -a             # PSNR-budget pruning
```

Old `.vol` mitsuba grids convert automatically on first use (no separate tool needed).

---

## Tests

```bash
python -m pytest gfields/tests/test_gfields.py -q     
```

Auto-skips without a `cuda_ad_rgb` build.

---

## BibTeX

```bibtex
@article{condor2026gaborfields,
    author = {Condor, Jorge and Hermann, Nicolai and Yurtsever, Mehmet Ata and Didyk, Piotr},
    title = {Gabor Fields: Orientation-Selective Level-of-Detail for Volume Rendering},
    year = {2026},
    issue_date = {July 2026},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {45},
    number = {4},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3811369},
    doi = {10.1145/3811369},
    journal = {ACM Trans. Graph.},
    month = jul,
    articleno = {62},
    numpages = {20}
}
@article{condor2025dsyg,
    author  = {Condor, Jorge and Speierer, Sebastien and Bode, Lukas and Bozic, Aljaz and Green, Simon and Didyk, Piotr and Jarabo, Adrian},
    title   = {Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media},
    journal = {ACM Trans. Graph.}, year = {2025}, doi = {10.1145/3711853},
}
```

## Acknowledgements

Funded by the Swiss National Science Foundation (SNSF, Grant 200502) and an
academic gift from Meta, with access to Alps at CSCS (project ID u6). MIT
licensed; see [`LICENSE.md`](LICENSE.md).
