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

---

## Install

```bash
git clone --recursive https://github.com/Arcanous98/gabor_fields
cd gabor_fields
./install.sh                 # Windows PowerShell:  .\install.ps1
#   ./install.sh --extras    # + torch / lpips / wandb / metrics (needed for the voxelizer)
```

This builds a reproducible [uv](https://docs.astral.sh/uv/) `.venv` and
**compiles the vendored Mitsuba 3 fork against it**
([Arcanous98/mitsuba3_gfields](https://github.com/Arcanous98/mitsuba3_gfields),
which adds per-ray visibility masking, the `is_gabor` shape flag, and
per-primitive `extent` tensors). Needs `uv`, `cmake`, a C++ compiler
(MSVC/VS 2022 on Windows; gcc/clang + Ninja on Linux), and an NVIDIA driver.

Validate:

```bash
python -m pytest gfields/tests/test_gfields.py -q
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import gfields; print('OK')"
```

---

## Train

```bash
examples/train.sh                     # bundled smoke asset, 512 Gauss + 4096 Gabor
examples/train.sh wdas8                # a named preset (needs its volume, see below)
examples/train.sh data/volumes/grids/mine.npy --opacity_scale 12   # your own volume
examples/train.sh --gaussian           # Gaussian-only baseline
```

Optimization is **tomographic** (absorption-only) and defaults to the OptiX
**any-hit** fast path for the primal pass — the main reason training is fast
here (`--no_anyhit` to fall back to the overlap-list walk). Tune primitive
budget and density from the CLI (`--gauss_count`, `--gabor_count`,
`--opacity_scale`) — see [`examples/README.md`](examples/README.md) for good
combos and how presets/recipe live in `gfields/presets.py`. Named presets
(`wdas8`, `embergen`, `explosion`, `fire`, `smoke`, `smokev2`, `tornado`,
`dust`) resolve to volumes under `data/volumes/grids/`, downloaded per
[`data/volumes/asset_sources.md`](data/volumes/asset_sources.md); only
`smoke` ships with the repo.

Output → `results/checkpoints/<asset>/`.

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
takes the minimum — no acceptance test, no overlap bookkeeping. Combine with
`--stochastic_selection_criteria power_law` / `--orientation_selection
importance` (steerable assets) for the Laplacian/orientation speedups
described above.

Full flag list: `python -m gfields.render --help` (and `... video --help`).

---

## Visualize

Three trained Gabor Fields ship in `data/examples/` (`wdas8`, `smoke`,
`tornado`) so the viewer works straight after a clone:

```bash
gfields viewer                                      # bundled wdas8 cloud + env map
gfields viewer data/examples/tornado --envmap sky.exr --integrator prb
```

Interactive polyscope viewer with progressive accumulation and an LOD slider;
launcher: `examples/view.{sh,ps1}`.

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

`.vol` grids convert automatically on first use (no separate tool needed).

---

## Tests

```bash
python -m pytest gfields/tests/test_gfields.py -q     # closed-form integrals, LOD parity, stochastic vs analytic
```

Auto-skips without a `cuda_ad_rgb` build.

---

## BibTeX

```bibtex
@article{10.1145/3811369,
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
@article{condor2025volprim,
    author  = {Condor, Jorge and Speierer, Sebastien and Bode, Lukas and Bozic, Aljaz and Green, Simon and Didyk, Piotr and Jarabo, Adrian},
    title   = {Don't Splat your Gaussians: Volumetric Ray-Traced Primitives for Modeling and Rendering Scattering and Emissive Media},
    journal = {ACM Trans. Graph.}, year = {2025}, doi = {10.1145/3711853},
}
```

## Acknowledgements

Funded by the Swiss National Science Foundation (SNSF, Grant 200502) and an
academic gift from Meta, with access to Alps at CSCS (project ID u6). MIT
licensed; see [`LICENSE.md`](LICENSE.md).
