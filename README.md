# Gabor Fields: Orientation-Selective Level-of-Detail for Volume Rendering

![Teaser: a smoke bunny reconstructed as a Gabor Field, shown at four
continuous levels of detail (512 → 33,280 Gabor kernels) with per-level optical
depth and power spectrum.](data/figures/teaser.png)

Project Page: https://arcanous98.github.io/projectPages/gaborVolumes.html 

Reference implementation of **Gabor Fields: Orientation-Selective
Level-of-Detail for Volume Rendering** 
Jorge Condor*, Nicolai Hermann*, Mehmet Ata Yurtsever, Piotr Didyk; (USI Lugano, Switzerland)
Transactions on Graphics, SIGGRAPH 2026). 

A **Gabor Field** is a continuous
volumetric representation built as a mixture of a low-pass **Gaussian** base
plus a residual of anisotropic **Gabor** kernels (a Gaussian envelope times a
harmonic carrier i.e., a cosine wave). Because each Gabor
occupies a known frequency band and orientation, this is a primitive-volume
analogue of a Laplacian / steerable pyramid, but one which can be decomposed
arbitrarily at runtime in a continuous manner:

- **Continuous, no-cost level of detail**. Filtering a band = dropping
  primitives. No re-fitting, no extra storage, no popping.
- **Stochastic Laplacian path tracing**. Use the Gaussians (~2% of the total set) as a control variate volume, and then stochastically select which Gabors to interact with based on your needs. Tremendous acceleration is possible in many scenarios (long paths, next event estimation) for small prices in terms of bias.
- **Orientation-selective culling**. Gabors are selective in frequencies and orientations. A ray can skip Gabors which will not contribute to the field before even tracing against them.
- **Foveated rendering, motion blur, and procedural clouds** from the same asset. See more in our paper. 

This repo is based on
[facebookresearch/volumetric_primitives](https://github.com/facebookresearch/volumetric_primitives)
(*Don't Splat your Gaussians*, Condor et al., SIGGRAPH 2025). 

We add the following new features:

(SUMMARIZE: )

---

## Installation

The one-shot installer builds a reproducible [uv](https://docs.astral.sh/uv/)
environment in `.venv`, **compiles the vendored Mitsuba 3 fork against it**, and
wires the build in so `import mitsuba` always matches source. Prerequisites:
`uv`, `cmake`, a C++ compiler (MSVC/VS 2022 on Windows; gcc/clang + Ninja on
Linux), and an NVIDIA driver for the `cuda_ad_rgb` variant.

```bash
git clone --recursive https://github.com/Arcanous98/gabor_fields
cd gabor_fields
./install.sh                 # Windows PowerShell:  .\install.ps1
#   ./install.sh --extras    # + torch / lpips / wandb / metrics (needed for the voxelizer)
```

> The fork ([Arcanous98/mitsuba3_gfields](https://github.com/Arcanous98/mitsuba3_gfields))
> adds three small C++ features: per-ray visibility masking (`ray.mask` + IAS
> pruning), the `is_gabor` shape flag, and per-primitive `extent` tensors. Build
> from source; a stale prebuilt binary shows up as
> `ellipsoidsmesh ... visibility_mask has the wrong type`.

> **Extras:** the voxelizer (`gfields voxelize`) requires `torch`; install it
> with `./install.sh --extras` (or `uv pip install --python .venv torch`).

Validate:

```bash
python -m pytest gfields/tests/test_gfields.py -q     # 10 core tests
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); import gfields; print('OK')"
```

---

## The scripts

Thin launchers in `examples/` (`.sh` for Linux/macOS/Git-Bash, `.ps1` for
PowerShell) locate `.venv` and forward every extra flag verbatim (later flags
win over presets). The same commands work directly as `python -m gfields.*`.

| Task | Launcher | Module |
|---|---|---|
| Train (Gaussian base + Gabor residual) | `examples/train.{sh,ps1}` | `python -m gfields.train` |
| Render (tomo / MS / video) | `examples/render.{sh,ps1}` | `python -m gfields.render` |
| Interactive viewer | `examples/view.{sh,ps1}` | `gfields viewer` |
| Decompose / clamp / prune | — | `gfields decompose {steerable,lod-partition,clamp,prune}` |
| Voxelize (needs `--extras` torch) | — | `gfields voxelize {asset,compare,grid}` |

### Train

```bash
# paper presets (framing / capacity / density auto-configured):
#   wdas8 embergen explosion fire smoke smokev2 tornado dust
examples/train.sh wdas8                         # champion recipe, 1024 Gauss + 32768 Gabor
examples/train.sh wdas8 --gauss_count 512 --gabor_count 4096   # smaller budget
examples/train.sh wdas8 --gaussian              # Gaussian-only baseline

# your own volume (.npy/.vol/.vdb) — give it a density scale:
examples/train.sh data/volumes/grids/mine.npy --opacity_scale 12
```

The paper's source grids are not shipped (tens of MB–GB each); download them
into the git-ignored `data/volumes/grids/` — see
[`data/volumes/asset_sources.md`](data/volumes/asset_sources.md) for the
Disney cloud, the free JangaFX volumes, and how to convert `.vdb` grids.

Optimization is **tomographic** (absorption-only) and defaults to the OptiX
**any-hit** primal fast path (`--no_anyhit` to disable). Post-training refinement
is opt-in: `--post_clamp` (adaptive extents) and `--post_prune`
(`--post_prune_budget 0.1` dB). Output → `results/checkpoints/<asset>/`.

### Render

```bash
python -m gfields.render <asset>                                   # tomography (ortho)
python -m gfields.render <asset> --use_perspective_cam             # perspective
python -m gfields.render <asset> --multiple_scattering --envmap sky.exr --spp 64
python -m gfields.render <asset> --multiple_scattering --stochastic_sampling   # prb_stochastic
python -m gfields.render video <asset> --motion turntable --frames 120         # MP4 via ffmpeg
python -m gfields.render video <asset> --motion envmap   --frames 120
```

Key integrator flags (independent axes, all validated): `--max_depth`,
`--rr_depth`, `--rr_nee_depth`, `--max_overlaps`,
`--solver_type {bisection,newton,disabled}`, `--use_anyhit` / `--no_anyhit`,
`--adaptive_extent`, `--stochastic_selection_criteria[_nee]`
(`deterministic` / `power_law` / `control_variate_power_law`),
`--orientation_selection[_nee]` (`deterministic` / `threshold` / `importance` /
`control_variate_threshold`, steerable assets), `--nee_transmittance
{analytic,stochastic}`. Full list: `python -m gfields.render --help`.

### Decompose (LOD assets)

```bash
gfields decompose steerable --asset <asset> --output <out> --num_orientations 3
gfields decompose lod-partition --path <checkpoint> --output <out>   # frequency-band Laplacian pyramid
gfields decompose clamp <asset> --level 1                            # adaptive per-primitive extents
gfields decompose prune  -p <asset> -r opacity -b 0.5 -a             # PSNR-budget pruning
```

### View / voxelize

Three trained Gabor Fields ship in `data/examples/` (`wdas8` — the Disney
cloud, `smoke`, `tornado`) so the viewer and renderer work straight after a
clone; `<asset>` below is any of those or your own `results/checkpoints/<name>`.

```bash
gfields viewer                                      # bundled wdas8 cloud + env map
gfields viewer data/examples/tornado --envmap sky.exr --integrator prb
gfields voxelize compare --asset data/examples/wdas8 --grid_res 256 256 256 --tomography_only
```

---

## Results to expect (RTX 4090)

Measured in this repo's validation campaign (`results/refactor_validation/`).
Training is tomographic with the any-hit fast path; render times are single-view
GPU **execution** time.

### Training (quality + wall-clock)

Full-asset tomography PSNR (64 spp, all 32 training views) vs the 8192-spp
reference, 600 iterations:

| Asset | 512 + 4 096 | 1 024 + 32 768 | Time (champion) |
|---|---|---|---|
| wdas8 (Disney cloud, 8th) | 41.8 dB | **48.3 dB** | ~22 min |
| smoke | 44.4 dB | 45.5 dB | ~10 min |
| tornado | 38.7 dB | 45.7 dB | ~15 min |

The champion budget lands ~45–48 dB at 600 iterations (wdas8 ≈ the paper's
48.7 dB; 900 iterations → ~49.6 dB). Sparse budgets (few thousand Gabors) fall a
few dB short with the default schedule because the Gaussian base is left
under-dense — raise `--init_opacity ~5e-4 --opacity_lr ~1e-3` (the trainer
prints a `[small-count]` hint). The recipe transfers across assets with no
retuning, though per-asset presets are only quality-verified for the champion
budgets.

### Rendering (256×256, 16 spp, single view)

| Mode | Quality (vs 256-spp ref) | Exec time |
|---|---|---|
| Tomography (ortho) | 61.8 dB | ~4 ms |
| Tomography + any-hit | 61.4 dB | **~2 ms** (2× faster, lossless) |
| Multiple scattering (env map, HG) | 52.5 dB | ~420 ms |
| MS, CDF solver `bisection`/`newton`/`disabled` | all 52.45 dB (identical) | 210 / 435 / 363 ms |
| Stochastic-Laplacian `power_law` selection | mean-preserving | **~2.5× faster** |
| Orientation `importance` selection (steerable) | mean-preserving | **~2.6× faster** |

`max_depth`, Russian-roulette, and `max_overlaps` trade quality for speed as
expected. Adaptive extents are lossless; the three CDF solvers agree exactly.

### Post-processing

| Pass | Effect |
|---|---|
| PSNR-budget prune (0.5 dB) | −12% primitives, −0.5 dB (as budgeted) |
| LOD / Laplacian partition | 4 096 Gabors → 3 frequency bands; recomposition **123.9 dB** (lossless) |
| Adaptive extent clamp | Gabor bounds tighten to ~1.6σ mean (faster traversal, lossless) |

### Compression (from the paper)

A 33 k-kernel Gabor Field is **~1.8 MB** on disk versus a 1024³ float32 grid at
~4.3 GB — a **~2000× compression**. The Disney cloud at 24 k Gabors is ~1.03 MB.
Stochastic-analytic estimators accelerate full multiple scattering **~2×** with
little bias.

---

## Integrators

| Integrator | Purpose |
|---|---|
| `gfields_tomography` | Absorption-only / optical-depth training and tomographies; static + stochastic LOD, steerable orientation selection, continuous frequency filtering. |
| `gfields_prb` | Volumetric path tracing (path-replay backprop): emitters, NEE, HG / SGGX phase functions, stochastic Laplacian, steerable sub-bands, continuous LOD. |
| `gfields_prb_stochastic` | Decomposition-tracking distance sampling (min-of-free-flights + envelope delta tracking). |

With default properties each reproduces its upstream baseline. Every option is
exposed as a plugin property (`mi.traverse`) and via `python -m gfields.render
--help`.

---

## Tests

```bash
python -m pytest gfields/tests/test_gfields.py -q     # closed-form integrals, LOD parity, stochastic vs analytic
python gfields/tests/test_gfields.py                  # plain runner (no pytest)
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
