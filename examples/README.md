# Examples

Three self-contained launchers wrap the example workflows. Use the `.sh`
variant on Linux/macOS/Git-Bash and the `.ps1` variant on PowerShell — each
shim only locates a python interpreter (`.venv` from the top-level installer,
else conda, else `$VOLPRIM_PYTHON`) and delegates to the `gfields` package.

```bash
./view.sh   <asset> [options...]                 # interactive polyscope viewer
./train.sh  <preset|volume> [options...]         # train a volumetric asset
./render.sh {tomo|ms|video} <asset> [options...] # render a trained asset
```

| Launcher | Delegates to | Modes |
|---|---|---|
| `view`   | `python -m gfields viewer`  | Interactive viewer (progressive accumulation, LOD slider) |
| `render` | `python -m gfields.render`  | `tomo` (constant light), `ms` (env-map + NEE), `video` (turntable / rotating envmap), `asset` (raw, no preset) |
| `train`  | `python -m gfields.train`   | Gaussian base **+** Gabor residual (the paper recipe); add `--gaussian` for a Gaussian-only baseline |

Installation lives **outside** this folder: run the top-level
[`install.ps1`](../install.ps1) / [`install.sh`](../install.sh), which builds the
uv `.venv`, installs `gfields` editable, and compiles + wires the Mitsuba fork via
[`gfields/build_mitsuba.py`](../gfields/build_mitsuba.py).

The render/view commands take a trained-asset folder (an experiment dir with
`optimized_asset_pyr*` is auto-resolved). The train command takes a paper-asset
preset name or a volume file (`.npy`, Mitsuba `.vol`, or OpenVDB `.vdb` — the
last two are converted and cached on first use; VDB conversion needs the
`openvdb` Python bindings, e.g. `conda install -c conda-forge openvdb`).

Every launcher forwards unrecognized flags verbatim to the underlying tool, and
forwarded flags **override** the baked-in recipe/preset defaults (argparse keeps
the last occurrence) — so any parameter can be tuned from here. See
`python -m gfields.train --help`, `python -m gfields.render --help`
(and `... video --help`), and `gfields viewer --help` for the full flag lists.

```bash
# Render / view a bundled example asset (data/examples/{wdas8,smoke,tornado}):
./view.sh   data/examples/wdas8
./render.sh ms  data/examples/wdas8 --spp 256 --envmap my_env.exr
./render.sh ms  data/examples/wdas8 --stochastic_sampling   # gfields_prb_stochastic integrator
./render.sh video data/examples/wdas8 --motion envmap --frames 240

# With no target at all, trains the bundled smoke asset (512 Gaussians +
# 4096 Gabors, --opacity_scale 10.0):
./train.sh
./train.sh --gaussian

# Train a named paper asset (same recipe/counts/density as above, only the
# volume file differs). The source grid must live in data/volumes/grids/ —
# see data/volumes/asset_sources.md for where to download it:
./train.sh wdas8

# Train your own volume:
./train.sh data/volumes/grids/mine.npy

# Override anything — later flags win over the recipe:
./train.sh wdas8 --gauss_count 2048 --gabor_count 32768 --opacity_scale 20
```

- **Output** defaults to `results/renders/<mode>` (renders) and
  `results/checkpoints/<asset>[_gaussian]` (training); override with `--output`.
- **Interpreter**: set `VOLPRIM_PYTHON=/path/to/python` to override detection.
- **W&B** logging is disabled by default; opt in with `WANDB_MODE=online`.

## Where the presets live

The recipe, named-asset volume lookup, render-mode presets, and volume-file
conversion all live in [`gfields/presets.py`](../gfields/presets.py) — the
single source of truth. `python -m gfields.train [preset|volume]` and
`python -m gfields.render {tomo|ms}` pull their defaults from there, so the
launchers stay thin and the same behavior is available without them. With no
target, training defaults to the bundled `smoke` asset (`DEFAULT_ASSET` in
`gfields/presets.py`).

## The winning training recipe (defaults)

Two settings are non-obvious but critical:

1. **Per-parameter learning rates are set explicitly.** The CLI default of
   `1.0 × global_lr` is ~100–100000× too high and plateaus quality near 39 dB.
   The recipe uses `global_lr 4.0` × `{centers 0.003, scales 0.0002,
   quats 0.0003, opacity 0.00002, omega 0.0005}`.
2. **Primitive size is hard-capped** (`--limit_max_scale --max_scale_mult 0.5`).
   Tighter primitives are *both* ~6× faster per step *and* higher quality than
   a few oversized, over-smoothing ones.

Other defaults: 32 cameras, `ref_spp 8192`, `opt_spp 2`, Gaussian base at
pyramid level 3 → Gabor residual at level 0, 240 + 1200 iterations, importance-
sampled means, single-hemisphere cameras, padded grid, no L1 opacity
regularization (light L1 scale regularization), Langevin noise
`init 50 / decay 0.75`. Measured on `wdas8`: ~48.7 dB in ~1 h GPU; it transfers
to the other assets with no retuning.

## Primitive counts and density

The only knobs that meaningfully differ across volumes are primitive
**counts** and the **density** (`--opacity_scale`), so the recipe bakes in a
single default for both — `--gauss_count 512 --gabor_count 4096
--opacity_scale 10.0` — rather than a per-asset preset table. Named assets
(`TRAIN_ASSETS` in `gfields/presets.py`) are just a shorthand mapping a short
name to its volume-grid path; tune counts/density per run from the CLI:

```bash
./train.sh wdas8 --gauss_count 1024 --gabor_count 24576 --opacity_scale 20
```

Other known-good `(gauss, gabor)` combos, roughly increasing quality/cost:
`512 + 8192`, `1024 + 24576`, `2048 + 32768`. `--opacity_scale 10.0` is a
reasonable middle-of-the-road default; paper assets individually used values
from 7.5 to 70 (see `gfields.training.configs`) — nudge it down if a render
looks washed out, or up if it looks too opaque.

The geometric **`to_world` transform** (e.g. the Disney cloud's non-uniform
box) is still picked from the volume path automatically by
`gfields.training.configs`; `--opacity_scale` always overrides that config's
default density.

## Training your own volume

1. Drop a dense grid in `data/volumes/grids/` (`.npy`, `.vol`, or `.vdb`). This
   directory is git-ignored; see
   [`data/volumes/asset_sources.md`](../data/volumes/asset_sources.md) for the
   paper volumes and how to convert `.vdb` grids.
2. Run the launcher on the file path (the recipe's default `--opacity_scale
   10.0` and generic isotropic 2-unit-cube transform apply automatically):

   ```bash
   ./train.sh data/volumes/grids/mine.npy --opacity_scale 15 \
       --cam_res_x 512 --cam_res_y 512 --output results/checkpoints/mine
   ```

If your volume is non-cubic and needs a specific `to_world`, add a
`VolumeConfig` subclass in `gfields/training/configs.py` (match on a substring
of the filename) — then the launcher picks it up automatically.
