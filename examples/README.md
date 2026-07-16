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

# Train a paper asset preset (framing / capacity / density auto-configured).
# The source grid must live in data/volumes/grids/ — see
# data/volumes/asset_sources.md for where to download it:
./train.sh wdas8
./train.sh wdas8 --gaussian

# Train your own volume (density scale required for unknown volumes):
./train.sh data/volumes/grids/mine.npy --opacity_scale 12

# Override anything — later flags win over the recipe and the preset:
./train.sh wdas8 --iterations_gabor 900 --cam_res_x 1024 --cam_res_y 1024
```

- **Output** defaults to `results/renders/<mode>` (renders) and
  `results/checkpoints/<asset>[_gaussian]` (training); override with `--output`.
- **Interpreter**: set `VOLPRIM_PYTHON=/path/to/python` to override detection.
- **W&B** logging is disabled by default; opt in with `WANDB_MODE=online`.

## Where the presets live

The recipe, per-asset presets, render-mode presets, and volume-file conversion
all live in [`gfields/presets.py`](../gfields/presets.py) — the single source of
truth. `python -m gfields.train <preset|volume>` and
`python -m gfields.render {tomo|ms}` pull their defaults from there, so the
launchers stay thin and the same behavior is available without them.

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
pyramid level 3 → Gabor residual at level 0, 240 + 600 iterations, importance-
sampled means, single-hemisphere cameras, padded grid, no L1 scale/opacity
regularization, Langevin noise `init 50 / decay 0.75`. Measured on `wdas8`:
~48.7 dB in ~1 h GPU (900 iters → ~49.6 dB); it transfers to the other assets
with no retuning.

## Per-asset presets

Only genuinely asset-specific handling lives in the presets (`TRAIN_PRESETS`
in `gfields/presets.py`): camera framing + **resolution**, primitive **counts**,
reference **spp**, and any special base level. The **density (opacity) scale**
and the geometric **`to_world` transform** (e.g. the Disney cloud's non-uniform
box, the tornado) are picked from the volume path automatically by
`gfields.training.configs`.

| asset | cam res | cam_scale | y_offset | gauss | gabor | ref_spp | notes |
|---|---|---|---|---|---|---|---|
| wdas8 (cloud) | 512×512 | 0.95 | −0.25 | 1024 | 32768 | 8192 | champion reference |
| embergen | 512×768 | 0.70 | 0.00 | 512 | 24064 | 8192 | |
| explosion | 512×512 | 1.00 | 0.00 | 1024 | 23552 | 8192 | |
| fire | 224×512 | 0.50 | −0.10 | 512 | 24064 | 16384 | |
| smoke | 256×512 | 0.50 | 0.00 | 512 | 15872 | 16384 | |
| smokev2 | 400×1000 | 0.35 | −0.12 | 512 | 24064 | 8192 | |
| tornado | 512×768 | 0.55 | −0.25 | 512 | 11520 | 8192 | base at level 2 |
| dust | 224×512 | 0.50 | −0.10 | 512 | 24064 | 16384 | Gabor count untuned |

Quality-verified with this recipe: `wdas8`, `embergen`, `explosion`, `fire`.
The rest inherit the same recipe for consistency (framing/capacity preserved) —
re-verify if you depend on exact numbers.

## Training your own volume

1. Drop a dense grid in `data/volumes/grids/` (`.npy`, `.vol`, or `.vdb`). This
   directory is git-ignored; see
   [`data/volumes/asset_sources.md`](../data/volumes/asset_sources.md) for the
   paper volumes and how to convert `.vdb` grids.
2. Run the launcher on the file path and give it a density scale:

   ```bash
   ./train.sh data/volumes/grids/mine.npy --opacity_scale 15 \
       --cam_res_x 512 --cam_res_y 512 --output results/checkpoints/mine
   ```

`--opacity_scale` is **required** for a volume with no named config; it selects
the generic isotropic 2-unit-cube transform. If your volume is non-cubic and
needs a specific `to_world`, add a `VolumeConfig` subclass in
`gfields/training/configs.py` (match on a substring of the filename) —
then the launcher picks it up with no `--opacity_scale` needed.
