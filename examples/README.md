# Examples

Three self-contained launchers wrap the example workflows. Use the `.sh`
variant on Linux/macOS/Git-Bash and the `.ps1` variant on PowerShell.
```bash
./view.sh   <asset> [options...]                 # interactive polyscope viewer
./train.sh  <preset|volume> [options...]         # train a volumetric asset
./render.sh {tomo|ms|video} <asset> [options...] # render a trained asset
```

| Launcher | Delegates to | Modes |
|---|---|---|
| `view`   | `python -m gfields viewer`  | Interactive viewer (progressive accumulation, LOD slider) |
| `render` | `python -m gfields.render`  | `tomo` (constant light), `ms` (env-map + multiple scattering), `video` (turntable / rotating envmap), `asset` (path) |
| `train`  | `python -m gfields.train`   | Gaussian base **+** Gabor residual. Add `--gaussian` for training only Gaussian kernels. |

Every launcher forwards unrecognized flags verbatim to the underlying tool. See
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

# Train a named paper asset. The source grid must live in data/volumes/grids/ —
# see data/volumes/asset_sources.md for where to download it:
./train.sh wdas8

# Train your own volume:
./train.sh data/volumes/grids/mine.npy

# Override anything
./train.sh wdas8 --gauss_count 2048 --gabor_count 32768 --opacity_scale 20
```

- **Output** defaults to `results/renders/<mode>` (renders) and
  `results/checkpoints/<asset>[_gaussian]` (training); override with `--output`.
- **W&B** logging is disabled by default; opt in with `WANDB_MODE=online`

## Primitive counts and density

The only knobs that meaningfully differ across volumes are primitive
**counts** and the **density** (`--opacity_scale`). 

```bash
./train.sh wdas8 --gauss_count 1024 --gabor_count 24576 --opacity_scale 20
```

## Training your own volume

1. Drop a dense grid in `data/volumes/grids/` (`.npy`, `.vol`, or `.vdb`). 
2. Run the launcher on the file path (the our default `--opacity_scale is
   10.0` and assume a bounding cube):

   ```bash
   ./train.sh data/volumes/grids/mine.npy --opacity_scale 15 \
       --cam_res_x 512 --cam_res_y 512 --output results/checkpoints/mine
   ```

If your volume is non-cubic and needs a specific `to_world`, add a
`VolumeConfig` subclass in `gfields/training/configs.py`,  then the launcher picks it up automatically.
