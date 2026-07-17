# Source volumes for training

Most of the dense source volumes used by the training presets are **not shipped**
with this repository (they are tens of MB to several GB each). Download them from
their original, freely-available sources and drop the resulting grids into

```
data/volumes/grids/
```

The one exception is **`smoke.vol`** (the smoke plume used in the paper), which is bundled at `data/volumes/smoke.vol` 

The training presets in [`gfields/presets.py`](../../gfields/presets.py) expect
the following file names:

| Preset     | Expected file                    | Source                          |
| ---------- | -------------------------------- | ------------------------------- |
| `smoke`    | `data/volumes/smoke.vol`         | **bundled** (paper smoke plume) |
| `wdas8`    | `data/volumes/grids/wdas8.npy`   | Disney Cloud dataset (see below) |
| `embergen` | `data/volumes/grids/embergen.npy`| JangaFX free VDB animations     |
| `explosion`| `data/volumes/grids/explosion.npy`| JangaFX free VDB animations    |
| `fire`     | `data/volumes/grids/fire.npy`    | JangaFX free VDB animations     |
| `smokev2`  | `data/volumes/grids/smokev2.npy` | JangaFX free VDB animations     |
| `tornado`  | `data/volumes/grids/tornado.npy` | JangaFX free VDB animations     |
| `dust`     | `data/volumes/grids/dust.npy`    | JangaFX free VDB animations     |

## Disney Cloud (`wdas8`)

The Walt Disney Animation Studios "Clouds" dataset (CC BY-SA 3.0):

<https://disneyanimation.com/resources/clouds/>

Download the OpenVDB cloud pack and pick a downsampled version of the full cloud (we use the 1/8th, hence
`wdas8`).

## JangaFX free VDB animations

JangaFX published a large pack of CC0 vdb data:

<https://jangafx.com/software/embergen/download/free-vdb-animations/>

Grab the volume you want, pick some frame (these are animation sequences),
and convert the `.vdb` to a dense grid.

## Converting a `.vdb` (or Mitsuba `.vol`) into a voxel grid

Two options, both produce the dense `.npy` grid the trainer consumes:

1. **Let the trainer convert on first use.** `resolve_volume()` in
   `gfields/presets.py` auto-converts `.vol` / `.vdb` to a cached `.npy`, so you
   can point training straight at the source file:

   ```bash
   examples/train.sh data/volumes/grids/wdas8.vdb --opacity_scale 12
   ```

2. **Convert explicitly** with the bundled OpenVDB to dense-grid script:

   ```bash
   python -m gfields.references.vdb \
       --input  data/volumes/grids/wdas8.vdb \
       --output data/volumes/grids/            # writes <name>.npy here
       # --grid NAME   # optional: pick a named grid (default: first grid)
   ```

   VDB conversion needs the OpenVDB Python bindings:

   ```bash
   conda install -c conda-forge openvdb
   ```

Once `data/volumes/grids/<name>.npy` exists, the matching preset, if desired, works directly:

```bash
examples/train.sh wdas8
```
