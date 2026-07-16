# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal OpenVDB -> dense ``.npy`` converter

Uses OpenVDB's single ``grid.copyToArray`` call over the active bounding box.
OpenVDB is optional (``conda install -c conda-forge openvdb``); only needed if
you train from ``.vdb`` volumes.

    python -m gfields.references.vdb --input cloud.vdb --output out_dir [--grid density]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import openvdb as _vdb
except ImportError:
    try:
        import pyopenvdb as _vdb
    except ImportError:
        _vdb = None


def read_vdb(path, grid_name=None) -> np.ndarray:
    """Read one grid of a ``.vdb`` file into a dense ``(X,Y,Z)`` float32 array."""
    if _vdb is None:
        raise ImportError("OpenVDB not installed. `conda install -c conda-forge openvdb` "
                          "to convert .vdb volumes.")
    grids, _ = _vdb.readAll(str(path))
    if not grids:
        raise ValueError(f"No grids found in {path}")
    grid = grids[0] if grid_name is None else {g.name: g for g in grids}[grid_name]
    bmin, bmax = grid.evalActiveVoxelBoundingBox()
    shape = tuple(int(bmax[i] - bmin[i] + 1) for i in range(3))
    arr = np.zeros(shape, dtype=np.float32)
    grid.copyToArray(arr, ijk=tuple(int(c) for c in bmin))
    return arr


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gfields.references.vdb", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="Output directory for the .npy")
    ap.add_argument("--grid", default=None, help="Grid name (default: first grid)")
    args = ap.parse_args(argv)
    arr = read_vdb(args.input, args.grid)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"{args.input.stem}.npy"
    np.save(out, arr)
    print(f"Wrote {out} {arr.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
