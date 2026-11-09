#!/usr/bin/env python3
"""Convert image slices or a compact NPY to a centered, zero-padded NPY cube.

The model reads volumes in ``(Z, Y, X)`` order. This exporter optionally
resamples the source stack and downsamples each slice, then places the result
at the center of a float32 zero cube. By default the source stack is assigned
to world Z. Use ``--stack-axis y`` when the slice sequence should run vertically
in world space instead.

When ``--cube-size`` is omitted, a conservative cube edge is calculated from
the GPU-memory budget. The defaults model four simultaneously resident
float32 volume-sized allocations and leave additional memory for Mitsuba,
Dr.Jit, rendering, and temporary buffers.

Example (from the repository root, for a 24 GiB RTX 4090):

    python data/volumes/images_to_npy.py LS-20261008 \
        --output data/volumes/LS-20261008_density_cube.npy \
        --gpu-memory-gib 24

Use ``--plan-only`` first to print the selected dimensions without writing the
potentially multi-gigabyte output file.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_PATTERN = "22_cc_Area_*.png"
INDEX_PATTERN = re.compile(r"(\d+)$")
GIB = 1024**3
FLOAT32_BYTES = np.dtype(np.float32).itemsize


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def fraction(value: str) -> float:
    result = float(value)
    if not 0 < result <= 1:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return result


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path", nargs="?", type=Path, default=script_dir,
        help="Directory containing image slices, or a compact (Z,Y,X) .npy file",
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN,
        help=f"Image glob pattern (default: {DEFAULT_PATTERN!r})",
    )
    parser.add_argument(
        "--output", type=Path, default=script_dir / "volume_cube.npy",
        help="Output .npy path",
    )
    parser.add_argument(
        "--gpu-memory-gib", type=positive_float, default=24.0,
        help="Installed GPU memory in GiB (default: 24)",
    )
    parser.add_argument(
        "--gpu-reserve-gib", type=float, default=4.0,
        help="Memory kept completely free for non-volume allocations (default: 4)",
    )
    parser.add_argument(
        "--memory-utilization", type=fraction, default=0.8,
        help="Usable fraction of memory remaining after the reserve (default: 0.8)",
    )
    parser.add_argument(
        "--resident-volume-copies", type=positive_float, default=4.0,
        help="Estimated simultaneous float32 volume-sized allocations (default: 4)",
    )
    parser.add_argument(
        "--alignment", type=int, default=8,
        help="Round the automatic cube edge down to this multiple (default: 8)",
    )
    parser.add_argument(
        "--cube-size", type=int,
        help="Explicit cube edge; overrides automatic GPU-budget sizing",
    )
    parser.add_argument(
        "--xy-scale", type=fraction, default=1.0,
        help="Optional additional XY scale (default: 1)",
    )
    parser.add_argument(
        "--z-slices", type=int,
        help="Linearly resample the source stack to this many slices (default: preserve it)",
    )
    parser.add_argument(
        "--stack-axis", choices=("z", "y"), default="z",
        help="World axis for the source slice sequence: z=depth, y=vertical (default: z)",
    )
    parser.add_argument(
        "--reverse-z", action="store_true",
        help="Reverse source slice order before Z interpolation",
    )
    parser.add_argument(
        "--flip-y", action="store_true",
        help="Flip every resized slice vertically (reverse the Y axis)",
    )
    parser.add_argument(
        "--flip-x", action="store_true",
        help="Flip every resized slice horizontally (reverse the X axis)",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print dimensions and memory estimates without creating the NPY",
    )
    return parser.parse_args()


def numbered_images(input_dir: Path, pattern: str) -> list[tuple[int, Path]]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    indexed: list[tuple[int, Path]] = []
    seen: dict[int, Path] = {}
    for path in input_dir.glob(pattern):
        if not path.is_file():
            continue
        match = INDEX_PATTERN.search(path.stem)
        if match is None:
            raise ValueError(f"Cannot read a numeric slice index from: {path.name}")
        index = int(match.group(1))
        if index in seen:
            raise ValueError(
                f"Duplicate slice index {index}: {seen[index].name}, {path.name}"
            )
        seen[index] = path
        indexed.append((index, path))

    if not indexed:
        raise ValueError(f"No images matching {pattern!r} in {input_dir}")
    indexed.sort(key=lambda item: item[0])

    actual = [index for index, _ in indexed]
    expected = list(range(actual[0], actual[-1] + 1))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        raise ValueError(f"Slice indices are not contiguous; missing: {missing}")
    return indexed


def budget_cube_edge(args: argparse.Namespace) -> tuple[int, float]:
    if args.gpu_reserve_gib < 0:
        raise ValueError("--gpu-reserve-gib cannot be negative")
    if args.gpu_reserve_gib >= args.gpu_memory_gib:
        raise ValueError("--gpu-reserve-gib must be smaller than --gpu-memory-gib")
    if args.alignment <= 0:
        raise ValueError("--alignment must be greater than zero")

    usable_bytes = (
        (args.gpu_memory_gib - args.gpu_reserve_gib)
        * args.memory_utilization
        * GIB
    )
    max_voxels = int(
        usable_bytes // (FLOAT32_BYTES * args.resident_volume_copies)
    )
    # Correct the floating-point cube-root estimate so exact powers such as
    # 1024**3 are not accidentally rounded down to 1023.
    raw_edge = round(max_voxels ** (1.0 / 3.0))
    while (raw_edge + 1) ** 3 <= max_voxels:
        raw_edge += 1
    while raw_edge**3 > max_voxels:
        raw_edge -= 1
    edge = raw_edge - raw_edge % args.alignment
    if edge <= 0:
        raise ValueError("GPU budget is too small for the requested alignment")
    return edge, usable_bytes


def gib(byte_count: float) -> float:
    return byte_count / GIB


def print_plan(
    *, args: argparse.Namespace,
    source_shape: tuple[int, int, int],
    resized_shape: tuple[int, int, int],
    cube_size: int,
    budget_edge: int,
    usable_bytes: float,
) -> None:
    z, source_h, source_w = source_shape
    _, resized_h, resized_w = resized_shape
    cube_voxels = cube_size**3
    cube_bytes = cube_voxels * FLOAT32_BYTES
    estimated_resident = cube_bytes * args.resident_volume_copies
    occupied = resized_shape[0] * resized_h * resized_w

    print("Volume export plan")
    print(f"  source shape (Z,Y,X):       {source_shape}")
    print(f"  resampled stack shape:      {resized_shape} (stack,row,column)")
    print(
        f"  effective XY scale:         "
        f"{resized_h / source_h:.6f} x {resized_w / source_w:.6f}"
    )
    print(f"  automatic budget edge:      {budget_edge}")
    print(f"  selected output cube:       ({cube_size}, {cube_size}, {cube_size})")
    print(f"  occupied voxel fraction:    {occupied / cube_voxels:.4%}")
    print(f"  output NPY data size:       {gib(cube_bytes):.3f} GiB")
    print(
        f"  estimated {args.resident_volume_copies:g}x volume memory: "
        f"{gib(estimated_resident):.3f} GiB"
    )
    print(f"  modeled usable GPU memory:  {gib(usable_bytes):.3f} GiB")
    print(
        "  GPU model:                  "
        f"({args.gpu_memory_gib:g} - {args.gpu_reserve_gib:g}) GiB "
        f"x {args.memory_utilization:g} utilization"
    )
    z_policy = (
        f"linear interpolation to {resized_shape[0]} slices"
        if resized_shape[0] != source_shape[0]
        else "preserve source slice count"
    )
    if args.reverse_z:
        z_policy += ", reversed"
    print(f"  Z policy:                   {z_policy}")
    print(f"  source stack world axis:    {args.stack_axis.upper()}")
    orientation = []
    if args.flip_y:
        orientation.append("flip Y")
    if args.flip_x:
        orientation.append("flip X")
    print(
        "  XY orientation:             "
        + (", ".join(orientation) if orientation else "unchanged")
    )
    if estimated_resident > usable_bytes:
        print("  WARNING: selected cube exceeds the modeled safe GPU budget")


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".npy":
        raise ValueError(f"Output must use the .npy extension: {output}")

    compact: np.ndarray | None = None
    slices: list[tuple[int, Path]] | None = None
    if input_path.is_file() and input_path.suffix.lower() == ".npy":
        compact = np.load(input_path, mmap_mode="r", allow_pickle=False)
        if compact.ndim != 3:
            raise ValueError(
                f"Compact NPY must have exactly 3 axes in (Z,Y,X) order; "
                f"got shape {compact.shape}"
            )
        source_z, source_h, source_w = map(int, compact.shape)
    else:
        slices = numbered_images(input_path, args.pattern)
        source_z = len(slices)
        with Image.open(slices[0][1]) as first:
            if first.mode != "L":
                raise ValueError(
                    f"Expected 8-bit grayscale images; {slices[0][1].name} "
                    f"has mode {first.mode!r}"
                )
            source_w, source_h = first.size

    budget_edge, usable_bytes = budget_cube_edge(args)
    source_shape = (source_z, source_h, source_w)
    source_max_xy = max(source_h, source_w)
    target_z = source_z if args.z_slices is None else args.z_slices
    if target_z <= 0:
        raise ValueError("--z-slices must be greater than zero")

    if args.cube_size is not None:
        if args.cube_size <= 0:
            raise ValueError("--cube-size must be greater than zero")
        cube_size = args.cube_size
    else:
        # Do not enlarge a small source merely because more GPU memory exists.
        desired_max_xy = max(1, round(source_max_xy * args.xy_scale))
        cube_size = min(desired_max_xy, budget_edge)

    if target_z > cube_size:
        raise ValueError(
            f"{target_z} output Z slices do not fit in a cube of edge {cube_size}. "
            "Use fewer --z-slices or a larger --cube-size."
        )

    fit_scale = min(1.0, cube_size / source_max_xy)
    xy_scale = min(args.xy_scale, fit_scale)
    resized_h = max(1, round(source_h * xy_scale))
    resized_w = max(1, round(source_w * xy_scale))
    resized_shape = (target_z, resized_h, resized_w)

    print_plan(
        args=args,
        source_shape=source_shape,
        resized_shape=resized_shape,
        cube_size=cube_size,
        budget_edge=budget_edge,
        usable_bytes=usable_bytes,
    )
    if args.plan_only:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output or partial file: {output}"
        )

    occupied_shape = (
        (target_z, resized_h, resized_w)
        if args.stack_axis == "z"
        else (resized_h, target_z, resized_w)
    )
    offsets = tuple((cube_size - size) // 2 for size in occupied_shape)
    z0, y0, x0 = offsets
    volume = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(cube_size, cube_size, cube_size),
    )
    try:
        # Fill in Z chunks so initialization never allocates another full cube.
        for start in range(0, cube_size, 16):
            volume[start : start + 16] = np.float32(0.0)

        target_size = (resized_w, resized_h)
        resized_stack = np.empty(
            (source_z, resized_h, resized_w), dtype=np.float32
        )
        for position in range(source_z):
            source_position = source_z - 1 - position if args.reverse_z else position
            if compact is not None:
                density = np.asarray(compact[source_position], dtype=np.float32)
                source_label = f"{input_path.name}[{source_position}]"
                if density.shape != (source_h, source_w):
                    raise RuntimeError(f"Unexpected NPY slice shape: {density.shape}")
                if density.shape != (resized_h, resized_w):
                    density = np.asarray(
                        Image.fromarray(density).resize(
                            target_size, resample=Image.Resampling.BOX
                        ),
                        dtype=np.float32,
                    )
            else:
                assert slices is not None
                index, path = slices[source_position]
                with Image.open(path) as image:
                    if image.mode != "L":
                        raise ValueError(
                            f"Expected 8-bit grayscale images; {path.name} "
                            f"has mode {image.mode!r}"
                        )
                    if image.size != (source_w, source_h):
                        raise ValueError(
                            f"Image size mismatch: {path.name} is {image.size}, "
                            f"expected {(source_w, source_h)}"
                        )
                    if image.size != target_size:
                        image = image.resize(
                            target_size, resample=Image.Resampling.BOX
                        )
                    density = (
                        np.asarray(image, dtype=np.float32) / np.float32(255.0)
                    )
                source_label = f"source index {index}: {path.name}"

            if not np.isfinite(density).all():
                raise ValueError(f"Non-finite density value in {source_label}")
            if args.flip_y:
                density = density[::-1, :]
            if args.flip_x:
                density = density[:, ::-1]
            resized_stack[position] = density
            print(
                f"Loaded source Z {position + 1:03d}/{source_z:03d} "
                f"({source_label})",
                flush=True,
            )

        minimum = float(resized_stack.min())
        maximum = float(resized_stack.max())
        source_positions = (
            np.linspace(0.0, source_z - 1, target_z, dtype=np.float64)
            if target_z > 1
            else np.array([(source_z - 1) * 0.5], dtype=np.float64)
        )
        for position, source_position in enumerate(source_positions):
            lower = int(math.floor(source_position))
            upper = min(lower + 1, source_z - 1)
            weight = np.float32(source_position - lower)
            if lower == upper or weight == 0:
                density = resized_stack[lower]
            else:
                density = (
                    resized_stack[lower] * (np.float32(1.0) - weight)
                    + resized_stack[upper] * weight
                )
            if args.stack_axis == "z":
                volume[
                    z0 + position,
                    y0 : y0 + resized_h,
                    x0 : x0 + resized_w,
                ] = density
                output_axis_position = z0 + position
            else:
                # The slice sequence becomes world Y (vertical). Image rows
                # become world Z and image columns remain world X:
                # (stack, row, column) -> (Z=row, Y=stack, X=column).
                volume[
                    z0 : z0 + resized_h,
                    y0 + position,
                    x0 : x0 + resized_w,
                ] = density
                output_axis_position = y0 + position
            print(
                f"Wrote interpolated stack slice {position + 1:03d}/{target_z:03d} "
                f"at cube {args.stack_axis.upper()}={output_axis_position:04d} "
                f"(source position={source_position:.3f})",
                flush=True,
            )
        del resized_stack
        volume.flush()
    except BaseException:
        del volume
        temporary.unlink(missing_ok=True)
        raise
    del volume

    check = np.load(temporary, mmap_mode="r", allow_pickle=False)
    expected_shape = (cube_size, cube_size, cube_size)
    if check.shape != expected_shape or check.dtype != np.float32:
        raise RuntimeError(
            f"Output verification failed: shape={check.shape}, dtype={check.dtype}"
        )
    if not (0.0 <= minimum <= maximum <= 1.0):
        raise RuntimeError(
            f"Output verification failed: invalid density range [{minimum}, {maximum}]"
        )
    # Verify all padding in small pieces so the check does not allocate a
    # cube-sized temporary boolean array.
    z1 = z0 + occupied_shape[0]
    y1 = y0 + occupied_shape[1]
    x1 = x0 + occupied_shape[2]
    for start in range(0, z0, 16):
        if np.any(check[start : min(start + 16, z0)] != 0):
            raise RuntimeError("Output verification failed: leading Z padding is not zero")
    for start in range(z1, cube_size, 16):
        if np.any(check[start : min(start + 16, cube_size)] != 0):
            raise RuntimeError("Output verification failed: trailing Z padding is not zero")
    for z in range(z0, z1):
        if y0 and np.any(check[z, :y0, :] != 0):
            raise RuntimeError("Output verification failed: leading Y padding is not zero")
        if y1 < cube_size and np.any(check[z, y1:, :] != 0):
            raise RuntimeError("Output verification failed: trailing Y padding is not zero")
        if x0 and np.any(check[z, y0:y1, :x0] != 0):
            raise RuntimeError("Output verification failed: leading X padding is not zero")
        if x1 < cube_size and np.any(check[z, y0:y1, x1:] != 0):
            raise RuntimeError("Output verification failed: trailing X padding is not zero")
    del check

    temporary.replace(output)
    print(f"Saved: {output}")
    print(f"shape={expected_shape}, axes=(Z, Y, X), dtype=float32")
    print(f"inserted region offset (Z,Y,X)={offsets}, shape={occupied_shape}")
    print(f"source density range=[{minimum:.6f}, {maximum:.6f}]")
    print("Padding is stored as float32 zeros inside the NPY; no --pad_grid is needed.")


if __name__ == "__main__":
    main()
