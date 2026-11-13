#!/usr/bin/env python3
"""Convert image slices or a ZYX NPY to an unpadded float32 volume.

Image slices run along Y by default: image rows become Z and columns become X.
Existing NPY inputs always use (Z,Y,X) order. Preserve every axis resolution
unless --res-x, --res-y, or --res-z is supplied. Upsample by interpolation and
downsample by area averaging; no automatic resizing or zero padding is used.

Example:
    python data/thin_layer/texture2npy.py data/laoshan/LS-20261008 \
        --output data/thin_layer/LS_density.npy --res-y 100
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


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
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
        "--output", type=Path, default=script_dir / "volume.npy",
        help="Output .npy path",
    )
    for axis in ('x', 'y', 'z'):
        parser.add_argument(
            f'--res-{axis}', type=positive_int,
            help=f'Output {axis.upper()} resolution (default: preserve original)',
        )
    parser.add_argument(
        "--stack-axis", choices=("z", "y"), default="y",
        help="Axis for image slices: y=thin Y layer, z=depth (default: y); NPY axes stay ZYX",
    )
    parser.add_argument(
        "--reverse-stack", "--reverse-z", dest="reverse_z", action="store_true",
        help="Reverse the slice sequence before resampling",
    )
    parser.add_argument(
        "--flip-y", action="store_true",
        help="Flip image rows (Z for Y stacks, Y for Z stacks)",
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


def resize_slice(density: np.ndarray, height: int, width: int) -> np.ndarray:
    # Resample each plane axis separately so mixed up/downsampling stays correct.
    image = Image.fromarray(density)
    if image.width != width:
        method = Image.Resampling.BOX if width < image.width else Image.Resampling.BILINEAR
        image = image.resize((width, image.height), resample=method)
    if image.height != height:
        method = Image.Resampling.BOX if height < image.height else Image.Resampling.BILINEAR
        image = image.resize((image.width, height), resample=method)
    return np.asarray(image, dtype=np.float32)


def resample_stack_slice(stack: np.ndarray, position: int, target_count: int) -> np.ndarray:
    source_count = stack.shape[0]
    if source_count == target_count:
        return stack[position]
    if target_count < source_count:
        # Integrate overlapping source cells for anti-aliased downsampling.
        begin = position * source_count / target_count
        end = (position + 1) * source_count / target_count
        result = np.zeros(stack.shape[1:], dtype=np.float32)
        for index in range(math.floor(begin), min(math.ceil(end), source_count)):
            weight = (min(end, index + 1) - max(begin, index)) / (end - begin)
            result += stack[index] * np.float32(weight)
        return result
    source_position = position * (source_count - 1) / (target_count - 1)
    lower = math.floor(source_position)
    upper = min(lower + 1, source_count - 1)
    weight = np.float32(source_position - lower)
    return stack[lower] * (np.float32(1.0) - weight) + stack[upper] * weight


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
        # Treat NPY axis labels as fixed; only change the internal slice view.
        if args.stack_axis == 'y':
            compact = np.swapaxes(compact, 0, 1)
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

    source_shape = ((source_z, source_h, source_w) if args.stack_axis == 'z'
                    else (source_h, source_z, source_w))
    if min(source_shape) <= 0:
        raise ValueError("Source dimensions must be greater than zero")
    output_shape = (args.res_z or source_shape[0], args.res_y or source_shape[1],
                    args.res_x or source_shape[2])
    target_z, resized_h, resized_w = (output_shape if args.stack_axis == 'z'
                                     else (output_shape[1], output_shape[0], output_shape[2]))
    print('Volume export plan')
    print(f'  source shape (Z,Y,X): {source_shape}')
    print(f'  output shape (Z,Y,X): {output_shape}')
    print(f'  resolution (X,Y,Z):   {output_shape[::-1]}')
    print(f'  output data size:     {math.prod(output_shape) * FLOAT32_BYTES / GIB:.3f} GiB')
    print('  padding:              none')
    if args.plan_only:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output or partial file: {output}"
        )

    volume = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=output_shape,
    )
    try:
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
                    density = (
                        np.asarray(image, dtype=np.float32) / np.float32(255.0)
                    )
                source_label = f"source index {index}: {path.name}"

            if not np.isfinite(density).all():
                raise ValueError(f"Non-finite density value in {source_label}")
            density = resize_slice(density, resized_h, resized_w)
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
        for position in range(target_z):
            density = resample_stack_slice(resized_stack, position, target_z)
            if args.stack_axis == "z":
                volume[position, :, :] = density
            else:
                # Map (stack,row,column) to (Z=row,Y=stack,X=column).
                volume[:, position, :] = density
            print(
                f"Wrote interpolated stack slice {position + 1:03d}/{target_z:03d} "
                f"at {args.stack_axis.upper()}={position:04d}",
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
    expected_shape = output_shape
    if check.shape != expected_shape or check.dtype != np.float32:
        raise RuntimeError(
            f"Output verification failed: shape={check.shape}, dtype={check.dtype}"
        )
    if not (0.0 <= minimum <= maximum <= 1.0):
        raise RuntimeError(
            f"Output verification failed: invalid density range [{minimum}, {maximum}]"
        )
    del check

    temporary.replace(output)
    print(f"Saved: {output}")
    print(f"shape={expected_shape}, axes=(Z, Y, X), dtype=float32")
    print(f"source density range=[{minimum:.6f}, {maximum:.6f}]")
    print("No zero padding was added.")


if __name__ == "__main__":
    main()
