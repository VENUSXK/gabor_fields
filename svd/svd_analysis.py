#!/usr/bin/env python3
"""Analyze axial low-rank structure in a stack of grayscale PNG slices.

The PNG stack is interpreted as a volume V[y, x, z].  The x/y dimensions
are flattened into a matrix A with shape (height * width, num_slices), and
an exact SVD is obtained from the small Gram matrix A.T @ A.

By default two analyses are produced:
  raw              The original voxel values.
  axial_centered   The per-(x, y) mean across z is removed.  This measures
                   how complicated the variation along z is without letting
                   the static mean volume dominate the result.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_RANKS = (1, 2, 4, 6, 8, 12, 16)
ENERGY_TARGETS = (0.90, 0.95, 0.99)


def natural_key(path: Path) -> list[object]:
    """Sort names naturally: slice_2.png before slice_10.png."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compute axial SVD and cumulative energy for PNG slices."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir,
        help="Folder containing PNG slices (default: script folder).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "svd_results",
        help="Output folder (default: <script folder>/svd_results).",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Input filename pattern (default: *.png).",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=list(DEFAULT_RANKS),
        help="Ranks to report and reconstruct.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=8,
        help="Number of spatial/axial singular modes to visualize.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=-1.0,
        help=("Also analyze pixels whose maximum value over z exceeds this "
              "8-bit threshold (default: disabled)."),
    )
    return parser.parse_args()


def load_stack(input_dir: Path, pattern: str) -> tuple[np.ndarray, list[Path]]:
    files = sorted(input_dir.glob(pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No PNG files matching {pattern!r} in {input_dir}")

    arrays: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None
    for path in files:
        with Image.open(path) as image:
            if image.mode != "L":
                image = image.convert("L")
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ValueError(
                    f"Image size mismatch: {path.name} is {image.size}, "
                    f"expected {expected_size}"
                )
            arrays.append(np.asarray(image, dtype=np.float32) / 255.0)

    # stack: [z, y, x], A: [(y*x), z]
    stack = np.stack(arrays, axis=0)
    return stack, files


def exact_svd_from_gram(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U, singular values, V for A = U diag(s) V.T."""
    gram = np.asarray(a.T @ a, dtype=np.float64)
    eigenvalues, v = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    v = v[:, order]
    singular_values = np.sqrt(eigenvalues)

    # Only 30 spatial modes are needed, so forming U is inexpensive relative
    # to storing the input matrix itself.
    u = np.zeros((a.shape[0], a.shape[1]), dtype=np.float32)
    tolerance = np.finfo(np.float64).eps * max(a.shape) * singular_values[0]
    valid = singular_values > tolerance
    u[:, valid] = (a @ v[:, valid] / singular_values[valid]).astype(np.float32)
    return u, singular_values, v


def save_mode_image(mode: np.ndarray, path: Path) -> None:
    scale = float(np.max(np.abs(mode)))
    if scale == 0.0:
        normalized = np.full(mode.shape, 0.5, dtype=np.float32)
    else:
        # Singular vectors are signed. Gray is zero, white positive, black negative.
        normalized = 0.5 + 0.5 * mode / scale
    Image.fromarray(np.uint8(np.clip(normalized, 0.0, 1.0) * 255.0)).save(path)


def psnr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = float(np.mean((reference - reconstructed) ** 2, dtype=np.float64))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def analyze_matrix(
    name: str,
    a: np.ndarray,
    spatial_shape: tuple[int, int] | None,
    ranks: list[int],
    modes: int,
    output_root: Path,
) -> dict[str, object]:
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    u, singular_values, v = exact_svd_from_gram(a)
    energy = singular_values**2
    energy_ratio = energy / energy.sum() if energy.sum() > 0 else np.zeros_like(energy)
    cumulative = np.cumsum(energy_ratio)

    effective_ranks = {
        target: int(np.searchsorted(cumulative, target) + 1)
        for target in ENERGY_TARGETS
    }

    with (output_dir / "singular_values.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "singular_value", "energy_ratio", "cumulative_energy"])
        for index, (s, ratio, total) in enumerate(
            zip(singular_values, energy_ratio, cumulative), start=1
        ):
            writer.writerow([index, f"{s:.12g}", f"{ratio:.12g}", f"{total:.12g}"])

    valid_ranks = sorted({r for r in ranks if 1 <= r <= len(singular_values)})
    with (output_dir / "reconstruction_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "retained_energy", "rmse", "psnr_db"])
        for rank in valid_ranks:
            reconstructed = (u[:, :rank] * singular_values[:rank]) @ v[:, :rank].T
            rmse = float(np.sqrt(np.mean((a - reconstructed) ** 2, dtype=np.float64)))
            writer.writerow([rank, cumulative[rank - 1], rmse, psnr(a, reconstructed)])

    components = np.arange(1, len(singular_values) + 1)
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(components, cumulative * 100.0, "o-", linewidth=1.7, markersize=4)
    for target in ENERGY_TARGETS:
        plt.axhline(target * 100.0, color="gray", linestyle="--", linewidth=0.8)
        rank = effective_ranks[target]
        plt.scatter([rank], [cumulative[rank - 1] * 100.0], zorder=3)
        plt.annotate(f"{target:.0%}: r={rank}", (rank, cumulative[rank - 1] * 100.0),
                     xytext=(5, -14), textcoords="offset points", fontsize=9)
    plt.xlabel("Rank r")
    plt.ylabel("Cumulative energy (%)")
    plt.title(f"Axial SVD cumulative energy: {name}")
    plt.xlim(1, len(singular_values))
    plt.ylim(0, 101)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_energy.png", dpi=180)
    plt.close()

    mode_count = min(modes, len(singular_values))
    z = np.arange(a.shape[1])
    plt.figure(figsize=(8.0, 4.8))
    for index in range(mode_count):
        plt.plot(z, v[:, index], marker="o", markersize=3, label=f"mode {index + 1}")
        if spatial_shape is not None:
            save_mode_image(
                u[:, index].reshape(spatial_shape),
                output_dir / f"spatial_mode_{index + 1:02d}.png",
            )
    plt.xlabel("Slice index z")
    plt.ylabel("Axial mode coefficient")
    plt.title(f"First {mode_count} axial modes: {name}")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "axial_modes.png", dpi=180)
    plt.close()

    summary_lines = [
        f"analysis: {name}",
        f"matrix_shape: {a.shape[0]} x {a.shape[1]}",
        *[f"rank_for_{target:.0%}_energy: {effective_ranks[target]}"
          for target in ENERGY_TARGETS],
        "",
        "selected cumulative energies:",
    ]
    for rank in valid_ranks:
        summary_lines.append(f"  r={rank:2d}: {cumulative[rank - 1]:.8%}")
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "effective_ranks": effective_ranks,
        "cumulative": cumulative,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack, files = load_stack(input_dir, args.pattern)
    num_slices, height, width = stack.shape
    a_raw = stack.reshape(num_slices, -1).T

    print(f"Loaded {num_slices} slices from {input_dir}")
    print(f"Image size: {width} x {height}; matrix A: {a_raw.shape}")
    print("Slice order:")
    for index, path in enumerate(files):
        print(f"  z={index:02d}: {path.name}")

    (output_dir / "slice_order.txt").write_text(
        "\n".join(f"{i:02d}\t{path.name}" for i, path in enumerate(files)) + "\n",
        encoding="utf-8",
    )

    analyses: list[tuple[str, np.ndarray]] = [("raw", a_raw)]
    axial_mean = np.mean(a_raw, axis=1, keepdims=True)
    analyses.append(("axial_centered", a_raw - axial_mean))

    if args.foreground_threshold >= 0.0:
        threshold = args.foreground_threshold / 255.0
        foreground_mask = np.max(a_raw, axis=1) > threshold
        foreground = a_raw[foreground_mask]
        if foreground.size:
            analyses.append((f"foreground_gt_{args.foreground_threshold:g}", foreground))
            np.save(output_dir / "foreground_mask.npy", foreground_mask.reshape(height, width))
        else:
            print("Warning: foreground mask is empty; foreground analysis skipped.")

    combined_rows: list[list[object]] = []
    for name, matrix in analyses:
        print(f"\nAnalyzing {name} with shape {matrix.shape} ...")
        result = analyze_matrix(
            name=name,
            a=matrix,
            spatial_shape=(height, width) if matrix.shape[0] == height * width else None,
            ranks=args.ranks,
            modes=args.modes,
            output_root=output_dir,
        )
        effective = result["effective_ranks"]
        cumulative = result["cumulative"]
        print(
            "  " + ", ".join(
                f"{target:.0%} energy -> rank {effective[target]}"
                for target in ENERGY_TARGETS
            )
        )
        for rank in sorted({r for r in args.ranks if 1 <= r <= num_slices}):
            combined_rows.append([name, rank, cumulative[rank - 1]])

    with (output_dir / "energy_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["analysis", "rank", "cumulative_energy"])
        writer.writerows(combined_rows)

    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
