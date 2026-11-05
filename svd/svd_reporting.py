"""Aggregate and visualize batched local axial SVD diagnostics.

Storage ratios here count scalar parameters, not encoded bytes.  Spatial SVD
templates are unrestricted arrays and are not fitted Gaussian primitives.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path, payload):
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}


def _aggregate(data, metadata, analysis):
    n = len(data["bounds"])
    bounds = np.asarray(data["bounds"])
    depth = bounds[:, 3] - bounds[:, 0]
    pixels = (bounds[:, 4] - bounds[:, 1]) * (bounds[:, 5] - bounds[:, 2])
    zero = np.asarray(data["near_zero"], dtype=bool)
    original = np.asarray(data["original_energy"])
    residual = np.maximum(original - np.asarray(data["mean_energy"]), 0)
    near_constant = residual <= metadata.get("energy_epsilon", 1e-12) * np.asarray(data["voxel_count"])
    scopes = {"all": np.ones(n, dtype=bool),
              "full_blocks": np.asarray(data["full_block"], dtype=bool),
              "signal_blocks": ~zero}
    rows, summaries = [], {}
    for scope, mask in scopes.items():
        count = int(mask.sum())
        voxels = int(np.asarray(data["voxel_count"])[mask].sum())
        informative = mask & ~zero
        summary = {"block_count": count, "voxel_count": voxels,
                   "near_zero_count": int((mask & zero).sum()),
                   "axial_near_constant_count": int((mask & near_constant).sum()),
                   "energy_rank_statistics_exclude_near_zero": True,
                   "effective_ranks": {str(target): _stats(np.asarray(data["effective_ranks"])[informative, j])
                                       for j, target in enumerate((0.90, 0.95, 0.99))},
                   "effective_relative_ranks": {str(target): _stats(
                       np.asarray(data["effective_ranks"])[informative, j] /
                       np.maximum(np.asarray(data["max_rank"])[informative], 1))
                       for j, target in enumerate((0.90, 0.95, 0.99))}}
        original_energy = float(np.asarray(data["original_energy"])[mask].sum())
        energy = float(np.asarray(data["energy"])[mask].sum())
        summary.update(original_energy=original_energy, analyzed_energy=energy)
        for j, rank in enumerate(metadata["ranks"]):
            actual = np.asarray(data["retained_ranks"])[mask, j]
            sse_by_block = np.asarray(data["sse"])[mask, j]
            sse = float(sse_by_block.sum(dtype=np.float64))
            mse = sse / voxels if voxels else float("nan")
            rmse = math.sqrt(max(mse, 0.0)) if voxels else float("nan")
            psnr = -10 * math.log10(mse) if mse > 0 else (float("inf") if voxels else float("nan"))
            parameter_count = int(np.sum(actual * (pixels[mask] + depth[mask] + 1)))
            if analysis == "axial_centered":
                parameter_count += int(pixels[mask].sum())
            full = actual >= np.asarray(data["max_rank"])[mask]
            relative = actual / np.maximum(np.asarray(data["max_rank"])[mask], 1)
            block_rmse = np.sqrt(np.maximum(sse_by_block, 0) / np.asarray(data["voxel_count"])[mask])
            rows.append({"resolution": metadata["resolution"], "slab": metadata["slab"],
                         "analysis": analysis, "scope": scope, "rank": int(rank),
                         "block_count": count, "voxel_count": voxels, "sse": sse,
                         "rmse": rmse, "psnr_db": psnr,
                         "block_rmse_p90": float(np.percentile(block_rmse, 90)) if count else float("nan"),
                         "retained_energy": 1 - sse / energy if energy > 0 else float("nan"),
                         "parameter_count": parameter_count,
                         "parameter_ratio": parameter_count / voxels if voxels else float("nan"),
                         "scalar_compression_ratio": voxels / parameter_count if parameter_count else float("inf"),
                         "full_rank_block_fraction": float(full.mean()) if count else float("nan"),
                         "all_blocks_full_rank": bool(full.all()) if count else False,
                         "mean_relative_rank": float(relative.mean()) if count else float("nan")})
        summaries[scope] = summary
    return rows, summaries


def _spatial_plot(data, metadata, values, path, title, label):
    bounds = np.asarray(data["bounds"])
    starts = np.unique(bounds[:, 0])
    ncols = min(4, len(starts))
    nrows = math.ceil(len(starts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows), squeeze=False,
                             layout="constrained")
    finite = np.asarray(values)[np.isfinite(values)]
    vmax = max(float(finite.max()), 1e-12) if finite.size else 1.0
    for ax, start in zip(axes.flat, starts):
        mask = bounds[:, 0] == start
        b = bounds[mask]
        xs = np.unique(np.concatenate((b[:, 2], b[:, 5])))
        ys = np.unique(np.concatenate((b[:, 1], b[:, 4])))
        grid = np.full((len(ys)-1, len(xs)-1), np.nan)
        grid[np.searchsorted(ys, b[:, 1]), np.searchsorted(xs, b[:, 2])] = np.asarray(values)[mask]
        mesh = ax.pcolormesh(xs, ys, grid, cmap="viridis", vmin=0, vmax=vmax, shading="flat")
        ax.set_ylim(ys[-1], ys[0])
        ax.set_aspect("equal")
        ax.set_title(f"z=[{start}, {b[0, 3]})")
        ax.set_xlabel("x (source pixels)")
        ax.set_ylabel("y (source pixels)")
    for ax in list(axes.flat)[len(starts):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.colorbar(mesh, ax=list(axes.flat)[:len(starts)], label=label, shrink=0.8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plots(data, metadata, directory, analysis):
    informative = ~np.asarray(data["near_zero"], dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bins = np.arange(-0.5, max(2, int(np.max(data["max_rank"])) + 2), 1)
    for j, target in enumerate((90, 95, 99)):
        ax.hist(np.asarray(data["effective_ranks"])[informative, j], bins=bins,
                histtype="step", label=f"{target}% energy")
    ax.set(xlabel="Required rank", ylabel="Informative block count", title=analysis)
    ax.legend()
    fig.savefig(directory / "rank_distribution.png", dpi=140)
    plt.close(fig)
    _spatial_plot(data, metadata, np.asarray(data["effective_ranks"])[:, 2],
                  directory / "rank_maps.png", f"{analysis}: rank for 99% energy", "Required rank")
    _spatial_plot(data, metadata, np.asarray(data["effective_ranks"])[:, 1],
                  directory / "rank_maps_95.png", f"{analysis}: rank for 95% energy", "Required rank")
    ranks = np.asarray(metadata["ranks"])
    j = int(np.argmin(abs(ranks - 2)))
    rmse = np.sqrt(np.maximum(np.asarray(data["sse"])[:, j], 0) / data["voxel_count"])
    _spatial_plot(data, metadata, rmse, directory / "error_maps.png",
                  f"{analysis}: RMSE at requested rank {ranks[j]}", "RMSE (range 0–1)")
    for requested in (1, 2, 4):
        match = np.flatnonzero(ranks == requested)
        if match.size:
            errors = np.sqrt(np.maximum(np.asarray(data["sse"])[:, match[0]], 0) / data["voxel_count"])
            _spatial_plot(data, metadata, errors, directory / f"error_maps_rank_{requested}.png",
                          f"{analysis}: RMSE at requested rank {requested}", "RMSE (range 0–1)")


def _examples(data, metadata, analysis, directory, stack, count):
    from local_svd_core import reconstruct_block
    ranks = np.asarray(metadata["ranks"])
    j = int(np.argmin(abs(ranks - 2)))
    errors = np.sqrt(np.maximum(np.asarray(data["sse"])[:, j], 0) / data["voxel_count"])
    informative = np.flatnonzero(~np.asarray(data["near_zero"], dtype=bool))
    if not informative.size:
        return
    ordered = informative[np.argsort(errors[informative])]
    selected = list(dict.fromkeys([int(ordered[len(ordered)//2]), int(ordered[-1])]))
    if count > 2:
        selected += [int(i) for i in ordered[np.linspace(0, len(ordered)-1, count, dtype=int)]
                     if int(i) not in selected]
    out = directory / "examples"
    out.mkdir(exist_ok=True)
    for i in selected[:count]:
        example = reconstruct_block(stack, data["bounds"][i], int(ranks[j]), analysis)
        original = np.asarray(example["original"])
        reconstructed = np.asarray(example["reconstructed"])
        # Core examples use source stack order [z,y,x].
        mid = original.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
        axes[0].imshow(original[mid], cmap="gray", vmin=0, vmax=1)
        axes[1].imshow(reconstructed[mid], cmap="gray", vmin=0, vmax=1)
        err = axes[2].imshow(abs(original[mid]-reconstructed[mid]), cmap="magma", vmin=0)
        for ax, title in zip(axes, ("Original", "Reconstructed", "Absolute error")):
            ax.set_title(title)
            ax.axis("off")
        fig.colorbar(err, ax=axes[2], shrink=0.7)
        fig.suptitle(f"Block {i}; bounds={data['bounds'][i].tolist()}; rank={ranks[j]}")
        fig.savefig(out / f"block_{i:06d}.png", dpi=140)
        plt.close(fig)
        nmodes = min(4, len(example["singular_values"]))
        fig, axes = plt.subplots(2, nmodes, figsize=(3*nmodes, 5), squeeze=False,
                                 layout="constrained")
        for k in range(nmodes):
            mode = np.asarray(example["spatial_modes"])[k]
            limit = max(float(np.abs(mode).max()), 1e-12)
            axes[0, k].imshow(mode, cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[0, k].set_title(f"Mode {k+1}; sigma={example['singular_values'][k]:.3g}")
            axes[0, k].axis("off")
            axes[1, k].plot(np.arange(original.shape[0]) + data["bounds"][i, 0],
                            np.asarray(example["axial_modes"])[k], "o-")
            axes[1, k].set_xlabel("Source slice index")
            axes[1, k].grid(alpha=.2)
        fig.suptitle(f"Block {i}: local spatial patterns and axial profiles")
        fig.savefig(out / f"block_{i:06d}_modes.png", dpi=140)
        plt.close(fig)


def save_config_results(result, output_dir: Path, stack=None, examples=2, make_plots=True):
    """Save one resolution/slab configuration; return comparison-table rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    _json(output_dir / "config.json", metadata)
    comparison = []
    for analysis, data in result["analyses"].items():
        directory = output_dir / analysis
        directory.mkdir(exist_ok=True)
        rows, summaries = _aggregate(data, metadata, analysis)
        comparison.extend(rows)
        _json(directory / "summary.json", {"metadata": metadata, "analysis": analysis,
              "scopes": summaries, "rank_metrics": rows,
              "notes": ["PSNR peak is 1; null PSNR denotes infinity or an empty scope.",
                        "Ratios count scalar parameters, not compressed bytes; native uint8 source and floating-point factors have different element sizes.",
                        "Centered capacity includes the per-pixel slab mean.",
                        "Full-rank rows are reconstruction checks, not evidence of low-rank compression."]})
        _csv(directory / "rank_metrics.csv", rows)
        np.savez_compressed(directory / "block_metrics.npz", **data)
        if make_plots:
            _plots(data, metadata, directory, analysis)
            if stack is not None and examples > 0:
                _examples(data, metadata, analysis, directory, stack, examples)
    return comparison


def save_comparison(rows, output_root: Path, make_plots=True):
    """Save cross-configuration table and finite quality/capacity comparisons."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _csv(output_root / "comparison.csv", rows)
    if not rows or not make_plots:
        return
    all_rows = [r for r in rows if r["scope"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, analysis in zip(axes, ("raw", "axial_centered")):
        for slab in sorted({r["slab"] for r in all_rows}):
            subset = [r for r in all_rows if r["analysis"] == analysis and r["slab"] == slab
                      and np.isfinite(r["psnr_db"]) and np.isfinite(r["scalar_compression_ratio"])]
            low_rank = [r for r in subset if not r["all_blocks_full_rank"]]
            full_rank = [r for r in subset if r["all_blocks_full_rank"]]
            if low_rank:
                ax.scatter([r["scalar_compression_ratio"] for r in low_rank], [r["psnr_db"] for r in low_rank],
                           label=f"slab {slab}", alpha=.8)
            if full_rank:
                ax.scatter([r["scalar_compression_ratio"] for r in full_rank], [r["psnr_db"] for r in full_rank],
                           marker="x", color="gray", alpha=.5)
        ax.set(xlabel="Source scalars / SVD parameters", ylabel="PSNR (dB)", title=analysis)
        ax.axvline(1, color="gray", linestyle="--", linewidth=.8)
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.savefig(output_root / "quality_capacity.png", dpi=150)
    plt.close(fig)
    chosen = sorted({r["rank"] for r in all_rows if r["rank"] in (1, 2, 4)})
    if chosen:
        fig, axes = plt.subplots(2, len(chosen), figsize=(4.5*len(chosen), 7), squeeze=False,
                                 layout="constrained")
        for row_index, analysis in enumerate(("raw", "axial_centered")):
            for ax, rank in zip(axes[row_index], chosen):
                for slab in sorted({r["slab"] for r in all_rows}):
                    subset = sorted([r for r in all_rows if r["analysis"] == analysis
                                     and r["slab"] == slab and r["rank"] == rank
                                     and np.isfinite(r["psnr_db"])], key=lambda r: r["resolution"])
                    if subset:
                        ax.plot([r["resolution"] for r in subset], [r["psnr_db"] for r in subset],
                                "o-", label=f"slab {slab}")
                ax.set(xlabel="Block side (source pixels)", ylabel="PSNR (dB)",
                       title=f"{analysis}, requested rank {rank}")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend()
                ax.grid(alpha=.2)
        fig.savefig(output_root / "quality_by_resolution.png", dpi=140)
        plt.close(fig)
