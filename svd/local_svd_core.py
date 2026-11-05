"""Batched local axial SVD diagnostics for volumes in [z, y, x] order.

All blocks are disjoint. Boundary blocks retain their real dimensions, and
no values are padded before decomposition. Eigenvalues are squared singular
values; retaining a rank therefore needs no explicit reconstruction.
"""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter
from typing import Sequence
import warnings

import numpy as np
import torch


ENERGY_TARGETS = (0.90, 0.95, 0.99)


def _device(value: str) -> torch.device:
    automatic = value == "auto"
    if automatic:
        value = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if result.type == "cuda":
        try:
            with torch.no_grad():
                probe = torch.eye(3, device=result, dtype=torch.float64)
                torch.linalg.eigvalsh(probe @ probe)
                torch.cuda.synchronize(result)
        except RuntimeError as error:
            message = ("Installed PyTorch CUDA kernels cannot run on this GPU. "
                       "Use --device cpu or install a PyTorch build supporting "
                       f"the GPU architecture. Original error: {error}")
            if automatic:
                warnings.warn(message + " Falling back to CPU.", RuntimeWarning)
                return torch.device("cpu")
            raise RuntimeError(message) from error
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bounds(shape: tuple[int, ...], resolution: int, slab: int) -> np.ndarray:
    depth, height, width = shape
    return np.asarray([
        (z, y, x, min(z + slab, depth), min(y + resolution, height),
         min(x + resolution, width))
        for z in range(0, depth, slab)
        for y in range(0, height, resolution)
        for x in range(0, width, resolution)
    ], dtype=np.int64)


def analyze_local(
    stack: np.ndarray,
    resolution: int,
    slab: int,
    ranks: Sequence[int] = (1, 2, 4, 8, 12, 16),
    device: str = "auto",
    batch_size: int = 128,
    dtype: str = "float64",
    energy_epsilon: float = 1e-12,
) -> dict:
    """Return per-block spectra and errors for raw and axial-centered data.

    ``energy_epsilon`` classifies a block as near-zero by *mean squared*
    analyzed value. It changes effective-rank summaries only, never SSE.
    Requested ranks are capped at the maximum possible configuration rank;
    rank zero and that maximum are always included for reference. Each
    boundary block independently caps retained ranks at its own maximum.

    GPU input is uploaded once when sufficient free memory is available;
    otherwise only each batch is transferred. Decomposition and summaries
    use the requested dtype, without autocast or autograd.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3 or min(stack.shape) < 1:
        raise ValueError("stack must have nonempty shape [Z, H, W]")
    if not np.issubdtype(stack.dtype, np.number) or not np.isfinite(stack).all():
        raise ValueError("stack must contain finite numeric values")
    if resolution < 1 or slab < 1 or batch_size < 1:
        raise ValueError("resolution, slab and batch_size must be positive")
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    if energy_epsilon < 0 or not np.isfinite(energy_epsilon):
        raise ValueError("energy_epsilon must be finite and nonnegative")
    if any(int(r) != r or r < 0 for r in ranks):
        raise ValueError("ranks must be nonnegative integers")
    selected_device = _device(device)
    torch_dtype = getattr(torch, dtype)
    spectral_size = min(resolution * resolution, slab, stack.shape[0])
    selected_ranks = sorted({0, spectral_size} | {
        int(r) for r in ranks if r <= spectral_size
    })
    bounds = _bounds(stack.shape, resolution, slab)
    dims = bounds[:, 3:] - bounds[:, :3]
    counts = np.prod(dims, axis=1)
    full = np.all(dims == np.asarray([slab, resolution, resolution]), axis=1)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, shape in enumerate(dims):
        groups[tuple(int(v) for v in shape)].append(index)
    nblocks = len(bounds)
    analyses = {}
    for name in ("raw", "axial_centered"):
        analyses[name] = {
            "bounds": bounds,
            "voxel_count": counts,
            "max_rank": np.minimum(dims[:, 1] * dims[:, 2],
                                   dims[:, 0] - (name == "axial_centered")),
            "full_block": full,
            "eigenvalues": np.zeros((nblocks, spectral_size), dtype=np.float64),
            "energy": np.zeros(nblocks, dtype=np.float64),
            "original_energy": np.zeros(nblocks, dtype=np.float64),
            "mean_energy": np.zeros(nblocks, dtype=np.float64),
            "near_zero": np.zeros(nblocks, dtype=bool),
            "effective_ranks": np.zeros((nblocks, len(ENERGY_TARGETS)), dtype=np.int64),
            "sse": np.zeros((nblocks, len(selected_ranks)), dtype=np.float64),
            "retained_ranks": np.zeros((nblocks, len(selected_ranks)), dtype=np.int64),
        }
    _sync(selected_device)
    started = perf_counter()
    source = torch.from_numpy(np.ascontiguousarray(stack))
    input_resident = False
    if selected_device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(selected_device)
        source_bytes = source.numel() * source.element_size()
        largest_batch = min(batch_size, nblocks) * resolution**2 * min(slab, stack.shape[0])
        estimated_work = largest_batch * torch.tensor([], dtype=torch_dtype).element_size() * 5
        if source_bytes + estimated_work < free_bytes * 0.7:
            source = source.to(selected_device)
            input_resident = True
    _sync(selected_device)
    transfer_seconds = perf_counter() - started
    rank_tensor = torch.tensor(selected_ranks, device=selected_device, dtype=torch.int64)
    targets = torch.tensor(ENERGY_TARGETS, device=selected_device, dtype=torch_dtype)
    with torch.no_grad():
        for shape, indices in groups.items():
            depth, height, width = shape
            for offset in range(0, len(indices), batch_size):
                ids = indices[offset:offset + batch_size]
                batch = torch.stack([
                    source[z0:z1, y0:y1, x0:x1]
                    for z0, y0, x0, z1, y1, x1 in bounds[ids]
                ]).to(device=selected_device, dtype=torch_dtype)
                matrix = batch.reshape(len(ids), depth, height * width).transpose(1, 2)
                means = matrix.mean(dim=-1, keepdim=True)
                original_energy = matrix.square().sum(dim=(-2, -1))
                mean_energy = means.square().sum(dim=(-2, -1)) * depth
                for name in ("raw", "axial_centered"):
                    centered = name == "axial_centered"
                    a = matrix - means if centered else matrix
                    max_rank = min(height * width, depth - int(centered))
                    energy = a.square().sum(dim=(-2, -1))
                    gram = (a.transpose(-2, -1) @ a if a.shape[-1] <= a.shape[-2]
                            else a @ a.transpose(-2, -1))
                    spectrum = torch.linalg.eigvalsh(gram).flip(-1).clamp_min(0)
                    spectrum[:, max_rank:] = 0
                    # Match the direct trace while preserving nonnegative tails.
                    trace = spectrum.sum(-1)
                    scale = energy / trace.clamp_min(torch.finfo(torch_dtype).tiny)
                    spectrum = spectrum * scale[:, None]
                    cumsum = spectrum.cumsum(-1)
                    near_zero = energy <= energy_epsilon * (depth * height * width)
                    effective = (cumsum[:, :, None] < energy[:, None, None] * targets).sum(1) + 1
                    effective = effective.clamp_max(max_rank)
                    effective[near_zero] = 0
                    retained = rank_tensor[None, :].expand(len(ids), -1).clamp_max(max_rank)
                    tail = torch.cat((spectrum.flip(-1).cumsum(-1).flip(-1),
                                      torch.zeros((len(ids), 1), device=selected_device,
                                                  dtype=torch_dtype)), dim=1)
                    sse = tail.gather(1, retained)
                    # Preserve exact rank-zero reference energy even for tiny blocks.
                    sse = torch.where(retained == 0, energy[:, None], sse)
                    values = {
                        "energy": energy, "original_energy": original_energy,
                        "mean_energy": mean_energy, "near_zero": near_zero,
                        "effective_ranks": effective, "sse": sse,
                        "retained_ranks": retained,
                    }
                    result = analyses[name]
                    result["eigenvalues"][ids, :spectrum.shape[1]] = spectrum.cpu().numpy()
                    for key, value in values.items():
                        result[key][ids] = value.cpu().numpy()
    _sync(selected_device)
    return {
        "metadata": {
            "shape": list(stack.shape), "resolution": resolution, "slab": slab,
            "ranks": selected_ranks, "energy_targets": list(ENERGY_TARGETS),
            "device": str(selected_device), "dtype": dtype, "batch_size": batch_size,
            "energy_epsilon": energy_epsilon,
            "near_zero_definition": "analyzed mean squared value <= energy_epsilon",
            "input_resident_on_gpu": input_resident,
            "initial_transfer_seconds": transfer_seconds,
            "timing_seconds": perf_counter() - started,
        },
        "analyses": analyses,
    }


def reconstruct_block(
    stack: np.ndarray, bounds: Sequence[int], rank: int, analysis: str = "raw",
) -> dict[str, np.ndarray]:
    """Direct float64 SVD for a selected visualization block.

    Images have [z,h,w] order, spatial_modes [mode,h,w], axial_modes
    [mode,z]. All available modes are returned; rank controls reconstruction.
    The centered reconstruction includes its exact slab mean.
    """
    if analysis not in ("raw", "axial_centered"):
        raise ValueError("analysis must be raw or axial_centered")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    z0, y0, x0, z1, y1, x1 = (int(value) for value in bounds)
    original = np.asarray(stack[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    if original.size == 0:
        raise ValueError("bounds select an empty block")
    depth, height, width = original.shape
    matrix = original.reshape(depth, -1).T
    mean = matrix.mean(axis=1, keepdims=True)
    a = matrix - mean if analysis == "axial_centered" else matrix
    u, singular_values, vh = np.linalg.svd(a, full_matrices=False)
    retained = min(rank, len(singular_values))
    reconstructed = (u[:, :retained] * singular_values[:retained]) @ vh[:retained]
    if analysis == "axial_centered":
        reconstructed += mean
    return {
        "original": original,
        "reconstructed": reconstructed.T.reshape(depth, height, width),
        "spatial_modes": u.T.reshape(len(singular_values), height, width),
        "axial_modes": vh,
        "singular_values": singular_values,
        "mean_image": mean.reshape(height, width),
    }
