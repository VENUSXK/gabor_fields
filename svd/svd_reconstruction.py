"""Reconstruct slice images from local/global SVD without a plotting backend."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

from local_svd_core import _device, _sync


def reconstruct_slices(stack, result, analysis, ranks, slices, device="cpu", batch_size=128):
    """Return float32 preview slices and float64 per-slice SSE before clipping.

    Only selected images are retained; errors are evaluated over every voxel.
    The same native, unpadded blocks as analyze_local are used.
    """
    stack = np.asarray(stack)
    ranks = list(ranks)
    slices = list(slices)
    depth, height, width = stack.shape
    outputs = np.zeros((len(ranks), len(slices), height, width), np.float32)
    slice_sse = np.zeros((len(ranks), depth), np.float64)
    clip_count = np.zeros(len(ranks), np.int64)
    bounds = result["analyses"][analysis]["bounds"]
    groups = defaultdict(list)
    for b in bounds:
        groups[tuple(b[3:] - b[:3])].append(b)
    dev = _device(device)
    with torch.no_grad():
        for shape, entries in groups.items():
            nz, ny, nx = (int(v) for v in shape)
            for start in range(0, len(entries), batch_size):
                batch_bounds = entries[start:start + batch_size]
                batch = np.stack([stack[z0:z1, y0:y1, x0:x1]
                                  for z0, y0, x0, z1, y1, x1 in batch_bounds])
                matrix = torch.as_tensor(batch, dtype=torch.float64, device=dev).reshape(
                    len(batch_bounds), nz, ny * nx).transpose(-2, -1)
                mean = matrix.mean(-1, keepdim=True)
                centered = analysis == "axial_centered"
                a = matrix - mean if centered else matrix
                # Axial Gram is at most depth x depth, even for the full image.
                _, vectors = torch.linalg.eigh(a.transpose(-2, -1) @ a)
                vectors = vectors.flip(-1)
                max_rank = min(ny * nx, nz - int(centered))
                for ri, requested in enumerate(ranks):
                    retained = min(requested, max_rank)
                    if retained >= max_rank:
                        reconstructed = matrix.clone()
                    else:
                        basis = vectors[..., :retained]
                        reconstructed = (a @ basis) @ basis.transpose(-2, -1)
                        if centered:
                            reconstructed += mean
                    error = (matrix - reconstructed).square().sum(-2).cpu().numpy()
                    clip_count[ri] += int(((reconstructed < 0) | (reconstructed > 1)).sum().item())
                    image_batch = reconstructed.transpose(-2, -1).reshape(
                        len(batch_bounds), nz, ny, nx).float().cpu().numpy()
                    for bi, (z0, y0, x0, z1, y1, x1) in enumerate(batch_bounds):
                        slice_sse[ri, z0:z1] += error[bi]
                        for si, z in enumerate(slices):
                            if z0 <= z < z1:
                                outputs[ri, si, y0:y1, x0:x1] = image_batch[bi, z-z0]
    _sync(dev)
    return {"images": outputs, "slice_sse": slice_sse, "outside_unit_interval": clip_count}


def block_modes(stack, bounds, analysis, modes=4):
    """Float64 Gram-based templates; scale/sign do not imply density."""
    z0, y0, x0, z1, y1, x1 = map(int, bounds)
    block = np.asarray(stack[z0:z1, y0:y1, x0:x1], dtype=np.float64)
    z, h, w = block.shape
    a = block.reshape(z, -1).T
    mean = a.mean(-1, keepdims=True)
    if analysis == "axial_centered":
        a = a - mean
    eig, vectors = np.linalg.eigh(a.T @ a)
    sigma = np.sqrt(np.maximum(eig[::-1], 0))
    vectors = vectors[:, ::-1]
    count = min(modes, h*w, z - int(analysis == "axial_centered"))
    spatial = a @ vectors[:, :count]
    for i in range(count):
        if sigma[i] > max(sigma[0] * 1e-10, 1e-14):
            spatial[:, i] /= sigma[i]
        else:
            spatial[:, i] = 0
    return {"spatial": spatial.T.reshape(count, h, w),
            "axial": vectors[:, :count].T, "sigma": sigma[:count],
            "mean": mean.reshape(h, w), "original": block}
