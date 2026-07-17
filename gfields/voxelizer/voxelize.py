# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Voxelization: turn a gabor-field primitive mixture into a dense voxel grid.

The single home for the voxelization itself (importance sampling of each
primitive's Gaussian envelope, with adaptive per-primitive sample counts and a
cosine term for Gabor levels). ``gfields voxelize compare`` calls this and
``gfields.references.grid`` to trace mixture-vs-grid comparisons.
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import gfields


def voxelize_asset(
    primitive_dir: str,
    grid_res: list,
    num_levels: int = 2,
    num_samples: int = 100_000,
    min_samples: int = 1_000_000,
    max_samples: int = 50_000_000,
    device: str = "cuda"
) -> tuple[torch.Tensor, dict]:
    """
    Voxelize a volumetric primitives asset using importance sampling.
    
    Args:
        primitive_dir: Path to the asset directory with a data/ folder holding the per-level .ply store
        grid_res: Grid resolution [x, y, z]
        num_levels: Number of pyramid levels in the asset
        num_samples: Number of samples per primitive for importance sampling
        device: Device to run on
        
    Returns:
        (grid, span) - voxelized grid tensor and span info
    """
    print("\n" + "="*70)
    print("VOXELIZING ASSET")
    print("="*70)
    print(f"Asset: {primitive_dir}")
    print(f"Grid resolution: {grid_res}")
    print(f"Pyramid levels: {num_levels}")
    
    # Load primitives from the unified per-level PLY store
    pyr_prims = {}
    for pyr in range(num_levels):
        a = gfields.io.load_ellipsoid_ply(
            os.path.join(primitive_dir, 'data', f'root.primitives_pyr{pyr}.ply'))

        means = torch.tensor(np.asarray(a['centers'], dtype=np.float32),
                           device=device, dtype=torch.float32)
        scales = torch.tensor(np.asarray(a['scales'], dtype=np.float32),
                            device=device, dtype=torch.float32)
        quaternions = np.asarray(a['quaternions'], dtype=np.float32)

        # Convert quaternions to rotation matrices using Mitsuba (matching Ellipsoid.gather)
        quat_mi = mi.Quaternion4f(mi.TensorXf(quaternions.T))
        rot_mi = dr.quat_to_matrix(mi.Matrix3f, quat_mi)
        rot_np = np.array(rot_mi)  # shape (3, 3, N) in drjit column-major layout
        # permute(2, 0, 1) gives R per primitive; kernel does p_local = R^T * (x - center)
        # so world transform is: x = center + R * diag(s) * u
        # In PyTorch row-vector convention: pts = u @ diag(s) @ R^T
        rotations = torch.tensor(rot_np, device=device, dtype=torch.float32).permute(2, 0, 1)

        # Load opacities (try different naming conventions)
        if 'opacities' in a:
            opacity = a['opacities']
        elif 'sigma_t' in a:
            opacity = a['sigma_t']
        else:
            opacity = a['sigmat']
        sigmats = torch.tensor(np.asarray(opacity, dtype=np.float32),
                             device=device, dtype=torch.float32)

        # Load omega for Gabor levels (pyr > 0)
        if pyr > 0 and 'omega' in a:
            omegas = torch.tensor(np.asarray(a['omega'], dtype=np.float32),
                                device=device, dtype=torch.float32)
        else:
            omegas = None
        
        pyr_prims[pyr] = {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'sigmats': sigmats.flatten(),
            'omegas': omegas.flatten() if omegas is not None else None
        }
        
        print(f"Level {pyr}: {means.shape[0]} primitives")
        print(f"  Opacity (sigmat) - min: {sigmats.min():.6f}, max: {sigmats.max():.6f}, mean: {sigmats.mean():.6f}, sum: {sigmats.sum():.6f}")
        if omegas is not None:
            print(f"  Omega (frequency) - min: {omegas.min():.6f}, max: {omegas.max():.6f}, mean: {omegas.mean():.6f}")
    
    # Initialize grid
    grid_res_t = torch.tensor(grid_res, device=device, dtype=torch.float32)
    num_voxels = int(np.prod(grid_res))
    grid = torch.zeros(num_voxels, device=device, dtype=torch.float32)
    
    # Define span (bounding box)
    span = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    side_length = span[3:] - span[:3]
    
    extent = 3.0
    
    # Voxelize each pyramid level
    for pyr in range(num_levels):
        means = pyr_prims[pyr]['means']
        scales = pyr_prims[pyr]['scales']
        rotations = pyr_prims[pyr]['rotations']
        sigmats = pyr_prims[pyr]['sigmats']
        omegas = pyr_prims[pyr]['omegas']
        
        P = means.shape[0]
        # Gabor-ness is derived from omega: a level oscillates iff any omega != 0.
        is_gabor = (omegas is not None) and bool(torch.any(omegas != 0))

        # Compute adaptive sample count per primitive based on volume coverage
        # A primitive with scale s covers ~(2*extent*s)^3 world-space volume
        # The number of voxels it spans is proportional to prod(s) / voxel_volume
        voxel_size = side_length / grid_res_t  # (3,) tensor
        voxel_vol_scalar = voxel_size.prod().item()
        
        # For each primitive, estimate voxels covered and scale samples accordingly
        prim_volumes = (2.0 * extent * scales).prod(dim=1)  # (P,)
        prim_voxel_counts = prim_volumes / voxel_vol_scalar  # approximate voxels per primitive
        
        # Target samples per voxel covered by the primitive
        # Gabors need significantly more due to oscillatory cosine term
        samples_per_voxel = 1024 if is_gabor else 512
        
        # For Gabors, also scale with frequency squared (higher omega = finer oscillations)
        if is_gabor and omegas is not None:
            freq_factor = 1.0 + omegas.abs().flatten() ** 2
            adaptive_samples = torch.clamp(
                (prim_voxel_counts * samples_per_voxel * freq_factor).long(),
                min=min_samples,
                max=max_samples
            )
        else:
            adaptive_samples = torch.clamp(
                (prim_voxel_counts * samples_per_voxel).long(),
                min=min_samples,
                max=max_samples
            )
        
        total_samples = adaptive_samples.sum().item()
        print(f"\nProcessing level {pyr} ({'Gabor' if is_gabor else 'Gaussian'}): {P} primitives")
        print(f"  Using opacity values: min={sigmats.min():.6f}, max={sigmats.max():.6f}, mean={sigmats.mean():.6f}")
        print(f"  Adaptive samples: min={adaptive_samples.min().item()}, max={adaptive_samples.max().item()}, mean={adaptive_samples.float().mean().item():.0f}, total={total_samples:.0f}")
        
        # Track contribution statistics
        total_contrib_for_level = 0.0
        grid_before_level = grid.sum().item()
        
        # Process each primitive, generating samples in GPU-friendly chunks
        CHUNK = 10_000_000
        
        for i in tqdm(range(P), desc=f"Level {pyr}"):
            mu = means[i]
            scale = scales[i]
            R = rotations[i]
            sigma = sigmats[i].item()
            N_i = int(adaptive_samples[i].item())
            beta = omegas[i].item() if is_gabor else 0.0
            
            prim_contrib = 0.0
            done = 0
            while done < N_i:
                batch_target = min(CHUNK, N_i - done)
                # Generate truncated N(0,I) on the fly
                raw = torch.randn(int(batch_target * 1.5), 3, device=device, dtype=torch.float32)
                local_chunk = raw[(raw.abs() <= extent).all(dim=-1)][:batch_target]
                n = local_chunk.shape[0]
                if n == 0:
                    continue
                del raw
                
                pts_world = (local_chunk * scale.unsqueeze(0)) @ R.T + mu.unsqueeze(0)
                
                lin_indices = (pts_world - span[0:3].unsqueeze(0)) / side_length.unsqueeze(0)
                lin_indices = torch.floor(lin_indices * grid_res_t.unsqueeze(0)).long()
                del pts_world
                
                valid_mask = (
                    (lin_indices[:, 0] >= 0) & (lin_indices[:, 0] < grid_res[0]) &
                    (lin_indices[:, 1] >= 0) & (lin_indices[:, 1] < grid_res[1]) &
                    (lin_indices[:, 2] >= 0) & (lin_indices[:, 2] < grid_res[2])
                )
                
                lin_indices_flat = (
                    lin_indices[:, 0] * grid_res[1] * grid_res[2] + 
                    lin_indices[:, 1] * grid_res[2] + 
                    lin_indices[:, 2]
                ).long()
                del lin_indices
                
                if is_gabor:
                    cosine_term = torch.cos(beta * local_chunk.sum(dim=-1))
                    contributions = (sigma * cosine_term / N_i)
                    del cosine_term
                else:
                    contributions = torch.full((n,), sigma / N_i, device=device, dtype=torch.float32)
                del local_chunk
                
                grid.scatter_add_(0, lin_indices_flat[valid_mask], contributions[valid_mask])
                prim_contrib += contributions[valid_mask].sum().item()
                del lin_indices_flat, valid_mask, contributions
                
                done += n
            
            total_contrib_for_level += prim_contrib
            if i < 3:
                print(f"    Primitive {i}: sigma={sigma:.6f}, N={N_i}, contrib={prim_contrib:.6f}")
        
        grid_after_level = grid.sum().item()
        print(f"  Level {pyr} contribution: {total_contrib_for_level:.6f} (grid increased by {grid_after_level - grid_before_level:.6f})")
    
    # Compute expected theoretical mass
    expected_mass = 0.0
    for pyr in range(num_levels):
        sigmats = pyr_prims[pyr]['sigmats']
        omegas = pyr_prims[pyr]['omegas']
        is_gabor = (omegas is not None) and bool(torch.any(omegas != 0))

        if is_gabor:
            beta_values = omegas
            gabor_damping = torch.exp(-0.5 * 3.0 * beta_values ** 2)
            expected_mass += (sigmats * gabor_damping).sum().item()
        else:
            expected_mass += sigmats.sum().item()
    
    grid_total_mass = grid.sum().item()
    print(f"\n=== Mass Conservation ===")
    print(f"Grid total (mass): {grid_total_mass:.6f}")
    print(f"Expected:          {expected_mass:.6f}")
    print(f"Ratio:             {grid_total_mass/expected_mass:.4f}")
    
    # Convert mass to density: divide by voxel volume
    voxel_volume = 8.0 / int(np.prod(grid_res))  # span is [-1,1]^3 = volume 8
    grid /= voxel_volume
    
    grid_total_density = grid.sum().item()
    print(f"\nVoxel volume: {voxel_volume:.10f}")
    print(f"Grid total (density): {grid_total_density:.2f}")
    print(f"Grid max (density):   {grid.max().item():.6f}")
    print(f"Grid min (density):   {grid.min().item():.6f}")
    
    # Check primitive positions
    for pyr in range(num_levels):
        means = pyr_prims[pyr]['means']
        print(f"\nLevel {pyr} primitive positions:")
        print(f"  Min: {means.min(dim=0).values.cpu().numpy()}")
        print(f"  Max: {means.max(dim=0).values.cpu().numpy()}")
        print(f"  Mean: {means.mean(dim=0).cpu().numpy()}")
        print(f"  Count: {means.shape[0]}")
    
    grid = torch.clamp(grid.reshape(grid_res), 0.0, None)
    span_dict = {
        'min': span[:3].cpu().numpy(),
        'max': span[3:].cpu().numpy()
    }
    
    # Visualize grid marginalizations
    print(f"\n=== Grid Visualization ===")
    grid_np = grid.cpu().numpy()
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sum over each axis
    sum_x = grid_np.sum(axis=0)  # Sum over X -> YZ plane
    sum_y = grid_np.sum(axis=1)  # Sum over Y -> XZ plane
    sum_z = grid_np.sum(axis=2)  # Sum over Z -> XY plane
    
    axes[0].imshow(sum_x.T, origin='lower', cmap='viridis')
    axes[0].set_title(f'Sum over X (YZ plane)\nSum: {sum_x.sum():.4f}')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('Z')
    
    axes[1].imshow(sum_y.T, origin='lower', cmap='viridis')
    axes[1].set_title(f'Sum over Y (XZ plane)\nSum: {sum_y.sum():.4f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    
    axes[2].imshow(sum_z.T, origin='lower', cmap='viridis')
    axes[2].set_title(f'Sum over Z (XY plane)\nSum: {sum_z.sum():.4f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('voxel_grid_marginalizations.png', dpi=150)
    print(f"Saved grid visualization to: voxel_grid_marginalizations.png")
    plt.close()
    
    return grid, span_dict


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Voxelize a gabor-field asset into a dense voxel grid (.npy)')
    ap.add_argument('--asset', required=True, help='Asset directory (data/root.primitives_pyr*.ply)')
    ap.add_argument('--output', default='voxelized_grid.npy', help='Output .npy grid path')
    ap.add_argument('--grid_res', type=int, nargs=3, default=[256, 256, 256], help='Grid resolution [x y z]')
    ap.add_argument('--num_levels', type=int, default=2, help='Number of pyramid levels')
    ap.add_argument('--num_samples', type=int, default=100_000, help='Base samples per primitive')
    ap.add_argument('--min_samples', type=int, default=1_000_000, help='Min samples per primitive (adaptive)')
    ap.add_argument('--max_samples', type=int, default=50_000_000, help='Max samples per primitive (adaptive)')
    args = ap.parse_args()

    grid, _span = voxelize_asset(
        primitive_dir=args.asset, grid_res=args.grid_res, num_levels=args.num_levels,
        num_samples=args.num_samples, min_samples=args.min_samples, max_samples=args.max_samples)
    np.save(args.output, grid.cpu().numpy())
    print(f'Saved voxel grid to {args.output}')


if __name__ == '__main__':
    main()
