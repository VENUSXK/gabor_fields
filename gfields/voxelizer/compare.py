# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Voxelize an asset and compare rendering of voxel grid vs original primitives.

Shallow comparison harness: voxelizes via gfields.voxelizer.voxelize and
renders both the dense grid and the analytic primitive mixture.

1. Voxelizes a volumetric primitives asset into a 3D grid (voxelize.voxelize_asset)
2. Renders the voxel grid using tomography and multiple scattering
3. Renders the original primitives using tomography and multiple scattering
4. Saves all outputs for comparison
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import gfields

# Add parent directory for imports


from gfields.voxelizer.voxelize import voxelize_asset


def render_voxel_grid(
    grid: torch.Tensor,
    span_dict: dict,
    output_path: str,
    multiple_scattering: bool = False,
    spp: int = 16,
    albedo: float = 0.9,
    volume_scale: float = 1.0,
    envmap_path: str = None,
    envmap_scale: float = 2.0,
):
    """
    Render a voxel grid using volume rendering.
    
    Args:
        grid: 3D tensor with voxel densities
        span_dict: Dictionary with 'min' and 'max' bounding box
        output_path: Path to save the rendered image
        multiple_scattering: Use multiple scattering or tomography
        spp: Samples per pixel
        albedo: Albedo value for multiple scattering
    """
    print(f"\n{'='*70}")
    print(f"RENDERING VOXEL GRID ({'Multiple Scattering' if multiple_scattering else 'Tomography'})")
    print(f"{'='*70}")
    
    # Convert grid to numpy, transpose to Mitsuba's (z, y, x) ordering, and clip
    grid_np = grid.cpu().numpy()
    grid_np = grid_np.transpose(2, 1, 0)  # (x,y,z) -> (z,y,x) for Mitsuba gridvolume
    grid_np = grid_np * volume_scale
    grid_np = np.clip(grid_np, 0.0, None)  # Clip negative densities only
    
    # Debug: Print grid statistics
    grid_orig = grid.cpu().numpy()
    print(f"Grid shape: {grid_np.shape}")
    print(f"Grid (before scale) - min: {grid_orig.min():.6f}, max: {grid_orig.max():.6f}, sum: {grid_orig.sum():.6f}")
    print(f"Grid (after scale={volume_scale} and clip to [0,1]) - min: {grid_np.min():.6f}, max: {grid_np.max():.6f}, sum: {grid_np.sum():.6f}")
    print(f"Non-zero voxels: {(grid_np > 0).sum()} / {grid_np.size} ({100*(grid_np > 0).sum()/grid_np.size:.2f}%)")
    print(f"Clipped voxels (hit 1.0 limit): {(grid_np >= 1.0).sum()} ({100*(grid_np >= 1.0).sum()/grid_np.size:.4f}%)")
    
    # Show density percentiles
    nonzero_densities = grid_np[grid_np > 0]
    if len(nonzero_densities) > 0:
        percentiles = np.percentile(nonzero_densities, [50, 75, 90, 95, 99])
        print(f"Non-zero density percentiles: 50%={percentiles[0]:.6f}, 75%={percentiles[1]:.6f}, 90%={percentiles[2]:.6f}, 95%={percentiles[3]:.6f}, 99%={percentiles[4]:.6f}")
    
    print(f"Bounding box: min={span_dict['min']}, max={span_dict['max']}")
    
    # Build scene dict
    T = mi.ScalarTransform4f
    
    if multiple_scattering:
        integrator_config = {
            'type': 'volpath',
            'max_depth': 8,
            'rr_depth': 8,
            'hide_emitters': True,
        }
        if envmap_path and os.path.exists(envmap_path):
            env_config = {
                'type': 'envmap',
                'filename': envmap_path,
                'scale': envmap_scale,
                'to_world': (T.rotate([0, 1, 0], 180.0) @
                             T.rotate([1, 0, 0], 0.0) @
                             T.rotate([0, 0, 1], 0.0)),
            }
        else:
            env_config = {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
            }
    else:
        integrator_config = {'type': 'volpath'}
        env_config = {'type': 'constant'}
    
    # Simple transform: gridvolume is in [0,1]^3, we scale by 2 and translate to [-1,1]^3
    # This matches the voxelization span
    gridvolume_to_world = T().scale([2.0, 2.0, 2.0]).translate([-0.5, -0.5, -0.5])
    
    print(f"Volume transform: scale([2, 2, 2]) -> translate([-0.5, -0.5, -0.5])")
    print(f"This maps [0,1]^3 grid space to [-1,1]^3 world space")
    
    cam_distance = 8.0
    cam_origin = np.array([0.0, 0.0, cam_distance])
    cam_target = np.array([0.0, 0.0, 0.0])
    
    if multiple_scattering:
        cam_dict = {
            'type': 'perspective',
            'fov': 15.0,
            'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
            'to_world': T().look_at(origin=list(cam_origin), target=list(cam_target), up=[0, 1, 0]),
        }
    else:
        cam_dict = {
            'type': 'orthographic',
            'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
            'to_world': T().look_at(origin=list(cam_origin), target=list(cam_target), up=[0, 1, 0]),
        }
    
    print(f"Camera: {cam_dict['type']}, distance={cam_distance}")
    
    scene_dict = {
        'type': 'scene',
        'integrator': integrator_config,
        'environment': env_config,
        'object': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'data': mi.TensorXf(grid_np),
                    'to_world': gridvolume_to_world,
                },
                'albedo': albedo if multiple_scattering else 0.0,
                'scale': 1.0,
            },
            'to_world': T(),
        },
        'cam': cam_dict,
    }
    
    # Render
    print(f"Loading scene...")
    scene = mi.load_dict(scene_dict)
    
    # Debug: Check scene bbox
    scene_bbox = scene.bbox()
    print(f"Scene bbox: min={np.array(scene_bbox.min)}, max={np.array(scene_bbox.max)}")
    
    # Test ray intersection
    print(f"\nTesting camera ray intersection with volume...")
    cam = scene.sensors()[0]
    
    # Cast a ray from camera center
    ray_origin = mi.Point3f(cam_origin)
    ray_dir = mi.Vector3f(cam_target - cam_origin) / np.linalg.norm(cam_target - cam_origin)
    ray = mi.Ray3f(ray_origin, ray_dir)
    
    si = scene.ray_intersect(ray)
    print(f"  Ray origin: {ray_origin}")
    print(f"  Ray direction: {ray_dir}")
    print(f"  Ray hits scene: {si.is_valid()}")
    if si.is_valid():
        print(f"  Hit point: {si.p}")
        print(f"  Hit distance: {si.t}")
    
    # Access volume grid from scene and verify data
    params = mi.traverse(scene)
    print(f"\nScene parameters:")
    for key in sorted(params.keys()):
        if 'sigma' in key.lower() or 'albedo' in key.lower() or 'scale' in key.lower():
            val = params[key]
            if hasattr(val, 'shape'):
                print(f"  {key}: shape={val.shape}")
                # If it's the sigma_t data, show statistics
                if 'sigma_t.data' in key:
                    val_np = np.array(val)
                    print(f"    -> min={val_np.min():.6f}, max={val_np.max():.6f}, mean={val_np.mean():.6f}, sum={val_np.sum():.6f}")
                    print(f"    -> non-zero: {(val_np > 0).sum()} / {val_np.size}")
            else:
                print(f"  {key}: {val}")
    
    print(f"\nRendering ({spp} spp)...")
    with gfields.benchmark.single_run('Rendering voxel grid'):
        img = mi.render(scene, spp=spp, seed=0)
    
    # Save
    mi.util.write_bitmap(output_path, img)
    print(f"Saved: {output_path}")
    
    # More detailed image statistics
    img_np = np.array(img)
    print(f"Image stats:")
    print(f"  Shape: {img_np.shape}")
    print(f"  Min: {img_np.min():.6f}, Max: {img_np.max():.6f}, Mean: {img_np.mean():.6f}")
    print(f"  Non-zero pixels: {(img_np > 1e-6).sum()} / {img_np.size} ({100*(img_np > 1e-6).sum()/img_np.size:.2f}%)")
    print(f"  Pixels == 1.0: {(np.abs(img_np - 1.0) < 1e-6).sum()}")


def render_primitives(
    primitive_dir: str,
    output_path: str,
    num_levels: int = 2,
    multiple_scattering: bool = False,
    spp: int = 16,
    albedo: float = 0.9,
    kernel_type: str = 'gabor_gaussian_mixture',
    envmap_path: str = None,
    envmap_scale: float = 2.0,
):
    """
    Render volumetric primitives asset.
    
    Args:
        primitive_dir: Path to asset directory
        output_path: Path to save rendered image
        num_levels: Number of pyramid levels
        multiple_scattering: Use multiple scattering or tomography
        spp: Samples per pixel
        albedo: Albedo value
        kernel_type: Kernel type
    """
    print(f"\n{'='*70}")
    print(f"RENDERING PRIMITIVES ({'Multiple Scattering' if multiple_scattering else 'Tomography'})")
    print(f"{'='*70}")
    
    T = mi.ScalarTransform4f

    # Build scene dict
    if multiple_scattering:
        integrator_config = {
            'type': 'gfields_prb',
            'max_depth': 8,
            'hide_emitters': True,
            'kernel_normalized': True,
            'kernel_type': kernel_type,
            'pyramid_level_count': num_levels,
            'asset_encoding': 'simple',
            'stochastic_selection_criteria': 'deterministic',
            'stochastic_selection_criteria_nee': 'deterministic',
            'solver_type': 'bisection',
            'solver_max_iterations': 32,
            'max_overlaps': 512,
            'rr_depth_primitive': 8192,
            'max_depth_primitive': 8192,
            'use_nee': True,
            'rr_depth': 1024,
            'rr_nee_depth': 1024,
        }
        if envmap_path and os.path.exists(envmap_path):
            env_config = {
                'type': 'envmap',
                'filename': envmap_path,
                'scale': envmap_scale,
                'to_world': (T().rotate([0, 1, 0], 180.0) @
                             T().rotate([1, 0, 0], 0.0) @
                             T().rotate([0, 0, 1], 0.0)),
            }
        else:
            env_config = {
                'type': 'constant',
                'radiance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]},
            }
    else:
        integrator_config = {
            'type': 'gfields_tomography',
            'max_depth': -1,
            'kernel_normalized': True,
            'kernel_full_range': True,
            'kernel_type': kernel_type,
            # Needed so the LODSelector's per-ray mask spans all levels; without
            # it plc defaults to 1 and only the Gaussian base (bit 0) renders.
            'pyramid_level_count': num_levels,
        }
        env_config = {'type': 'constant'}
    
    scene_dict = {
        'type': 'scene',
        'integrator': integrator_config,
        'environment': env_config,
    }
    
    # Load primitives for each level from the unified per-level PLY store
    for pyr in range(num_levels):
        a = gfields.io.load_ellipsoid_ply(
            os.path.join(primitive_dir, 'data', f'root.primitives_pyr{pyr}.ply'))
        centers = np.asarray(a['centers'], dtype=np.float32)
        scales = np.asarray(a['scales'], dtype=np.float32)
        quaternions = np.asarray(a['quaternions'], dtype=np.float32)

        if 'opacities' in a:
            opacities = a['opacities']
        elif 'sigma_t' in a:
            opacities = a['sigma_t']
        else:
            opacities = a['sigmat']
        opacities = np.asarray(opacities, dtype=np.float32)

        num_prims = centers.shape[0]

        if pyr > 0 and 'omega' in a:
            omega = np.asarray(a['omega'], dtype=np.float32)
        else:
            omega = np.zeros((num_prims, 1), dtype=np.float32)
        
        scene_dict[f'primitives_pyr{pyr}'] = {
            'type': 'ellipsoidsmesh',
            'shell': 'uv_sphere',
            'centers': mi.TensorXf(np.ravel(centers), shape=(num_prims, 3)),
            'scales': mi.TensorXf(np.ravel(scales), shape=(num_prims, 3)),
            'quaternions': mi.TensorXf(np.ravel(quaternions), shape=(num_prims, 4)),
            'opacities': mi.TensorXf(np.ravel(opacities), shape=(num_prims, 1)),
            'omega': mi.TensorXf(np.ravel(omega), shape=(num_prims, 1)),
            'albedo': mi.TensorXf(np.ones((num_prims, 3), dtype=np.float32) * albedo),
            'extent': 3.0,
            'visibility_mask': 2 ** pyr,
        }
        
        print(f"Level {pyr}: {num_prims} primitives loaded")
    
    cam_distance = 8.0
    cam_origin = np.array([0.0, 0.0, cam_distance])
    cam_target = np.array([0.0, 0.0, 0.0])
    
    if multiple_scattering:
        cam_dict = {
            'type': 'perspective',
            'fov': 15.0,
            'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
            'to_world': T().look_at(origin=list(cam_origin), target=list(cam_target), up=[0, 1, 0]),
        }
    else:
        cam_dict = {
            'type': 'orthographic',
            'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
            'to_world': T().look_at(origin=list(cam_origin), target=list(cam_target), up=[0, 1, 0]),
        }
    
    print(f"Camera: {cam_dict['type']}, distance={cam_distance}")
    scene_dict['cam'] = cam_dict
    
    # Render
    print(f"Loading scene...")
    scene = mi.load_dict(scene_dict)
    
    # Debug: Check scene bbox
    scene_bbox = scene.bbox()
    print(f"Scene bbox: min={np.array(scene_bbox.min)}, max={np.array(scene_bbox.max)}")
    
    print(f"Rendering ({spp} spp)...")
    with gfields.benchmark.single_run('Rendering primitives'):
        if multiple_scattering:
            img = mi.render(scene, spp=spp, seed=0)
        else:
            img = dr.clamp(mi.render(scene, spp=spp, seed=0), 0.0, 1.0)
    
    # Save
    mi.util.write_bitmap(output_path, img)
    print(f"Saved: {output_path}")
    print(f"Image stats: min={np.array(img).min():.6f}, max={np.array(img).max():.6f}, mean={np.array(img).mean():.6f}")


def run_controlled_test(args):
    """
    Controlled test: create known Gaussian/Gabor primitives, voxelize using
    both direct evaluation and importance sampling, render both, and compare.
    """
    import matplotlib.pyplot as plt
    T = mi.ScalarTransform4f
    device = 'cuda'
    
    print("\n" + "="*70)
    print("CONTROLLED TEST: Direct evaluation vs Importance Sampling")
    print("="*70)
    
    # Test grid
    res = [128, 128, 128]
    span_min = np.array([-1.0, -1.0, -1.0])
    span_max = np.array([1.0, 1.0, 1.0])
    side = span_max - span_min  # [2, 2, 2]
    voxel_size = side / np.array(res)
    voxel_volume = float(np.prod(voxel_size))
    
    print(f"Grid: {res}, voxel_size={voxel_size}, voxel_volume={voxel_volume:.8f}")
    
    # Test primitives: 1 Gaussian + 1 Gabor at known locations
    # Single Gaussian for clean calibration
    # A primitive is a Gabor iff its omega != 0 (a plain Gaussian otherwise).
    test_prims = [
        {'center': [0.0, 0.0, 0.0], 'scale': [0.3, 0.3, 0.3], 'sigma': 1.0, 'omega': 0.0},
    ]
    
    # Expected optical depth through center along z (from density_integral code):
    # density_integral = (1/|w|) * exp(-0.5*(C-B²)) / (2π * prod(s))
    # For center ray along z: |w| = 1/sz, so 1/|w| = sz; C=B²=0
    # = sz / (2π * prod(s))
    s = np.array(test_prims[0]['scale'])
    sigma = test_prims[0]['sigma']
    expected_di = s[2] / (2.0 * np.pi * np.prod(s))
    expected_tau = sigma * expected_di
    print(f"\nExpected center-ray analytical values:")
    print(f"  density_integral = {expected_di:.6f}")
    print(f"  optical_depth = sigma * di = {expected_tau:.6f}")
    print(f"  transmittance = exp(-tau) = {np.exp(-expected_tau):.6f}")
    
    # What normalization makes the grid line integral match?
    # ∫ (sigma * eval(x) / NORM) dx along z = sigma * density_integral
    # sigma/NORM * sz * sqrt(2π) = sigma * sz / (2π * prod(s))
    # NORM = sqrt(2π) * 2π * prod(s) / 1 = (2π)^(3/2) * prod(s)
    correct_norm = (2.0 * np.pi) ** 1.5 * np.prod(s)
    print(f"  Correct grid normalization: (2pi)^(3/2) * prod(s) = {correct_norm:.6f}")
    print(f"  Peak grid density should be: sigma / norm = {sigma / correct_norm:.6f}")
    
    # ==========================================================================
    # Method 1: Direct evaluation on grid (ground truth density)
    # ==========================================================================
    print("\n--- Method 1: Direct evaluation (ground truth) ---")
    
    # Create world-space coordinates for each voxel center
    x_coords = torch.linspace(span_min[0] + voxel_size[0]/2, span_max[0] - voxel_size[0]/2, res[0], device=device)
    y_coords = torch.linspace(span_min[1] + voxel_size[1]/2, span_max[1] - voxel_size[1]/2, res[1], device=device)
    z_coords = torch.linspace(span_min[2] + voxel_size[2]/2, span_max[2] - voxel_size[2]/2, res[2], device=device)
    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = torch.stack([X, Y, Z], dim=-1)  # (res, res, res, 3)
    
    grid_direct = torch.zeros(res, device=device, dtype=torch.float32)
    
    for prim in test_prims:
        center = torch.tensor(prim['center'], device=device, dtype=torch.float32)
        scale = torch.tensor(prim['scale'], device=device, dtype=torch.float32)
        sigma = prim['sigma']
        omega = prim['omega']
        
        # Grid stores sigma * eval(x) / [(2π)^(3/2) * prod(s)]
        # This ensures ∫ grid(ray(t)) dt = sigma * density_integral(ray)
        p = (grid_points - center) / scale
        gaussian = torch.exp(-0.5 * torch.sum(p * p, dim=-1))
        norm = (2.0 * np.pi) ** 1.5 * scale.prod().item()
        if omega != 0:
            cosine = torch.cos(omega * torch.sum(p, dim=-1))
            kernel_val = gaussian * cosine / norm
        else:
            kernel_val = gaussian / norm
        
        grid_direct += sigma * kernel_val
    
    print(f"  Direct grid: min={grid_direct.min():.6f}, max={grid_direct.max():.6f}, sum={grid_direct.sum():.6f}")
    print(f"  Non-zero: {(grid_direct.abs() > 1e-8).sum()}")
    
    # ==========================================================================
    # Method 2: Importance sampling (our voxelization method)
    # ==========================================================================
    print("\n--- Method 2: Importance sampling ---")
    
    num_samples = 500_000
    local_samples = torch.randn(num_samples, 3, device=device, dtype=torch.float32)
    
    grid_res_t = torch.tensor(res, device=device, dtype=torch.float32)
    span_t = torch.tensor(np.concatenate([span_min, span_max]), device=device, dtype=torch.float32)
    side_t = span_t[3:] - span_t[:3]
    
    grid_is = torch.zeros(int(np.prod(res)), device=device, dtype=torch.float32)
    
    for prim in test_prims:
        center = torch.tensor(prim['center'], device=device, dtype=torch.float32)
        scale = torch.tensor(prim['scale'], device=device, dtype=torch.float32)
        sigma = prim['sigma']
        omega = prim['omega']
        
        # Transform samples to world space: x = scale * u + center (R = I)
        pts_world = local_samples * scale.unsqueeze(0) + center.unsqueeze(0)
        
        # Compute voxel indices
        lin_indices = (pts_world - span_t[0:3].unsqueeze(0)) / side_t.unsqueeze(0)
        lin_indices = torch.floor(lin_indices * grid_res_t.unsqueeze(0)).long()
        
        valid_mask = (
            (lin_indices[:, 0] >= 0) & (lin_indices[:, 0] < res[0]) &
            (lin_indices[:, 1] >= 0) & (lin_indices[:, 1] < res[1]) &
            (lin_indices[:, 2] >= 0) & (lin_indices[:, 2] < res[2])
        )
        
        lin_indices_flat = (
            lin_indices[:, 0] * res[1] * res[2] +
            lin_indices[:, 1] * res[2] +
            lin_indices[:, 2]
        ).long()
        
        # Importance sampling from N(0,I):
        # IS from N(0,I) with (2π)^(3/2)*prod(s) norm: factors cancel exactly
        # contribution = sigma * cos_term / N, then divide by voxel_vol
        if omega != 0:
            cosine_term = torch.cos(omega * local_samples.sum(dim=-1))
            contributions = (sigma * cosine_term / num_samples)
        else:
            contributions = torch.full((num_samples,), sigma / num_samples,
                                      device=device, dtype=torch.float32)
        
        grid_is.scatter_add_(0, lin_indices_flat[valid_mask], contributions[valid_mask])
    
    # Convert mass to density by dividing by voxel volume
    grid_is_density = grid_is / voxel_volume
    grid_is_density = grid_is_density.reshape(res)
    grid_is_mass = grid_is.reshape(res)
    
    print(f"  IS grid (mass): min={grid_is_mass.min():.6f}, max={grid_is_mass.max():.6f}, sum={grid_is_mass.sum():.6f}")
    print(f"  IS grid (density = mass/voxel_vol): min={grid_is_density.min():.6f}, max={grid_is_density.max():.6f}, sum={grid_is_density.sum():.6f}")
    
    # ==========================================================================
    # Compare the two methods
    # ==========================================================================
    print("\n--- Comparison ---")
    
    diff = (grid_direct - grid_is_density).abs()
    nonzero = grid_direct.abs() > 1e-6
    if nonzero.any():
        rel_err = diff[nonzero] / grid_direct[nonzero].abs()
        print(f"  Max abs diff:    {diff.max():.6f}")
        print(f"  Mean abs diff:   {diff.mean():.8f}")
        print(f"  Mean rel error:  {rel_err.mean()*100:.2f}%")
        print(f"  Max rel error:   {rel_err.max()*100:.2f}%")
        corr = torch.corrcoef(torch.stack([grid_direct.flatten(), grid_is_density.flatten()]))[0, 1].item()
        print(f"  Correlation:     {corr:.6f}")
        
        print(f"\n  Direct peak density:  {grid_direct.max():.6f}")
        print(f"  IS peak density:      {grid_is_density.max():.6f}")
        print(f"  Ratio (IS/Direct):    {grid_is_density.max() / grid_direct.max():.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    mid = res[2] // 2
    
    axes[0, 0].imshow(grid_direct[:, :, mid].cpu().numpy().T, origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Direct eval (Z slice)\nmax={grid_direct.max():.4f}')
    axes[0, 1].imshow(grid_direct.sum(dim=2).cpu().numpy().T, origin='lower', cmap='viridis')
    axes[0, 1].set_title(f'Direct eval (sum over Z)\nsum={grid_direct.sum():.4f}')
    axes[0, 2].imshow(grid_direct.sum(dim=0).cpu().numpy().T, origin='lower', cmap='viridis')
    axes[0, 2].set_title('Direct eval (sum over X)')
    
    axes[1, 0].imshow(grid_is_density[:, :, mid].cpu().numpy().T, origin='lower', cmap='viridis')
    axes[1, 0].set_title(f'IS density (Z slice)\nmax={grid_is_density.max():.4f}')
    axes[1, 1].imshow(grid_is_density.sum(dim=2).cpu().numpy().T, origin='lower', cmap='viridis')
    axes[1, 1].set_title(f'IS density (sum over Z)\nsum={grid_is_density.sum():.4f}')
    axes[1, 2].imshow(grid_is_density.sum(dim=0).cpu().numpy().T, origin='lower', cmap='viridis')
    axes[1, 2].set_title('IS density (sum over X)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'controlled_test_comparison.png'), dpi=150)
    print(f"\nSaved comparison plot to: {os.path.join(args.output, 'controlled_test_comparison.png')}")
    plt.close()
    
    # ==========================================================================
    # Render the direct-eval grid with Mitsuba (verify rendering pipeline)
    # ==========================================================================
    print("\n--- Rendering tests ---")
    
    for label, grid_data in [('direct', grid_direct), ('importance_sampled', grid_is_density)]:
        grid_np = grid_data.cpu().numpy()
        grid_np = np.clip(grid_np, 0.0, None)
        
        print(f"\n  Rendering '{label}' grid: shape={grid_np.shape}, min={grid_np.min():.6f}, max={grid_np.max():.6f}")
        
        gridvolume_to_world = T().scale([2.0, 2.0, 2.0]).translate([-0.5, -0.5, -0.5])
        
        scene_dict = {
            'type': 'scene',
            'integrator': {'type': 'volpath'},
            'object': {
                'type': 'cube',
                'bsdf': {'type': 'null'},
                'interior': {
                    'type': 'heterogeneous',
                    'sigma_t': {
                        'type': 'gridvolume',
                        'data': mi.TensorXf(grid_np),
                        'to_world': gridvolume_to_world,
                    },
                    'albedo': 0.0,
                    'scale': 1.0,
                },
                'to_world': T(),
            },
            'environment': {'type': 'constant'},
            'cam': {
                'type': 'orthographic',
                'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
                'to_world': T().look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            },
        }
        
        scene = mi.load_dict(scene_dict)
        img = mi.render(scene, spp=1024, seed=0)
        out_path = os.path.join(args.output, f'controlled_{label}.exr')
        mi.util.write_bitmap(out_path, img)
        img_np = np.array(img)
        print(f"  Image: min={img_np.min():.6f}, max={img_np.max():.6f}, mean={img_np.mean():.6f}")
        print(f"  Saved: {out_path}")
    
    # ==========================================================================
    # Render the same primitives using volprim_tomography
    # ==========================================================================
    print("\n  Rendering primitives with volprim_tomography...")
    
    prim_scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'gfields_tomography',
            'max_depth': -1,
            'kernel_normalized': True,
            'kernel_full_range': True,
            'kernel_type': 'gabor_gaussian_mixture',
            'pyramid_level_count': 2,  # gauss base (pyr0) + gabor (pyr1)
        },
        'environment': {'type': 'constant'},
        'cam': {
            'type': 'orthographic',
            'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
            'to_world': T().look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
        },
    }
    
    # Build primitives
    all_centers = []
    all_scales = []
    all_quats = []
    all_sigmas = []
    all_omegas = []
    all_albedos = []
    
    for prim in test_prims:
        all_centers.append(prim['center'])
        all_scales.append(prim['scale'])
        all_quats.append([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        all_sigmas.append([prim['sigma']])
        all_omegas.append([prim['omega']])
        all_albedos.append([0.0, 0.0, 0.0])
    
    n = len(test_prims)
    
    # Gaussian level (pyr0): split on omega (0 => Gaussian, nonzero => Gabor).
    gauss_mask = [p['omega'] == 0 for p in test_prims]
    gabor_mask = [p['omega'] != 0 for p in test_prims]
    
    gauss_idx = [i for i, m in enumerate(gauss_mask) if m]
    gabor_idx = [i for i, m in enumerate(gabor_mask) if m]
    
    if gauss_idx:
        ng = len(gauss_idx)
        prim_scene_dict['primitives_pyr0'] = {
            'type': 'ellipsoidsmesh',
            'shell': 'uv_sphere',
            'centers': mi.TensorXf(np.array([all_centers[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 3)),
            'scales': mi.TensorXf(np.array([all_scales[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 3)),
            'quaternions': mi.TensorXf(np.array([all_quats[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 4)),
            'opacities': mi.TensorXf(np.array([all_sigmas[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 1)),
            'omega': mi.TensorXf(np.array([all_omegas[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 1)),
            'albedo': mi.TensorXf(np.array([all_albedos[i] for i in gauss_idx], dtype=np.float32).ravel(), shape=(ng, 3)),
            'extent': 3.0,
            'visibility_mask': 1,
        }
    
    if gabor_idx:
        ngb = len(gabor_idx)
        prim_scene_dict['primitives_pyr1'] = {
            'type': 'ellipsoidsmesh',
            'shell': 'uv_sphere',
            'centers': mi.TensorXf(np.array([all_centers[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 3)),
            'scales': mi.TensorXf(np.array([all_scales[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 3)),
            'quaternions': mi.TensorXf(np.array([all_quats[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 4)),
            'opacities': mi.TensorXf(np.array([all_sigmas[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 1)),
            'omega': mi.TensorXf(np.array([all_omegas[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 1)),
            'albedo': mi.TensorXf(np.array([all_albedos[i] for i in gabor_idx], dtype=np.float32).ravel(), shape=(ngb, 3)),
            'extent': 3.0,
            'visibility_mask': 2,
        }
    
    prim_scene = mi.load_dict(prim_scene_dict)
    img_prims = mi.render(prim_scene, spp=64, seed=0)
    out_path = os.path.join(args.output, 'controlled_primitives.exr')
    mi.util.write_bitmap(out_path, img_prims)
    img_prims_np = np.array(img_prims)
    print(f"  Image: min={img_prims_np.min():.6f}, max={img_prims_np.max():.6f}, mean={img_prims_np.mean():.6f}")
    print(f"  Saved: {out_path}")
    
    print("\n" + "="*70)
    print("CONTROLLED TEST COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Voxelize asset and compare voxel vs primitive rendering'
    )
    
    parser.add_argument('--asset', type=str,
                       default=r'C:\Users\jorge\repos\volumetric_primitives\bunny_gabor_1024\optimized_asset_pyr1',
                       help='Path to the asset directory')
    parser.add_argument('--output', type=str, default='./voxel_comparison',
                       help='Output directory for results')
    parser.add_argument('--grid_res', type=int, nargs=3, default=[577, 572, 438],
                       help='Voxel grid resolution [x y z] (default matches bunny target_grid.npy)')
    parser.add_argument('--num_levels', type=int, default=2,
                       help='Number of pyramid levels')
    parser.add_argument('--num_samples', type=int, default=100_000,
                       help='Base number of samples per primitive for voxelization')
    parser.add_argument('--min_samples', type=int, default=1_000_000,
                       help='Minimum samples per primitive (adaptive)')
    parser.add_argument('--max_samples', type=int, default=50_000_000,
                       help='Maximum samples per primitive (adaptive)')
    parser.add_argument('--spp', type=int, default=64,
                       help='Samples per pixel for rendering')
    parser.add_argument('--albedo', type=float, default=0.9,
                       help='Albedo value for multiple scattering')
    parser.add_argument('--envmap', type=str,
                       default='./data/envmaps/qwantani_dusk_2_1k.exr',
                       help='Path to environment map for multiple scattering')
    parser.add_argument('--envmap_scale', type=float, default=2.0,
                       help='Environment map intensity scale')
    parser.add_argument('--kernel_type', type=str, default='gabor_gaussian_mixture',
                       help='Kernel type for primitives')
    parser.add_argument('--tomography_only', action='store_true',
                       help='Only render tomography mode (skip multiple scattering)')
    parser.add_argument('--volume_scale', type=float, default=1.0,
                       help='Scale factor for voxel grid densities (default: 1.0)')
    parser.add_argument('--force_revoxelize', action='store_true',
                       help='Force re-voxelization even if cached grid exists')
    parser.add_argument('--test_controlled', action='store_true',
                       help='Run controlled test with known Gaussian/Gabor primitives')
    parser.add_argument('--test_volume', action='store_true',
                       help='Test with a simple synthetic volume instead of voxelizing asset')
    parser.add_argument('--reference_grid', type=str,
                       default=r'C:\Users\jorge\repos\volumetric_primitives\resources\vdb\bunny_cloud.npy',
                       help='Path to reference voxel grid (.npy) for the original asset')
    parser.add_argument('--reference_opacity_scale', type=float, default=10.0,
                       help='Opacity scale for the reference grid (BunnyConfig=10.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Controlled accuracy test
    if args.test_controlled:
        run_controlled_test(args)
        return
    
    # Step 1: Create or voxelize grid (with caching)
    grid_path = os.path.join(args.output, 'voxelized_grid.npy')
    grid_meta_path = os.path.join(args.output, 'voxelized_grid_meta.npy')
    
    if args.test_volume:
        print("\n" + "="*70)
        print("TEST MODE: Creating synthetic volume")
        print("="*70)
        
        # Create a simple sphere in the center
        res = args.grid_res
        grid = torch.zeros(res, dtype=torch.float32, device='cuda')
        
        # Create coordinate grids
        x = torch.linspace(0, 1, res[0], device='cuda')
        y = torch.linspace(0, 1, res[1], device='cuda')
        z = torch.linspace(0, 1, res[2], device='cuda')
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Sphere centered at (0.5, 0.5, 0.5) with radius 0.25
        dist = torch.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
        grid[dist < 0.25] = 0.01  # Constant density inside sphere
        
        print(f"Created test sphere: {(grid > 0).sum()} voxels, sum={grid.sum():.6f}")
        
        span_dict = {
            'min': np.array([-1.0, -1.0, -1.0]),
            'max': np.array([1.0, 1.0, 1.0])
        }
    else:
        # Check cache: reuse voxelized grid if asset/res/samples haven't changed
        cache_valid = False
        if not args.force_revoxelize and os.path.exists(grid_path) and os.path.exists(grid_meta_path):
            meta = np.load(grid_meta_path, allow_pickle=True).item()
            if (meta.get('asset') == args.asset and
                meta.get('grid_res') == args.grid_res and
                meta.get('num_levels') == args.num_levels and
                meta.get('num_samples') == args.num_samples):
                print(f"\nLoading cached voxelized grid from: {grid_path}")
                grid = torch.tensor(np.load(grid_path), device='cuda', dtype=torch.float32)
                span_dict = {'min': np.array([-1., -1., -1.]), 'max': np.array([1., 1., 1.])}
                cache_valid = True
        
        if not cache_valid:
            grid, span_dict = voxelize_asset(
                primitive_dir=args.asset,
                grid_res=args.grid_res,
                num_levels=args.num_levels,
                num_samples=args.num_samples,
                min_samples=args.min_samples,
                max_samples=args.max_samples,
            )
            # Save grid and metadata
            np.save(grid_path, grid.cpu().numpy())
            np.save(grid_meta_path, {
                'asset': args.asset, 'grid_res': args.grid_res,
                'num_levels': args.num_levels, 'num_samples': args.num_samples,
            })
            print(f"\nSaved voxel grid to: {grid_path}")
    
    # Step 2: Render voxel grid (tomography)
    render_voxel_grid(
        grid=grid,
        span_dict=span_dict,
        output_path=os.path.join(args.output, 'voxel_tomography.exr'),
        multiple_scattering=False,
        spp=args.spp,
        albedo=0.0,
        volume_scale=args.volume_scale
    )
    
    # Step 3: Render primitives (tomography) - always re-rendered
    render_primitives(
        primitive_dir=args.asset,
        output_path=os.path.join(args.output, 'primitives_tomography.exr'),
        num_levels=args.num_levels,
        multiple_scattering=False,
        spp=args.spp,
        albedo=0.0,
        kernel_type=args.kernel_type
    )
    
    # Optional: Multiple scattering renders
    if not args.tomography_only:
        # Step 4: Render voxel grid (multiple scattering)
        render_voxel_grid(
            grid=grid,
            span_dict=span_dict,
            output_path=os.path.join(args.output, 'voxel_multiscatter.exr'),
            multiple_scattering=True,
            spp=args.spp,
            albedo=args.albedo,
            volume_scale=args.volume_scale,
            envmap_path=args.envmap,
            envmap_scale=args.envmap_scale,
        )
        
        # Step 5: Render primitives (multiple scattering) - always re-rendered
        render_primitives(
            primitive_dir=args.asset,
            output_path=os.path.join(args.output, 'primitives_multiscatter.exr'),
            num_levels=args.num_levels,
            multiple_scattering=True,
            spp=args.spp,
            albedo=args.albedo,
            kernel_type=args.kernel_type,
            envmap_path=args.envmap,
            envmap_scale=args.envmap_scale,
        )
    
    # Step 6: Render reference voxel grid (original asset)
    # Rendered directly matching gabor_volume_regression_utils.py (no transpose!)
    if args.reference_grid and os.path.exists(args.reference_grid):
        print(f"\n{'='*70}")
        print(f"RENDERING REFERENCE GRID: {args.reference_grid}")
        print(f"{'='*70}")
        
        ref_grid_np = np.load(args.reference_grid)
        print(f"Reference grid: shape={ref_grid_np.shape}, min={ref_grid_np.min():.4f}, max={ref_grid_np.max():.4f}")
        print(f"Opacity scale: {args.reference_opacity_scale}")
        
        T = mi.ScalarTransform4f
        ref_to_world = T().scale([2, 2, 2]).translate([-0.5, -0.5, -0.5])
        
        def render_reference(output_path, ms, albedo_val, spp):
            if ms:
                integ = {'type': 'volpath', 'max_depth': 8, 'rr_depth': 8, 'hide_emitters': True}
                if args.envmap and os.path.exists(args.envmap):
                    env = {'type': 'envmap', 'filename': args.envmap, 'scale': args.envmap_scale,
                           'to_world': (T.rotate([0,1,0], 180.0) @ T.rotate([1,0,0], 0.0) @ T.rotate([0,0,1], 0.0))}
                else:
                    env = {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [1., 1., 1.]}}
                cam = {'type': 'perspective', 'fov': 15.0,
                       'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
                       'to_world': T().look_at(origin=[0, 0, 8], target=[0, 0, 0], up=[0, 1, 0])}
            else:
                integ = {'type': 'volpath'}
                env = {'type': 'constant'}
                cam = {'type': 'orthographic',
                       'film': {'type': 'hdrfilm', 'width': 800, 'height': 800},
                       'to_world': T().look_at(origin=[0, 0, 8], target=[0, 0, 0], up=[0, 1, 0])}
            
            scene = mi.load_dict({
                'type': 'scene',
                'integrator': integ,
                'object': {
                    'type': 'cube',
                    'bsdf': {'type': 'null'},
                    'interior': {
                        'type': 'heterogeneous',
                        'sigma_t': {
                            'type': 'gridvolume',
                            'data': mi.TensorXf(ref_grid_np),
                            'to_world': ref_to_world,
                        },
                        'albedo': albedo_val,
                        'scale': args.reference_opacity_scale,
                    },
                    'to_world': T(),
                },
                'environment': env,
                'cam': cam,
            })
            print(f"  Rendering {'MS' if ms else 'tomography'} ({spp} spp)...")
            img = mi.render(scene, spp=spp, seed=0)
            mi.util.write_bitmap(output_path, img)
            img_np = np.array(img)
            print(f"  Image: min={img_np.min():.6f}, max={img_np.max():.6f}, mean={img_np.mean():.6f}")
            print(f"  Saved: {output_path}")
        
        render_reference(os.path.join(args.output, 'reference_tomography.exr'), False, 0.0, args.spp)
        
        if not args.tomography_only:
            render_reference(os.path.join(args.output, 'reference_multiscatter.exr'), True, args.albedo, args.spp)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput files in: {args.output}")
    print("  - voxelized_grid.npy (voxel data)")
    print("  - voxel_tomography.exr (voxel grid, tomography)")
    print("  - primitives_tomography.exr (primitives, tomography)")
    if args.reference_grid and os.path.exists(args.reference_grid):
        print("  - reference_tomography.exr (original voxel grid, tomography)")
    if not args.tomography_only:
        print("  - voxel_multiscatter.exr (voxel grid, multiple scattering)")
        print("  - primitives_multiscatter.exr (primitives, multiple scattering)")
        if args.reference_grid and os.path.exists(args.reference_grid):
            print("  - reference_multiscatter.exr (original voxel grid, multiple scattering)")


if __name__ == '__main__':
    main()

