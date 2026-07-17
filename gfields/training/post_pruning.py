# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pathlib import Path
from typing import Literal, Tuple, Dict, Any
from datetime import datetime
import shutil

import numpy as np
import json
import gfields.training.regression as gu
import mitsuba as mi
import drjit as dr
from argparse import ArgumentParser, Namespace

from colorist import green, effect_dim, blue, yellow

from gfields.optimizers import BoundedAdam
from gfields.losses import psnr


def parse_args() -> Namespace:
    """
    Parses command-line arguments for the post-pruning script.
    Example usage:
    gfields decompose prune -b 0.3 -r opacity -p results/checkpoints/PATH
    gfields decompose prune -b 0.3 -r opacity -p results/checkpoints/PATH -o pruned_outputs/my_asset --save_assets
    gfields decompose prune -b 1.0 -r density -p results/checkpoints/PATH -l gauss  # prune only Gaussians

    :return: Parsed arguments as a Namespace object.
    """
    parser = ArgumentParser(description="Post-pruning of Gabor mixture models based on specified criteria.")
    parser.add_argument('--model_path', '-p', type=str, required=True, help='Path to the saved Gabor mixture model.')
    parser.add_argument('--pruning_rule', '-r', type=str, choices=['density', 'opacity'], required=True, help='Criterion for pruning components.')
    parser.add_argument('--psnr_budget', '-b', type=float, default=0.3, help='PSNR budget for pruning (if applicable).')
    parser.add_argument('--psnr_target', '-t', type=float, default=0.0, help='PSNR target for pruning (if applicable).')
    parser.add_argument('--spp', '-s', type=int, default=8, help='Samples per pixel for rendering during pruning evaluation.')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output directory for pruned assets and stats. Defaults to model_path/pruned_{rule}/')
    parser.add_argument('--save_assets', '-a', action='store_true', help='Save the pruned model assets (npy + ply files).')
    parser.add_argument('--no_stats', action='store_true', help='Do not save pruning statistics JSON.')
    parser.add_argument('--level', '-l', type=str, default='auto', choices=['auto', 'gauss', 'gabor', '0', '1'],
                        help='Which level to prune: "auto" (highest available), "gauss"/"0" (Gaussians), "gabor"/"1" (Gabors).')
    args = parser.parse_args()
    args.model_path = Path(args.model_path)
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)
    return args

def prepare_mixture(path: Path) -> Tuple[bool, mi.Scene, mi.SceneParameters, BoundedAdam, mi.Sensor, mi.TensorXf]:
    """
    Prepares the mixture model, scene, and related parameters for pruning.

    :param path: Path to the saved Gabor mixture model.
    :type path: Path
    :return: A tuple containing:
        - has_gabor: Boolean indicating if the model has a Gabor level.
        - scene: The Mitsuba scene object.
        - scene_params: The scene parameters for optimization.
        - adam: The BoundedAdam optimizer with model parameters.
        - cams: The camera sensor array.
        - ref_images: The reference images for PSNR computation.
    """
    model = gu.load_model(path, -1)
    has_gabor = 'primitives_pyr1' in model.keys()
    assert 'primitives_pyr2' not in model.keys(), "Pyramids with more than 2 levels are not supported."
    max_level = int(has_gabor)

    # Log what was loaded
    prim_keys = [k for k in model.keys() if k.startswith('primitives_pyr')]
    blue(f"Loaded model with levels: {prim_keys} (has_gabor={has_gabor}, max_level={max_level})")
    
    scene: mi.Scene = mi.load_dict(model)
    scene_params = mi.traverse(scene)
    adam = BoundedAdam()
    for lvl in range(max_level + 1):
        gu.write_scene_params_to_optimizer(scene_params, adam, level=lvl)
        gu.write_optimizer_params_to_scene(adam, scene_params, level=lvl)
    
    # args.json / references live in the model dir (legacy combined layout) or
    # one level up (train.py writes them to the run root, the asset to a
    # optimized_asset_pyr* subdir); accept either.
    def _find(fname: str) -> Path:
        for base in (path, path.parent):
            if (base / fname).exists():
                return base / fname
        return path / fname  # fall through -> clear FileNotFoundError below

    # read optimization params
    with open(str(_find('args.json')), "r") as f:
        optim_args = json.load(f)
        if "sample_single_hemisphere" not in optim_args:
            optim_args['sample_single_hemisphere'] = optim_args.get('use_platonic_rotations', False)

    cams = gu.create_cam_array(gu.prepare_cameras(
        cam_count=optim_args['cam_count'], cam_res_x=optim_args['cam_res_x'], cam_res_y=optim_args['cam_res_y'],
        cam_scale=optim_args['cam_scale'], sample_single_hemisphere=optim_args['sample_single_hemisphere'], cam_y_offset=optim_args.get('cam_y_offset', 0.0)
    ))
    ref_images = mi.TensorXf(mi.Bitmap(str(_find(f'reference_pyr{max_level}.exr'))))

    return has_gabor, scene, scene_params, adam, cams, ref_images



def reorder_based_on_rule(rule: Literal['density', 'opacity'], opt: BoundedAdam, level: int) -> Tuple[mi.TensorXf, np.ndarray, np.ndarray]:
    """
    Reorders the primitives based on the specified pruning rule.

    :param rule: The pruning rule to use ('density' or 'opacity').
    :param opt: The BoundedAdam optimizer containing model parameters.
    :param level: The pyramid level to consider (0 for Gaussians, 1 for Gabors).
    :return: A tuple containing:
        - weight: The computed weights for each primitive.
        - weight_np: The weights as a flattened NumPy array.
        - ordering: The indices that would sort the weights in ascending order.
    """
    if rule == 'density':
        weight = dr.abs(opt[f'opacities_{level}'] / (dr.sqrt(8.0 * dr.pi * dr.pi * dr.pi) * dr.prod(opt[f'scales_{level}'])))
    elif rule == 'opacity':
        weight = dr.abs(opt[f'opacities_{level}'])
    else:
        raise ValueError(f"Unknown pruning rule: {rule}")
    weight_np = weight.numpy().ravel()  # flatten to 1D
    ordering = np.argsort(weight_np)  # ascending order
    return weight, weight_np, ordering


def render(
        threshold: float, *, level: int, weight: mi.TensorXf, original_opacity: mi.TensorXf,
        adam: BoundedAdam, scene: mi.Scene, scene_params: mi.SceneParameters,
        cams: mi.Sensor, spp: int, ref_images: mi.TensorXf) -> Tuple[float, int, mi.TensorXf, np.ndarray]:
    """
    Renders the scene with primitives pruned based on the given threshold and computes PSNR.

    :param threshold: The pruning threshold.
    :param level: The pyramid level to consider.
    :param weight: The computed weights for each primitive.
    :param original_opacity: The original opacity values of the primitives.
    :param adam: The BoundedAdam optimizer containing model parameters.
    :param scene: The Mitsuba scene object.
    :param scene_params: The scene parameters for optimization.
    :param cams: The camera sensor array.
    :param spp: Samples per pixel for rendering.
    :param ref_images: The reference images for PSNR computation.
    :return: A tuple containing:
        - psnr_val: The computed PSNR value.
        - removed: The number of primitives removed.
        - img: The rendered image after pruning.
        - keep_mask: Boolean mask of primitives to keep (True = keep, False = prune).
    """
    prune_mask = weight <= threshold
    adam[f'opacities_{level}'] = dr.select(prune_mask, 0.0, original_opacity)
    gu.write_optimizer_params_to_scene(adam, scene_params, level=level)
    img = mi.render(scene, scene_params, cams, spp=spp, seed=0)
    psnr_val = psnr(ref_images, dr.clip(img, 0, 1))
    keep_mask = ~prune_mask.numpy().ravel()  # flatten to 1D boolean mask

    return psnr_val.array[0], int(prune_mask.numpy().sum()), img, keep_mask


def _apply_mask_to_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a boolean mask to an array along the primitives axis.
    Handles both 1D arrays (N,) and 2D arrays (D, N) where N is num primitives.
    """
    if arr.ndim == 1:
        return arr[mask]
    elif arr.ndim == 2:
        # Primitives are along last axis (D, N)
        return arr[:, mask]
    else:
        raise ValueError(f"Unexpected array shape: {arr.shape}")


def save_pruned_assets(
        output_path: Path, 
        model_path: Path,
        adam: BoundedAdam, 
        keep_mask: np.ndarray, 
        pruned_level: int,
        max_level: int
) -> None:
    """
    Saves the pruned model assets to disk, removing pruned primitives entirely.
    Uses gu.save_model for compatibility with the rest of the codebase.

    :param output_path: Directory to save the pruned assets.
    :param model_path: Path to the original model (for copying args.json).
    :param adam: The BoundedAdam optimizer containing model parameters.
    :param keep_mask: Boolean mask indicating which primitives to keep (1D).
    :param pruned_level: The pyramid level that was pruned.
    :param max_level: The highest pyramid level in the model (0 for Gauss-only, 1 for Gauss-Gabor).
    """
    from copy import deepcopy
    
    import sys
    blue(f"  Saving {max_level + 1} pyramid level(s) (pruned_level={pruned_level}, max_level={max_level})")
    sys.stdout.flush()

    # Build a new model dict with pruned primitives
    # Start with the base scene structure
    pruned_model = {
        'type': 'scene',
        'integrator': {
            'type': 'gfields_tomography',
            'max_depth': -1,
            'kernel_normalized': True,
            'kernel_full_range': True,
            'kernel_type': "gabor_gaussian_mixture",
        },
        'environment': { 'type': 'constant' }
    }

    # Add each pyramid level
    for l in range(max_level + 1):
        print(f"    Processing level {l}...", flush=True)
        
        # Get params from optimizer and convert to TensorXf
        # Optimizer stores as (D, N), we need (N, D) for create_scene_dict_from_primitives
        try:
            centers_raw = adam[f'centers_{l}'].numpy()
            scales_raw = adam[f'scales_{l}'].numpy()
            quats_raw = adam[f'quaternions_{l}'].numpy()
            opacities_raw = adam[f'opacities_{l}'].numpy()
            albedo_raw = adam[f'albedo_{l}'].numpy()
            omega_raw = adam[f'omega_{l}'].numpy()
            print(f"      Raw shapes: centers={centers_raw.shape}, omega={omega_raw.shape}", flush=True)
        except Exception as e:
            print(f"      ERROR getting optimizer data for level {l}: {e}", flush=True)
            raise
        
        # Transpose to (N, D) format
        print(f"      Transposing...", flush=True)
        centers = centers_raw.T if centers_raw.ndim == 2 else centers_raw.reshape(-1, 3)
        scales = scales_raw.T if scales_raw.ndim == 2 else scales_raw.reshape(-1, 3)
        quats = quats_raw.T if quats_raw.ndim == 2 else quats_raw.reshape(-1, 4)
        opacities = opacities_raw.T if opacities_raw.ndim == 2 else opacities_raw.reshape(-1, 1)
        albedo = albedo_raw.T if albedo_raw.ndim == 2 else albedo_raw.reshape(-1, 3)
        omega = omega_raw.T if omega_raw.ndim == 2 else omega_raw.reshape(-1, 1) if omega_raw.ndim == 1 else omega_raw
        print(f"      Transposed OK", flush=True)
        
        n_prims_before = centers.shape[0]
        
        # Apply mask only to the pruned level
        if l == pruned_level:
            print(f"      Applying mask: keep_mask.shape={keep_mask.shape}, sum={keep_mask.sum()}, n_prims={n_prims_before}", flush=True)
            print(f"      Array shapes: centers={centers.shape}, scales={scales.shape}, quats={quats.shape}", flush=True)
            print(f"                    opacities={opacities.shape}, albedo={albedo.shape}, omega={omega.shape}", flush=True)
            print(f"      centers[mask]...", flush=True)
            centers = centers[keep_mask]
            print(f"      scales[mask]...", flush=True)
            scales = scales[keep_mask]
            print(f"      quats[mask]...", flush=True)
            quats = quats[keep_mask]
            print(f"      opacities[mask]...", flush=True)
            opacities = opacities[keep_mask]
            print(f"      opacities done, shape={opacities.shape}", flush=True)
            try:
                print(f"      albedo[mask]... shape={albedo.shape}, dtype={albedo.dtype}", flush=True)
                albedo = albedo[keep_mask]
                print(f"      albedo done", flush=True)
            except Exception as e:
                print(f"      ALBEDO ERROR: {type(e).__name__}: {e}", flush=True)
                raise
            print(f"      omega[mask]...", flush=True)
            omega = omega[keep_mask]
            print(f"      All masks applied", flush=True)
            n_prims_after = centers.shape[0]
            level_type = "Gabors" if l > 0 else "Gaussians"
            green(f"    Level {l} ({level_type}): {n_prims_before} -> {n_prims_after} primitives (pruned)")
        else:
            n_prims_after = n_prims_before
            level_type = "Gabors" if l > 0 else "Gaussians"
            green(f"    Level {l} ({level_type}): {n_prims_after} primitives (unchanged)")
        
        # Debug: print shapes before creating scene dict
        blue(f"      Shapes: centers={centers.shape}, scales={scales.shape}, quats={quats.shape}")
        blue(f"              opacities={opacities.shape}, albedo={albedo.shape}, omega={omega.shape}")
        
        # Create scene dict for this level
        try:
            level_dict = gu.create_scene_dict_from_primitives(
                max_depth=-1,
                level=l,
                centers=mi.TensorXf(centers),
                scales=mi.TensorXf(scales),
                quaternions=mi.TensorXf(quats),
                opacities=mi.TensorXf(opacities),
                albedo=mi.TensorXf(albedo),
                omega=mi.TensorXf(omega),
            )
            pruned_model[f'primitives_pyr{l}'] = level_dict[f'primitives_pyr{l}']
            blue(f"      Successfully created primitives_pyr{l}")
        except Exception as e:
            yellow(f"      ERROR creating primitives_pyr{l}: {e}")
            raise

    # Verify we have primitives in the model
    prim_keys = [k for k in pruned_model.keys() if k.startswith('primitives_pyr')]
    blue(f"  Model dict contains: {prim_keys}")
    
    # Load as scene and save using gu.save_model
    try:
        pruned_scene = mi.load_dict(pruned_model)
        blue(f"  Successfully loaded pruned scene")
    except Exception as e:
        yellow(f"  ERROR loading pruned scene: {e}")
        raise
    
    try:
        gu.save_model(output_path, pruned_scene, max_level)
        blue(f"  Called gu.save_model with max_level={max_level}")
    except Exception as e:
        yellow(f"  ERROR in gu.save_model: {e}")
        raise
    
    # Verify files were created
    npy_path = output_path / f'optimized_asset_pyr{max_level}' / 'npy_data'
    if npy_path.exists():
        saved_files = list(npy_path.glob('*.npy'))
        green(f"  Saved {len(saved_files)} files to {npy_path}")
        for f in sorted(saved_files)[:10]:  # Show first 10 files
            green(f"    - {f.name}")
    else:
        yellow(f"  WARNING: Expected output directory does not exist: {npy_path}")
    
    green(f"  Saved pruned assets to {output_path / f'optimized_asset_pyr{max_level}'}")


def copy_supporting_files(model_path: Path, output_path: Path) -> None:
    """
    Copies supporting files (args.json, grid files, reference images) from the 
    original model to the output directory. These are needed by other scripts
    like lod_partition.py.
    """
    # Copy args.json from original model to output root
    src_args = model_path / 'args.json'
    if src_args.exists():
        shutil.copy(src_args, output_path / 'args.json')
        green(f"  Copied args.json to {output_path}")
    else:
        yellow(f"  Warning: args.json not found in {model_path}")

    # Copy grid files (needed by lod_partition.py and other scripts)
    for grid_file in model_path.glob('grid_pyr*.npy'):
        shutil.copy(grid_file, output_path / grid_file.name)
        green(f"  Copied {grid_file.name} to {output_path}")

    # Copy reference images if they exist
    for ref_file in model_path.glob('reference_pyr*.exr'):
        shutil.copy(ref_file, output_path / ref_file.name)
        green(f"  Copied {ref_file.name} to {output_path}")


def save_pruning_stats(
        output_path: Path,
        stats: Dict[str, Any]
) -> None:
    """
    Saves pruning statistics to a JSON file.

    :param output_path: Directory to save the stats file.
    :param stats: Dictionary containing pruning statistics.
    """
    stats_file = output_path / 'pruning_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    green(f"  Saved pruning statistics to {stats_file}")


def prune(
        model_path: Path, 
        pruning_rule: Literal['density', 'opacity'], 
        PSNR_budget: float, 
        PSNR_target: float = 0.0, 
        spp: int = 8,
        output_dir: Path = None,
        save_assets: bool = False,
        save_stats: bool = True,
        prune_level: str = 'auto'
) -> Dict[str, Any]:
    """
    Prunes the Gabor mixture model based on the specified pruning rule and PSNR constraints.
    
    :param model_path: Path to the saved Gabor mixture model.
    :param pruning_rule: The pruning rule to use ('density' or 'opacity').
    :param PSNR_budget: The PSNR budget for pruning.
    :param PSNR_target: The PSNR target for pruning (if applicable).
    :param spp: Samples per pixel for rendering during pruning evaluation.
    :param output_dir: Output directory for results. Defaults to model_path/pruned_{rule}/.
    :param save_assets: Whether to save the pruned model assets.
    :param save_stats: Whether to save pruning statistics as JSON.
    :param prune_level: Which level to prune: 'auto' (highest), 'gauss'/'0', or 'gabor'/'1'.
    :return: Dictionary containing pruning statistics.
    """
    has_gabor, scene, scene_params, adam, cams, ref_images = prepare_mixture(model_path)
    max_level = int(has_gabor)

    # Determine which level to prune
    if prune_level in ['auto']:
        level = max_level  # highest available level
    elif prune_level in ['gauss', '0']:
        level = 0
    elif prune_level in ['gabor', '1']:
        if not has_gabor:
            raise ValueError("Cannot prune Gabors: model does not contain Gabor primitives (only Gaussians).")
        level = 1
    else:
        raise ValueError(f"Unknown prune_level: {prune_level}")
    
    level_name = "Gabors" if level == 1 else "Gaussians"
    
    # Get primitive counts (handle both 1D and 2D array shapes)
    scales_0 = adam['scales_0'].numpy()
    n_gaussians = scales_0.shape[-1] if scales_0.ndim > 1 else scales_0.shape[0] // 3
    n_gabors = 0
    if has_gabor:
        scales_1 = adam['scales_1'].numpy()
        n_gabors = scales_1.shape[-1] if scales_1.ndim > 1 else scales_1.shape[0] // 3

    if has_gabor:
        blue(f"Loaded Gauss-Gabor mixture from {model_path} with {n_gaussians} Gaussians and {n_gabors} Gabors.")
    else:
        blue(f"Loaded Gaussian model from {model_path} with {n_gaussians} Gaussians.")
    
    yellow(f"Pruning level {level} ({level_name})")

    # Setup output directory
    if output_dir is None:
        output_dir = model_path / f'pruned_{pruning_rule}'
    output_dir.mkdir(parents=True, exist_ok=True)
    yellow(f"Output directory: {output_dir}")

    orig_opacity = dr.copy(adam[f'opacities_{level}'])

    weights, weight_np, ordering = reorder_based_on_rule(pruning_rule, adam, level=level)
    sorted_weights = weight_np[ordering]  # sorted weights (already flattened)
    n_total = ordering.shape[0]  # number of primitives

    compute = partial(render,
        level=level,
        weight=weights,
        original_opacity=orig_opacity,
        adam=adam,
        scene=scene,
        scene_params=scene_params,
        cams=cams,
        spp=spp,
        ref_images=ref_images
    )

    PSNR_0, removed, img, _ = compute(0.0)
    assert removed == 0, "Initial rendering should not remove any primitives."

    PSNR_min = (PSNR_0 - PSNR_budget) if PSNR_target <= 0.0 else PSNR_target
    green(f"Initial PSNR: {PSNR_0:.2f} dB. Target PSNR after pruning: {PSNR_min:.2f} dB.")

    mi.util.write_bitmap(str(output_dir / f'original_pyr{level}_{spp}spp.exr'), img)

    low, high = 0, ordering.shape[0]  # low = 0 means "drop 0", high = N means "drop all"
    best_k = 0
    best_threshold = 0.0
    best_img = img
    best_psnr = PSNR_0
    best_keep_mask = np.ones(n_total, dtype=bool)

    iters = int(np.log2(ordering.shape[0]))
    iter_count = 1
    blue(f"Starting binary search to find optimal pruning threshold over {iters} iterations...")
    
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            threshold = 0.0
        else:
            threshold = sorted_weights[mid - 1]  # drop all with weight <= s[mid-1]

        curr_psnr, removed, img, keep_mask = compute(threshold)

        effect_dim(f"({iter_count:<2d}/{iters}) Trying {pruning_rule} threshold {threshold:.6f}: PSNR = {curr_psnr:.2f} dB (removed {removed} primitives).")
        iter_count += 1

        if curr_psnr >= PSNR_min:
            # Still acceptable quality → we can drop at least `mid` primitives
            best_k = mid
            best_threshold = threshold
            best_img = img
            best_psnr = curr_psnr
            best_keep_mask = keep_mask
            low = mid + 1    # try dropping more
        else:
            # PSNR too low → we dropped too many primitives
            high = mid - 1

    # Final results
    n_remaining = n_total - best_k
    compression_ratio = n_total / n_remaining if n_remaining > 0 else float('inf')
    
    green(f"Pruning complete. Dropping {best_k} ({best_k / n_total:.2%}) primitives with threshold {best_threshold:.6f}.")
    green(f"Final: {n_remaining} primitives remaining (compression ratio: {compression_ratio:.2f}x)")

    # Save final pruned render
    pruned_render_path = output_dir / f'pruned_pyr{level}_{spp}spp.exr'
    mi.util.write_bitmap(str(pruned_render_path), best_img)
    green(f"  Saved pruned render to {pruned_render_path}")

    # Compile statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'output_dir': str(output_dir),
        'pruning_rule': pruning_rule,
        'psnr_budget': PSNR_budget,
        'psnr_target': PSNR_target,
        'spp': spp,
        'pyramid_level_pruned': level,
        'level_name': level_name,
        'has_gabor': has_gabor,
        'metrics': {
            'initial_psnr_dB': float(PSNR_0),
            'final_psnr_dB': float(best_psnr),
            'psnr_drop_dB': float(PSNR_0 - best_psnr),
            'target_psnr_dB': float(PSNR_min),
        },
        'pruning_results': {
            'total_primitives': int(n_total),
            'primitives_removed': int(best_k),
            'primitives_remaining': int(n_remaining),
            'removal_percentage': float(best_k / n_total * 100),
            'compression_ratio': float(compression_ratio),
            'pruning_threshold': float(best_threshold),
        },
        'model_info': {
            'n_gaussians': int(n_gaussians),
            'n_gabors': int(n_gabors),
        },
        'output_files': {
            'original_render': f'original_pyr{level}_{spp}spp.exr',
            'pruned_render': f'pruned_pyr{level}_{spp}spp.exr',
        }
    }

    # Copy supporting files (args.json, grid files, reference images) - always needed
    copy_supporting_files(model_path, output_dir)

    # Save statistics
    if save_stats:
        save_pruning_stats(output_dir, stats)

    # Save pruned assets
    if save_assets:
        # Re-apply the best mask to the optimizer
        adam[f'opacities_{level}'] = dr.select(mi.TensorXf(~best_keep_mask), 0.0, orig_opacity)
        gu.write_optimizer_params_to_scene(adam, scene_params, level=level)
        
        save_pruned_assets(output_dir, model_path, adam, best_keep_mask, level, max_level)
        stats['output_files']['pruned_assets'] = f'optimized_asset_pyr{max_level}/npy_data/'
        stats['output_files']['args_json'] = 'args.json'

    return stats



if __name__ == "__main__":
    args = parse_args()
    stats = prune(
        model_path=args.model_path,
        pruning_rule=args.pruning_rule,
        PSNR_budget=args.psnr_budget,
        PSNR_target=args.psnr_target,
        spp=args.spp,
        output_dir=args.output_dir,
        save_assets=args.save_assets,
        save_stats=not args.no_stats,
        prune_level=args.level
    )
    
    # Print summary
    print("\n" + "=" * 60)
    green("PRUNING SUMMARY")
    print("=" * 60)
    print(f"  Level pruned:       {stats['pyramid_level_pruned']} ({stats['level_name']})")
    print(f"  Initial PSNR:       {stats['metrics']['initial_psnr_dB']:.2f} dB")
    print(f"  Final PSNR:         {stats['metrics']['final_psnr_dB']:.2f} dB")
    print(f"  PSNR drop:          {stats['metrics']['psnr_drop_dB']:.2f} dB")
    print(f"  Primitives removed: {stats['pruning_results']['primitives_removed']} / {stats['pruning_results']['total_primitives']} ({stats['pruning_results']['removal_percentage']:.1f}%)")
    print(f"  Compression ratio:  {stats['pruning_results']['compression_ratio']:.2f}x")
    print(f"  Output directory:   {stats['output_dir']}")
    print("=" * 60)
