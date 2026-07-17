# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script implements a simple optimization pipeline that reconstructs an
absorbing volume using a gabor gauss mixture (no scattering!).
Example usage:
    See gabor_regression_test.ps1
"""

from typing import Any, Dict, Final, List, Optional, Tuple, Union
try:
    import wandb
except ImportError:  # optional training/analysis dependency
    wandb = None
import time
import random
import shutil
from pathlib import Path

import numpy as np

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
T = mi.ScalarTransform4f
import gfields
from gfields.integrators.common import *
from gfields.training.regression import *  # noqa: F401,F403
from gfields.training.callbacks import (
    CallbackABC,
    Callbacks,
    GradientTrackerCallback,
    MetricsAccumulatorCallback,
    PlotCallback,
)
from gfields.training.configs import get_volume_config
from tqdm import tqdm
from colorist import blue, yellow, green, effect_dim, Effect

GAUSS_LVL: Final[int] = 0
GABOR_LVL: Final[int] = 1


def _integrator_scene_kwargs() -> Dict[str, Any]:
    return {
        'integrator_type': args.integrator,
        'use_kernel_freeze': args.use_kernel_freeze,
        'use_anyhit': getattr(args, 'use_anyhit', False),
    }


def _print_memory_budget() -> None:
    budget = estimate_memory_budget(
        cam_res_x=args.cam_res_x,
        cam_res_y=args.cam_res_y,
        cam_count=args.cam_count,
        ref_spp=args.ref_spp,
        ref_spp_chunk=getattr(args, 'ref_spp_chunk', 1024),
        opt_cam_batch=getattr(args, 'opt_cam_batch', 0),
        gabor_count=getattr(args, 'gabor_count', 0),
        gauss_count=getattr(args, 'gauss_count', 0),
        integrator=args.integrator,
    )
    effect_dim('Estimated GPU memory budget (MiB):')
    for key, val in budget.items():
        effect_dim(f'  {key}: {val:.2f}' if isinstance(val, float) and key.endswith('_mib') else f'  {key}: {val}')


def _batched_render_loss(
    scene: mi.Scene,
    scene_params: mi.SceneParameters,
    ref_image: mi.TensorXf,
    cam_batches: List[Tuple[mi.Sensor, slice]],
    *,
    spp: int,
    spp_grad: int,
    seed: int,
    l1_weight: float,
    ssim_weight: float,
) -> Tuple[mi.TensorXf, mi.Float, mi.Float, mi.Float, mi.Float]:
    '''Render camera batches and average L1/SSIM loss (reduces peak ray count).

    Returns ``(last_batch_image, loss, l1, ssim, mse)`` where ``loss``, ``l1``,
    ``ssim`` and ``mse`` are full-image averages over every camera batch.  The
    MSE is returned so a full-image PSNR can be logged even when
    ``opt_cam_batch`` renders only a subset of cameras per pass (the returned
    image is just the last batch).
    '''
    n_batches = len(cam_batches)
    loss: mi.Float | None = None
    l1_acc: mi.Float | None = None
    ssim_acc: mi.Float | None = None
    mse_acc: mi.Float | None = None
    last_image: mi.TensorXf | None = None
    for sensor, sl in cam_batches:
        image = mi.render(
            scene, scene_params, sensor=sensor,
            spp=spp, spp_grad=spp_grad, seed=seed,
        )
        batch_ref = slice_batched_ref_image(ref_image, args.cam_res_x, sl)
        n_cams = (sl.stop or args.cam_count) - (sl.start or 0)
        batch_l1 = gfields.losses.l1(batch_ref, image)
        batch_ssim = ssim_loss_fn(batch_ref, image, n_cams)
        batch_mse = dr.mean(dr.square(batch_ref - image), axis=None)
        batch_loss = batch_l1 * l1_weight + batch_ssim * ssim_weight
        loss = batch_loss if loss is None else (loss + batch_loss)
        l1_acc = batch_l1 if l1_acc is None else (l1_acc + batch_l1)
        ssim_acc = batch_ssim if ssim_acc is None else (ssim_acc + batch_ssim)
        mse_acc = batch_mse if mse_acc is None else (mse_acc + batch_mse)
        last_image = image
    assert (loss is not None and last_image is not None and l1_acc is not None
            and ssim_acc is not None and mse_acc is not None)
    inv = 1.0 / float(n_batches)
    return last_image, loss * inv, l1_acc * inv, ssim_acc * inv, mse_acc * inv


# Optimization Setup
blue(f'{Effect.BOLD}Setting Up Gabor Volume Regression{Effect.BOLD_OFF}')
args = setup()
volume_config = get_volume_config(args.volume_grid, opacity_scale=getattr(args, 'opacity_scale', None))
print(f'Using volume configuration: {volume_config.name} (opacity_scale={volume_config.opacity_scale})')

cameras = prepare_cameras(args.cam_count, args.cam_res_x, args.cam_res_y, cam_scale = args.cam_scale, cam_y_offset=args.cam_y_offset, sample_single_hemisphere=args.sample_single_hemisphere)
camera_array = create_cam_array(cameras)  # the render all cameras into a single image
opt_cam_batches = iter_cam_batches(cameras, getattr(args, 'opt_cam_batch', 0))
_print_memory_budget()

# Reference images are cached in a SHARED content-addressed store keyed by their
# generating parameters (volume + cameras + spp + ...), so render_reference_images
# below loads them from disk when a matching set exists and only renders once
# otherwise -- independent of --output, so re-optimizing an asset never re-renders.
gauss_ref_volume, ref_volume = prepare_ref_volume(args.volume_grid, args.output, args.gauss_target_level, args.gabor_target_level, regen_pyramid=args.regen_pyramid, pad_grid=args.pad_grid)


ssim_loss_fn = gfields.losses.SSIMLoss(window_size=11, sigma=1.5)

effect_dim(f'Volume max: {np.max(ref_volume)}')
effect_dim(f'Volume min: {np.min(ref_volume)}')
effect_dim(f'Volume sum: {np.sum(ref_volume)}')
effect_dim(f'Volume sum: {np.sum(ref_volume) * volume_config.opacity_scale * ( 8.0 / np.max(ref_volume.shape)**3)}')

# Gaussian Optimization ----------------------------------------------------

gaussian_prims_ckpt: Optional[Dict[str, Union[np.ndarray, mi.TensorXf]]] = None
if args.from_gaussian_checkpoint:
    ckpt_path = Path(args.from_gaussian_checkpoint)
    if ckpt_path.exists():
        gaussian_prims_ckpt = load_single_level(ckpt_path, level=0)
        green(f'Loaded Gaussian checkpoint from {ckpt_path} with {len(gaussian_prims_ckpt["centers"])} kernels.')
        if not (args.output / f'optimized_asset_pyr{GAUSS_LVL}' / 'npy_data').exists():
            effect_dim("- Saving checkpoint also to this directory for completeness.")
            scene_dict_ckpt = create_scene_dict_from_primitives(
                args.max_depth, GAUSS_LVL, **_integrator_scene_kwargs(), **gaussian_prims_ckpt)
            scene_ckpt: mi.Scene = mi.load_dict(scene_dict_ckpt)
            save_model(args.output, scene_ckpt, level=GAUSS_LVL)
            del scene_dict_ckpt, scene_ckpt
    else:
        yellow(f'Gaussian checkpoint not found at {ckpt_path}; optimizing Gaussian level from scratch.')

if not args.skip_gaussian_optim:
    blue(f'\n{Effect.BOLD}Starting Gaussian Level Regression{Effect.BOLD_OFF}')  # for better readability in the console
    # Scene Creation

    # initialize volumetric primitive scene ----------------------------------------
    # use gaussian checkpoint if exists else generate random kernels
    gaussian_prims = gaussian_prims_ckpt or generate_random_kernels(
        args.gauss_count, args.init_opacity, args.init_omega, args.init_scale, args.init_albedo,
        level=GAUSS_LVL, target_level=args.gauss_target_level, importance_sample_means=args.importance_sample_means, volume=ref_volume, volume_config=volume_config, opacity_sampling_grid=gauss_ref_volume
        )
    scene_dict = create_scene_dict_from_primitives(
        args.max_depth, GAUSS_LVL, **_integrator_scene_kwargs(), **gaussian_prims)

    for i, cam in enumerate(cameras):
        scene_dict[f'cam_{i:04d}'] = cam

    scene: mi.Scene = mi.load_dict(scene_dict)
    scene_params: mi.SceneParameters = mi.traverse(scene)


    # Create reference images (shared content-addressed cache: loaded from disk
    # if a matching set exists, rendered once otherwise).
    print()  # for better readability in the console
    ref_images = render_reference_images(
        gauss_ref_volume, volume_config, cameras, args.output,
        level=GAUSS_LVL, spp=args.ref_spp, ref_spp_chunk=args.ref_spp_chunk,
        init_albedo=args.init_albedo, regen=args.regen_pyramid,
        # Cache validation parameters
        volume_grid_path=args.volume_grid, cam_res_x=args.cam_res_x, cam_res_y=args.cam_res_y,
        cam_scale=args.cam_scale, sample_single_hemisphere=args.sample_single_hemisphere,
        target_level=args.gauss_target_level,
    )
    mi.util.write_bitmap(str(args.output / 'reference_pyr0.exr'), gfields.utils.concatenate_tensors(ref_images))
    ref_image = gfields.utils.concatenate_tensors(ref_images)


    # render initial scene
    init_image = mi.render(scene, sensor=camera_array, spp=args.ref_kernel_spp)

    effect_dim(f"- Saving initial images...")
    mi.util.write_bitmap(str(args.output / 'initial_pyr0.exr'), init_image)

    del init_image


    # Setup Optimizer

    opt = gfields.optimizers.BoundedAdam()

    write_scene_params_to_optimizer(scene_params, opt, level=GAUSS_LVL)

    lr_dict = create_lr_dict(args, args.global_lr_gaussian, level=GAUSS_LVL)
    opt.set_learning_rate(lr_dict)

    opt.set_bounds('scales_0', lower=1e-08, upper=1.0) # if done correctly, this should cover roughly the Gaussian low pass region
    opt.set_bounds('omega_0', lower=0.0, upper=0.0) # base is a TRUE Gaussian: omega==0. The mixture kernel oscillates for ANY nonzero omega (that is the sole thing marking a primitive as Gabor), so a free base omega would render the base as Gabors.
    opt.set_bounds('opacities_0', lower=1e-8, upper=1.0) # 1.0 for gabor, 1e-2 for gaussian
    opt.set_bounds('albedo_0', lower=1e-9, upper=1.0)
    opt.set_bounds('centers_0', lower=-1.0, upper=1.0)

    write_optimizer_params_to_scene(opt, scene_params, level=GAUSS_LVL)

    # Optimization Loop

    # precompute the reference image portion for the SSIM loss function
    ssim_loss_fn.precompute_reference_convs(ref_image, args.cam_count)

    fine_tune_period = args.iterations_gaussian - args.densify_until
    
    lr_scheduler = LRScheduler(
        opt, args.global_lr_gaussian, args.iterations_gaussian,
        base_lr_dict=lr_dict, warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio
    )

    callbacks = Callbacks(args, GAUSS_LVL)
    callbacks.register_callbacks([
        GradientTrackerCallback(),
        PlotCallback(),
        MetricsAccumulatorCallback()
    ])
    callbacks.setup()


    green(f'\nRun optimization (level 0: Gaussians Only) {Effect.DIM}[Send KeyboardInterrupt to save the model and skip this level]{Effect.DIM_OFF}')

    # Initialize benchmark timer
    timer = gfields.benchmark.OptimizationTimer(log_every=9999, log_level=1)
    try:
        pbar = tqdm(range(args.epoch_start, args.epoch_start + args.iterations_gaussian), desc='Gaussian Optimization', unit='step')
        for it in range(args.iterations_gaussian):
            timer.start_iteration()

            # reset gradients
            dr.clear_grad(opt.variables)

            if it + 1 > args.densify_until:
                lr_scheduler.step()

            if it + 1 >= args.iterations_gaussian * 0.9:
                args.opt_spp = 2

            with timer.track('forward'):
                image, img_loss, l1, ssim, img_mse = _batched_render_loss(
                    scene, scene_params, ref_image, opt_cam_batches,
                    spp=args.opt_spp, spp_grad=args.grad_spp, seed=it + args.seed_offset,
                    l1_weight=0.75, ssim_weight=0.25,
                )

            with timer.track('loss'):
                loss = img_loss

                regularization = frequency_regularization(opt, ref_volume, volume_config, GAUSS_LVL, weight=args.lambda_frequency)

                opacity_penalty = dr.sum(dr.abs(opt[f'opacities_{GAUSS_LVL}'])) * args.lambda_opacity
                scale_penalty = dr.sum(dr.abs(opt[f'scales_{GAUSS_LVL}'])) * args.lambda_scale

                loss = loss + regularization + opacity_penalty + scale_penalty

            with timer.track('backward'):
                dr.backward(loss)


            callbacks.on_before_opt_step(it, opt)
            with timer.track('opt_step'):
                opt.step()
                write_optimizer_params_to_scene(opt, scene_params, level=GAUSS_LVL)

            # Metrics force a device->host sync; only pay for them on the
            # configured cadence (plus the image-inspection iterations).
            inspect_iters = (2, 4, 6, 8, 10, 15, 20, 25, 50, 59)
            do_log = ((it + 1) % args.log_every == 0
                      or (it + 1) == args.iterations_gaussian
                      or (it + 1) in inspect_iters)
            relocate_now = (it + 1) % 30 == 0 and (it + 1) <= args.relocate_until
            deletable = 0
            if do_log or relocate_now:
                deletable = (dr.abs(opt[f'opacities_{GAUSS_LVL}']).numpy() < 1e-6).sum()
            if relocate_now:
                print(f"Relocating dead primitives at it={it}...")
                relocate_dead_primitives(opt, scene_params, GAUSS_LVL)
                deletable2 = (dr.abs(opt[f'opacities_{GAUSS_LVL}']).numpy() < 1e-6).sum()
                print(f"Revived {deletable - deletable2} primitives.")

            timer.end_iteration(it)



            with dr.suspend_grad():
                img_write_modulo = (it + 1) % args.write_image_every
                if do_log:
                    # Full-image PSNR from batch-averaged MSE (image is only the
                    # last camera batch when opt_cam_batch>0).
                    psnr = -10.0 / dr.log(mi.Float(10.0)) * dr.log(dr.maximum(img_mse, 1e-12))
                    metrics: Dict[str, float] = {
                        "loss": loss.array[0], "L1": l1.array[0],
                        "SSIM": ssim.array[0], "PSNR": psnr.array[0],
                        "Regu": regularization.array[0], "OpRegu": opacity_penalty.array[0],
                        "ScaleRegu": scale_penalty.array[0], "deletable": float(deletable),
                    }
                    callbacks.on_suspend_grad(it, opt, metrics)

                if img_write_modulo == 0 or (it + 1) in inspect_iters:
                    mi.util.write_bitmap(str(args.output / f'frames_pyr{GAUSS_LVL}' / f'image_{it+1:04d}.exr'), image)

                # Frequent checkpoint (resumable via --from_gaussian_checkpoint).
                if (it + 1) % args.checkpoint_every == 0:
                    save_model(args.output, scene, level=GAUSS_LVL)

            pbar.update(1)
            if do_log:
                pbar.set_postfix(next_imgs_in=args.write_image_every - img_write_modulo, **metrics)
        pbar.close()
        green('\nDone with optimization, saving current state...')
    except KeyboardInterrupt:
        pbar.close()
        yellow('\nOptimization interrupted, saving current state...')

    # Print benchmark summary
    timer.summary()

    # Final degenerate cleanup (see the Gabor-phase counterpart): recycle
    # loops handle collapse during optimization; any scale-collapsed
    # survivor is dropped here so it never reaches the saved asset.
    n_collapsed = prune(opt, scene_params, GAUSS_LVL, rule='opacity', custom_th=0.0)
    if n_collapsed > 0:
        yellow(f'Removed {n_collapsed} scale-collapsed primitives from level {GAUSS_LVL}.')

    # Save Results

    optimized = mi.render(scene, sensor=camera_array, spp=args.ref_kernel_spp)
    green("- Saving optimized images...")
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mi.util.write_bitmap(str(args.output / f'optimized_pyr0.exr'), optimized)

    green("- Saving optimized model...")
    save_model(args.output, scene, level=GAUSS_LVL)

    print(f'\nFinal PSNR: {gfields.losses.psnr(ref_image, dr.clip(optimized, 0.0, 1.0)).array[0]}')
    print(f'Final SSIM: {ssim_loss_fn(ref_image, dr.clip(optimized, 0.0, 1.0), args.cam_count).array[0]}')
    print(f'Final L1: {gfields.losses.l1(ref_image, dr.clip(optimized, 0.0, 1.0)).array[0]}')

    callbacks.on_optim_end()


if args.skip_gabor_optim:
    blue(f'\n{Effect.BOLD}Skipping Gabor Level Regression{Effect.BOLD_OFF}')
    import sys; sys.exit(0)

# Gabor Optimization ------------------------------------------------------
blue(f'\n{Effect.BOLD}Starting Gabor Level Regression{Effect.BOLD_OFF}')  # for better readability in the console

# Scene Creation

# init gabor primitives from the low res and full res difference volume
if args.from_gabor_checkpoint:
    # load the checkpoint from the previous Gabor optimization
    gaussian_prims_ckpt = load_single_level(Path(args.from_gabor_checkpoint), level=0)
    gabor_prims_ckpt = load_single_level(Path(args.from_gabor_checkpoint), level=1)
    green(f'Loaded Gabor checkpoint from {args.from_gabor_checkpoint} with {len(gabor_prims_ckpt["centers"])} kernels.')
else:
    gabor_prims_ckpt = generate_random_kernels(
        args.gabor_count, args.init_opacity, args.init_omega, args.init_scale, args.init_albedo,
        level=GABOR_LVL, target_level=args.gabor_target_level, importance_sample_means=args.importance_sample_means,
        volume=ref_volume, volume_config=volume_config, opacity_sampling_grid=ref_volume - gauss_ref_volume
    )
scene_dict = create_scene_dict_from_primitives(
    args.max_depth, GABOR_LVL, **_integrator_scene_kwargs(), **gabor_prims_ckpt)

# load gaussian level and add it to the scene dict
use_pretrained_gauss_ckpt = args.from_gaussian_checkpoint and args.skip_gaussian_optim
scene_dict[f'primitives_pyr{GAUSS_LVL}'] = (
    create_scene_dict_from_primitives(args.max_depth, GAUSS_LVL, **gaussian_prims_ckpt)[f'primitives_pyr{GAUSS_LVL}']  # load the Gaussian primitives from the checkpoint
    if use_pretrained_gauss_ckpt else
    load_model(args.output / f'optimized_asset_pyr{GAUSS_LVL}', args.max_depth)[f'primitives_pyr{GAUSS_LVL}']  # load previous optimization
)

for i, camera in enumerate(cameras):
    scene_dict[f'cam_{i:04d}'] = camera  # to store the cameras for serialization

cam_array = create_cam_array(cameras)
gabor_opt_batches = iter_cam_batches(cameras, getattr(args, 'opt_cam_batch', 0))
# Stochastic camera subsampling: precompute single-camera batches so each gabor
# iteration can render a random K-camera subset (SGD over views) instead of all
# cameras. Each entry is a contiguous 1-camera (sensor, slice) reusing the same
# ref-slicing path; the loss is averaged over the sampled subset.
_cam_subsample = int(getattr(args, 'cam_subsample', 0) or 0)
per_cam_batches = iter_cam_batches(cameras, 1) if _cam_subsample > 0 else None
_cam_rng = random.Random(1234 + args.seed_offset)

scene: mi.Scene = mi.load_dict(scene_dict)
scene_params: mi.SceneParameters = mi.traverse(scene)


# Reference Image Creation
print()  # for better readability in the console
# Reminder: output folders are created in the setup function
assert (args.output / f'frames_pyr{GABOR_LVL}').exists(), f'Output folder {str(args.output / f"frames_pyr{GABOR_LVL}")} does not exist but should be created in setup().'

ref_images = render_reference_images(
    ref_volume, volume_config, cameras, args.output,
    level=GABOR_LVL, spp=args.ref_spp, ref_spp_chunk=args.ref_spp_chunk,
    init_albedo=args.init_albedo, regen=args.regen_pyramid,
    # Cache validation parameters
    volume_grid_path=args.volume_grid, cam_res_x=args.cam_res_x, cam_res_y=args.cam_res_y,
    cam_scale=args.cam_scale, sample_single_hemisphere=args.sample_single_hemisphere,
    target_level=args.gabor_target_level,
)
mi.util.write_bitmap(str(args.output / f'reference_pyr{GABOR_LVL}.exr'), dr.concat(ref_images, axis=1))

ref_image = gfields.utils.concatenate_tensors(ref_images)
del ref_images  # free memory


init_image = mi.render(scene, sensor=cam_array, spp=args.ref_kernel_spp)
mi.util.write_bitmap(str(args.output / f'initial_pyr{GABOR_LVL}.exr'), init_image)
effect_dim(f"- Saving initial images...")

del init_image



# Setup Optimizer

bounds_dict = {}
bounds_dict[f'min_scale_{GAUSS_LVL}'] = 1e-6
bounds_dict[f'max_scale_{GAUSS_LVL}'] = 1.0
bounds_dict[f'min_frequency_{GAUSS_LVL}'] = 0.0
bounds_dict[f'max_frequency_{GAUSS_LVL}'] = 1000.0
bounds_dict[f'min_opacity_{GAUSS_LVL}'] = 0.0
bounds_dict[f'max_opacity_{GAUSS_LVL}'] = 1.0

bounds_dict[f'min_scale_{GABOR_LVL}'] = gaussian_nyquist_spatial_scale(-1, 1, ref_volume.shape, eps=0.25)
print(f"Gabor min scale bound: {bounds_dict[f'min_scale_{GABOR_LVL}']}")

# Count-aware max-scale cap. `--limit_max_scale --max_scale_mult 0.5` was tuned
# for the large-count regime (~32k kernels): tiny primitives tile the volume, so
# a tight cap is both faster and higher quality. With far fewer primitives each
# must cover proportionally more volume (size ~ N^{-1/3}), so a fixed cap starves
# the fit -- the `--limit_max_scale` help notes it "can harm if the number is
# low". We relax the cap for small counts by (N_ref/N)^{1/3}, clamped so the
# large-count behaviour is left exactly as tuned.
MAX_SCALE_REF_COUNT = 32768
if args.limit_max_scale:
    n_gabor = max(int(args.gabor_count), 1)
    count_relax = max(1.0, (MAX_SCALE_REF_COUNT / n_gabor) ** (1.0 / 3.0))
    bounds_dict[f'max_scale_{GABOR_LVL}'] = (
        gaussian_min_spatial_scale(3, -1, 1) * args.max_scale_mult * count_relax)
    if count_relax > 1.001:
        yellow(f"[small-count] gabor_count={n_gabor} < {MAX_SCALE_REF_COUNT}: "
               f"relaxing max-scale cap x{count_relax:.2f} (effective max_scale_mult "
               f"{args.max_scale_mult * count_relax:.3f}). For very sparse assets also "
               f"consider dropping --limit_max_scale. NOTE: the default opacity "
               f"schedule (init_opacity 8e-5, opacity_lr multiplier tuned for long "
               f"32k runs) leaves the Gaussian base UNDER-DENSE in short/sparse fits "
               f"-- raise --init_opacity (~5e-4) and --opacity_lr (~1e-3) for a dense base.")
else:
    bounds_dict[f'max_scale_{GABOR_LVL}'] = np.inf
bounds_dict[f'min_frequency_{GABOR_LVL}'] = 0.5
bounds_dict[f'max_frequency_{GABOR_LVL}'] = 3.0
bounds_dict[f'min_opacity_{GABOR_LVL}'] = -1.0
bounds_dict[f'max_opacity_{GABOR_LVL}'] = 1.0

opt = gfields.optimizers.BoundedAdam(noise_init=args.noise_init, noise_decay=args.noise_decay)
write_scene_params_to_optimizer(scene_params, opt, level=GABOR_LVL)
opt[f'opacities_{GAUSS_LVL}']  = scene_params[f'primitives_pyr{GAUSS_LVL}.opacities']

lr_dict = create_lr_dict(args, args.global_lr_gabor, GABOR_LVL)
opt.set_learning_rate(lr_dict)

# set learning rate for the gaussian level of gaussian opacity to allow slight adjustments
opt.set_learning_rate({f'opacities_{GAUSS_LVL}': args.global_lr_gabor * args.opacity_lr * 0.1})  # allow slight adjustments of the Gaussian level

set_opt_bounds(opt, bounds_dict, GABOR_LVL)
write_optimizer_params_to_scene(opt, scene_params, GABOR_LVL)
scene_params[f'primitives_pyr{GAUSS_LVL}.opacities'] = opt[f'opacities_{GAUSS_LVL}']

# Restore Adam moments (t, m, v) from the checkpoint so a resume continues
# without the cold-start dip (first step with v=0 kicks a converged solution off
# by ~10 dB). Only restores params whose count is unchanged; densify/prune/count
# changes safely cold-start those entries.
if args.from_gabor_checkpoint:
    n_restored = load_optimizer_state(opt, Path(args.from_gabor_checkpoint), GABOR_LVL)
    if n_restored > 0:
        green(f'Restored Adam optimizer state for {n_restored} parameter groups (dip-free resume).')
    else:
        yellow('No optimizer state found in checkpoint; resume will cold-start Adam (expect a transient PSNR dip).')


# Optimization Loop


callbacks = Callbacks(args, GABOR_LVL)
callbacks.register_callbacks([
    GradientTrackerCallback(),
    PlotCallback(),
    MetricsAccumulatorCallback()
])
callbacks.setup()

# Precompute the reference image convolutions for SSIM (ref_image is a list of batched tensors)
ssim_loss_fn.precompute_reference_convs(ref_image, args.cam_count)

densify_idx = 0

# Volume pyramid used by densify(): pyr[0] is the full-resolution gabor target
# and the coarser entries are lower-frequency references. densify() adds new
# primitives where |pyr[0] - pyr[band]| (the high-frequency residual) is large.
# ref_volume (gabor target) and gauss_ref_volume (gaussian/low-pass target) share
# the same grid shape, so their difference is the gabor-relevant detail. This was
# previously undefined, which crashed any run with densify_until > 0.
pyr = [ref_volume, gauss_ref_volume]

lr_scheduler = LRScheduler(
    opt, args.global_lr_gabor, args.iterations_gabor,# - args.relocate_until,
    base_lr_dict=lr_dict, warmup_ratio=args.warmup_ratio, min_lr_ratio=args.min_lr_ratio
)
green(f'\nRun optimization (level {GABOR_LVL}: Gabors) {Effect.DIM}[Send KeyboardInterrupt to save the model and skip this level]{Effect.DIM_OFF}')
gabor_timer = gfields.benchmark.OptimizationTimer(log_every=9999, log_level=1)

# Best-checkpoint selection: the MCMC exploration noise + relocation leave the
# FINAL state at a random point of a multi-dB oscillation (small/simple assets
# saturate early and then just jitter), so the last iteration is often not the
# best. Track the best CLEAN-render PSNR (ref_kernel_spp, not the noisy 2-spp
# loop metric) and stash that mixture; restore it at the end if it beat the
# final. Disable with --no_save_best.
best_gabor_psnr = -float('inf')
best_ckpt_dir = args.output / '_best_ckpt'

def _clean_gabor_psnr(seed: int) -> float:
    with dr.suspend_grad():
        img = mi.render(scene, sensor=camera_array, spp=args.ref_kernel_spp, seed=seed)
        return float(gfields.losses.psnr(ref_image, dr.clip(img, 0.0, 1.0)).array[0])

try:
    pbar = tqdm(range(args.iterations_gabor), desc='Gabor Optimization', unit='step')
    for it in range(args.iterations_gabor):
        gabor_timer.start_iteration()
        # reset gradients
        dr.clear_grad(opt.variables)
        lr_scheduler.step()
        
        if it + 1 >= args.iterations_gabor * 0.9:
            args.opt_spp = 2
        
        # Pick the camera batches to optimize this iteration. With
        # --cam_subsample K>0 we draw K random single-camera batches (stochastic
        # SGD over views) so per-iteration ray work scales with K, not cam_count;
        # otherwise use the full (memory-chunked) batch set.
        if per_cam_batches is not None:
            k = min(_cam_subsample, len(per_cam_batches))
            iter_batches = _cam_rng.sample(per_cam_batches, k)
        else:
            iter_batches = gabor_opt_batches
        with gabor_timer.track('forward'):
            image, img_loss, l1, ssim, img_mse = _batched_render_loss(
                scene, scene_params, ref_image, iter_batches,
                spp=args.opt_spp, spp_grad=args.grad_spp, seed=it + args.seed_offset,
                l1_weight=0.80, ssim_weight=0.20,
            )

        if dr.any(dr.isnan(image)):
            print("Rendering I*B has nans")

        with gabor_timer.track('loss'):
            loss = img_loss


        regularization = frequency_regularization(opt, ref_volume, volume_config, GABOR_LVL, weight=args.lambda_frequency)

        opacity_penalty = dr.sum(dr.abs(opt[f'opacities_{GABOR_LVL}'])) * args.lambda_opacity
        scale_penalty = dr.sum(dr.abs(opt[f'scales_{GABOR_LVL}'])) * args.lambda_scale

        loss = loss + regularization + opacity_penalty + scale_penalty
        with gabor_timer.track('backward'):
            dr.backward(loss)
        callbacks.on_before_opt_step(it, opt)
        with gabor_timer.track('opt_step'):
            opt.step()


        write_optimizer_params_to_scene(opt, scene_params, level=GABOR_LVL)
        scene_params[f'primitives_pyr{GAUSS_LVL}.opacities'] = opt[f'opacities_{GAUSS_LVL}']

        if (it + 1) >= args.start_pruning_from and (it + 1 - args.start_pruning_from) % args.prune_every == 0:
            t = time.time()
            num_pruned = prune(opt, scene_params, GABOR_LVL, rule='opacity', custom_th=1e-5)
            if num_pruned > 0:
                with dr.suspend_grad():
                    post_prune_render = mi.render(scene, scene_params, sensor=camera_array, spp=args.ref_kernel_spp, seed=it)
                    post_prune_psnr = gfields.losses.psnr(ref_image, post_prune_render).numpy()
                    # Pre-prune PSNR from the full-image MSE (image is only the
                    # last camera batch when opt_cam_batch>0).
                    psnr = float(-10.0 / dr.log(mi.Float(10.0)) * dr.log(dr.maximum(img_mse, 1e-12)))
            else:
                post_prune_psnr = psnr = 0
            print(f"Pruning took {(time.time() - t) * 1000:.2f} ms ; PSNR change: {post_prune_psnr - psnr:.2f}")


        # we need to keep gradients going while densifying the volume
        # otherwise backward pass will fail.
        if (it + 1) % args.densify_every == 0 and it + 1 < args.densify_until:
            t = time.time()
            band = max(-2 - densify_idx//2, -len(pyr) + 1)  # move closer to the full resolution level every densification step
            densify(opt, scene_params, GABOR_LVL, args, [pyr[0], pyr[band]], volume_config)
            densify_idx += 1
            print(f"Densification took {(time.time() - t) * 1000:.2f} ms, with band {band} and init_omega {args.init_omega:.3f}")

        # Metrics force a device->host sync; only pay for them on the
        # configured cadence (plus the image-inspection iterations).
        inspect_iters = {2, 4, 6, 8, 10, 15, 20}
        do_log = ((it + 1) % args.log_every == 0
                  or (it + 1) == args.iterations_gabor
                  or (it + 1) in inspect_iters)
        relocate_now = (it + 1) % 30 == 0 and (it + 1) <= args.relocate_until
        deletable = 0
        if do_log or relocate_now:
            deletable = (dr.abs(opt[f'opacities_{GABOR_LVL}']).numpy() < 1e-6).sum()
        if relocate_now:
            print(f"Relocating dead primitives at it={it}...")
            relocate_dead_primitives(opt, scene_params, GABOR_LVL)
            deletable2 = (dr.abs(opt[f'opacities_{GABOR_LVL}']).numpy() < 1e-6).sum()
            print(f"Revived {deletable - deletable2} primitives.")

        with dr.suspend_grad():
            img_write_modulo = (it + 1) % args.write_image_every
            if do_log:
                # Full-image PSNR from the batch-averaged MSE (the returned
                # ``image`` is only the last camera batch when opt_cam_batch>0).
                psnr = -10.0 / dr.log(mi.Float(10.0)) * dr.log(dr.maximum(img_mse, 1e-12))
                metrics: Dict[str, float] = {
                    "loss": loss.array[0], "L1": l1.array[0],
                    "SSIM": ssim.array[0], "PSNR": psnr.array[0],
                    "Regu": regularization.array[0], "OpRegu": opacity_penalty.array[0], "ScaleRegu": scale_penalty.array[0], "deletable": float(deletable),
                }
                callbacks.on_suspend_grad(it, opt, metrics)

            if img_write_modulo == 0 or it + 1 in inspect_iters:
                mi.util.write_bitmap(str(args.output / f'frames_pyr{GABOR_LVL}' / f'image_{it + 1:04d}.exr'), image)

            # Frequent checkpoint so a teardown/kill mid-optimization leaves a
            # usable + resumable asset (resume via --from_gabor_checkpoint).
            if (it + 1) % args.checkpoint_every == 0:
                save_model(args.output, scene, level=GABOR_LVL)
                save_optimizer_state(opt, args.output, GABOR_LVL)
                # Best-checkpoint: stash this mixture if its clean PSNR is a new
                # best (skips the first 1/3 of training, where quality is still
                # ramping and a stash would just be churn).
                if args.save_best and (it + 1) >= args.iterations_gabor // 3:
                    cp = _clean_gabor_psnr(seed=it)
                    if cp > best_gabor_psnr:
                        best_gabor_psnr = cp
                        save_model(best_ckpt_dir, scene, level=GABOR_LVL)

        pbar.update(1)
        if do_log:
            pbar.set_postfix(next_imgs_in=args.write_image_every - img_write_modulo, **metrics)

        gabor_timer.end_iteration(it)

    pbar.close()
    gabor_timer.summary()
    green('\nDone with optimization, saving current state...')
except KeyboardInterrupt:
    pbar.close()
    yellow(f'\nOptimization of level {GABOR_LVL} interrupted, saving current state...')


# Final degenerate cleanup: primitives whose scales collapsed onto the lower
# bound are invisible at render resolution (no data gradient, ~zero
# covariance-scaled MCMC noise) yet carry astronomical point densities that
# distort any faithful evaluation (e.g. the any-hit transmittance queries).
# custom_th=0 keeps every primitive with nonzero opacity, so only the
# scale-collapse criterion inside prune() acts.
#
# Only the Gabor level is cleaned here: this optimizer owns the Gabor-level
# parameters (plus opacities_0 for the light base-density coupling), but NOT
# scales_0/centers_0/... -- so prune(GAUSS_LVL) would KeyError on 'scales_0'.
# The Gaussian base was already scale-cleaned at the end of its own phase
# (its scales are frozen during the Gabor phase, so nothing new collapses).
n_collapsed = prune(opt, scene_params, GABOR_LVL, rule='opacity', custom_th=0.0)
if n_collapsed > 0:
    yellow(f'Removed {n_collapsed} scale-collapsed primitives from level {GABOR_LVL}.')

# Save Results
with dr.suspend_grad():
    optimized: mi.TensorXf = mi.render(scene, sensor=camera_array, spp=args.ref_kernel_spp)
green("- Saving optimized images...")
mi.util.write_bitmap(str(args.output / f'optimized_pyr{GABOR_LVL}.exr'), optimized)

green("- Saving optimized model...")
save_model(args.output, scene, level=GABOR_LVL)
save_optimizer_state(opt, args.output, GABOR_LVL)

final_gabor_psnr = float(gfields.losses.psnr(ref_image, dr.clip(optimized, 0.0, 1.0)).array[0])
# Best-checkpoint: if a mid-training mixture had a higher clean PSNR than the
# final iterate (the noise/relocation oscillation left the last step below the
# peak), swap the saved Gabor asset for that best one. The Gaussian base (pyr0)
# is untouched — only the Gabor level oscillates.
if args.save_best and best_gabor_psnr > final_gabor_psnr:
    src = best_ckpt_dir / f'optimized_asset_pyr{GABOR_LVL}'
    dst = args.output / f'optimized_asset_pyr{GABOR_LVL}'
    if src.exists():
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        green(f'- Best checkpoint ({best_gabor_psnr:.2f} dB) beat the final iterate '
              f'({final_gabor_psnr:.2f} dB); saved the best.')
        final_gabor_psnr = best_gabor_psnr
shutil.rmtree(best_ckpt_dir, ignore_errors=True)  # tidy the scratch stash

print(f'\nFinal PSNR: {final_gabor_psnr}')

callbacks.on_optim_end()


# ---------------------------------------------------------------------------
# Post-training asset-refinement passes (opt-in). Each is wrapped so a failure
# here never invalidates the already-saved optimized asset.
# ---------------------------------------------------------------------------
_asset_dir = args.output / f'optimized_asset_pyr{GABOR_LVL}'

if getattr(args, 'post_clamp', False):
    blue(f'\n{Effect.BOLD}Post-training: adaptive extent clamping{Effect.BOLD_OFF}')
    try:
        from gfields.training.adaptive import clamp_asset_level
        for _lvl in range(GABOR_LVL + 1):
            try:
                clamp_asset_level(_asset_dir, _lvl)
            except FileNotFoundError as _e:
                yellow(f'  [post_clamp] skipped level {_lvl}: {_e}')
    except Exception as _e:  # noqa: BLE001 - never lose the trained asset
        yellow(f'  [post_clamp] failed (trained asset is intact): {_e}')

if getattr(args, 'post_prune', False):
    blue(f'\n{Effect.BOLD}Post-training: PSNR-budget pruning '
         f'(budget {args.post_prune_budget:.2f} dB){Effect.BOLD_OFF}')
    try:
        from gfields.training.post_pruning import prune as _prune_asset
        _prune_asset(_asset_dir, 'opacity',
                     PSNR_budget=args.post_prune_budget, save_assets=True)
    except Exception as _e:  # noqa: BLE001 - never lose the trained asset
        yellow(f'  [post_prune] failed (trained asset is intact): {_e}')


