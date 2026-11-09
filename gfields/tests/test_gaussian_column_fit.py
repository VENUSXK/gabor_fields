"""Fit one Gaussian column to a known density field, without rendering or BVH.

Run from the root .venv: python -m gfields.tests.test_gaussian_column_fit
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import Ellipsoid, GaussianColumn, GaussianKernel


def EvaluateDensity(gaussian_columns, query_points, gaussian_kernel):
    centers, scales, quaternions, density_weights = gaussian_columns.Expand()
    density = mi.Float(0.0)
    for component_index in range(dr.width(density_weights)):
        parameter_index = mi.UInt32(component_index)
        quaternion = dr.gather(mi.Quaternion4f, quaternions, parameter_index)
        ellipsoid = Ellipsoid(center=dr.gather(mi.Point3f, centers, parameter_index), scale=dr.gather(mi.Vector3f, scales, parameter_index), quat=quaternion, rot=dr.quat_to_matrix(mi.Matrix3f, quaternion))
        density += dr.gather(mi.Float, density_weights, parameter_index) * gaussian_kernel.pdf(query_points, ellipsoid, mi.Bool(True))
    return density


def CreateOptimizer(parameter_values):
    column_optimizer = gfields.optimizers.BoundedAdam(lr=0.01, noise_init=0.0)
    for parameter_name, values in parameter_values.items():
        column_optimizer[parameter_name] = mi.Float(values)
    for parameter_name in ("center_x", "center_y", "axial_centers"):
        column_optimizer.set_bounds(parameter_name, lower=-0.8, upper=0.8)

    for parameter_name in ("scale_x", "scale_y", "axial_scales"):
        column_optimizer.set_bounds(parameter_name, lower=0.04, upper=0.65)
        column_optimizer.set_learning_rate({parameter_name: 0.005})
    column_optimizer.set_bounds("density_weights", lower=1e-5, upper=1.0)
    column_optimizer.set_learning_rate({"density_weights": 0.003, "rotation_angle": 0.02})
    return column_optimizer


def ReadParameters(column_optimizer):
    return {parameter_name: values.numpy().tolist() for parameter_name, values in column_optimizer.items()}


def GetFont(size=18):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def SaveImages(output_path, target_grid, initial_grid, fitted_grid):
    depth, height, width = target_grid.shape
    slice_positions = [int(np.argmin(abs(np.linspace(-1, 1, depth) - value))) for value in (-0.38, 0.0, 0.36)]
    contact_image = Image.new("RGB", (1300, 80 + 310 * len(slice_positions)), "white")
    drawing = ImageDraw.Draw(contact_image)
    drawing.text((15, 15), "One column / two axial Gaussians: fitting a known 3D density", fill="black", font=GetFont(22))
    drawing.text((15, 45), "Density range: 0..1; error range: 0..0.1 (white >= 0.1).", fill="black", font=GetFont(17))

    for row_index, slice_index in enumerate(slice_positions):
        initial_error = abs(target_grid[slice_index] - initial_grid[slice_index])
        fitted_error = abs(target_grid[slice_index] - fitted_grid[slice_index])
        values = [target_grid[slice_index], initial_grid[slice_index], initial_error / 0.1, fitted_grid[slice_index], fitted_error / 0.1]
        labels = ["target", "initial", "initial_error_max0p1", "fitted", "fitted_error_max0p1"]

        for column_index, (image_values, label) in enumerate(zip(values, labels)):
            image = Image.fromarray(np.uint8(np.rint(np.clip(image_values, 0.0, 1.0) * 255)))
            image.save(output_path / f"z{slice_index:03d}__{label}.png")
            preview = image.resize((240, 240), Image.Resampling.NEAREST).convert("RGB")
            contact_image.paste(preview, (column_index * 260 + 10, row_index * 310 + 135))
            drawing.text((column_index * 260 + 10, row_index * 310 + 90), f"z={slice_index}: {label}", fill="black", font=GetFont(16))

    contact_image.save(output_path / "density_fit__target_initial_fitted_errors.png")
    middle_y = height // 2
    middle_x = width // 2
    for label, grid in (("target", target_grid), ("initial", initial_grid), ("fitted", fitted_grid)):
        image_values = np.uint8(np.rint(np.clip(grid[:, middle_y, :], 0.0, 1.0) * 255))
        Image.fromarray(image_values).save(output_path / f"xz_section_y{middle_y:03d}__{label}.png")
        image_values = np.uint8(np.rint(np.clip(grid[:, :, middle_x], 0.0, 1.0) * 255))
        Image.fromarray(image_values).save(output_path / f"yz_section_x{middle_x:03d}__{label}.png")


def SaveLossCurve(output_path, loss_history):
    image = Image.new("RGB", (1000, 600), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text((25, 15), "Gaussian column density fitting: training MSE", fill="black", font=GetFont(22))
    drawing.text((25, 50), "Log display floor: 1e-16; CSV retains actual zero values.", fill="black", font=GetFont(16))
    logarithmic_losses = np.log10(np.maximum([row["training_mse"] for row in loss_history], 1e-16))
    lower_bound = math.floor(float(logarithmic_losses.min()))
    upper_bound = max(lower_bound + 1, math.ceil(float(logarithmic_losses.max())))

    for exponent in range(lower_bound, upper_bound + 1):
        position_y = 500 - (exponent - lower_bound) / (upper_bound - lower_bound) * 400
        drawing.line((100, position_y, 950, position_y), fill="#dddddd")
        drawing.text((15, position_y - 9), f"1e{exponent}", fill="black", font=GetFont(16))
    points = [(100 + index / max(len(loss_history) - 1, 1) * 850, 500 - (value - lower_bound) / (upper_bound - lower_bound) * 400) for index, value in enumerate(logarithmic_losses)]
    drawing.line(points, fill="#1769aa", width=3)
    drawing.line((100, 90, 100, 500, 950, 500), fill="black", width=2)

    for step in (0, len(loss_history) // 2, len(loss_history) - 1):
        drawing.text((100 + step / max(len(loss_history) - 1, 1) * 850 - 12, 515), str(step), fill="black", font=GetFont(16))
    drawing.text((460, 555), "Optimizer steps", fill="black", font=GetFont(19))
    image.save(output_path / "training_mse__log_scale.png")


def HorizontalCovariance(parameters):
    rotation_angle = parameters["rotation_angle"][0]
    rotation_matrix = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)], [math.sin(rotation_angle), math.cos(rotation_angle)]])
    return rotation_matrix @ np.diag([parameters["scale_x"][0] ** 2, parameters["scale_y"][0] ** 2]) @ rotation_matrix.T


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--sample_count", type=int, default=32768)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "gaussian_column_fit_results")
    arguments = parser.parse_args()
    if arguments.iterations <= 0 or arguments.sample_count < 1024:
        parser.error("iterations must be positive and sample_count must be at least 1024")

    target_parameters = {"center_x": [0.12], "center_y": [-0.08], "scale_x": [0.32], "scale_y": [0.18], "rotation_angle": [0.5],
                         "axial_centers": [-0.38, 0.36], "axial_scales": [0.13, 0.17], "density_weights": [0.08, 0.12]}
    initial_parameters = {"center_x": [-0.16], "center_y": [0.12], "scale_x": [0.4], "scale_y": [0.24], "rotation_angle": [-0.15],
                          "axial_centers": [-0.22, 0.19], "axial_scales": [0.21, 0.24], "density_weights": [0.07, 0.09]}
    target_arrays = {parameter_name: mi.Float(values) for parameter_name, values in target_parameters.items()}
    target_columns = GaussianColumn(target_arrays, components_per_column=2)
    column_optimizer = CreateOptimizer(initial_parameters)
    fitted_columns = GaussianColumn(column_optimizer, components_per_column=2)

    gaussian_kernel = GaussianKernel(mi.Properties())
    random_generator = np.random.default_rng(20261109)
    training_points = mi.Point3f(random_generator.uniform(-1, 1, (3, arguments.sample_count)).astype(np.float32))
    validation_points = mi.Point3f(random_generator.uniform(-1, 1, (3, 16384)).astype(np.float32))
    target_training_density = EvaluateDensity(target_columns, training_points, gaussian_kernel)
    target_validation_density = EvaluateDensity(target_columns, validation_points, gaussian_kernel)
    dr.eval(target_training_density, target_validation_density)

    # 固定规则网格只用于检查与出图，不参与梯度更新。
    grid_shape = (48, 64, 64)
    coordinates_z, coordinates_y, coordinates_x = np.meshgrid(np.linspace(-1, 1, grid_shape[0]), np.linspace(-1, 1, grid_shape[1]), np.linspace(-1, 1, grid_shape[2]), indexing="ij")
    grid_points = mi.Point3f([coordinates_x.ravel().astype(np.float32), coordinates_y.ravel().astype(np.float32), coordinates_z.ravel().astype(np.float32)])
    target_grid = EvaluateDensity(target_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)
    initial_grid = EvaluateDensity(fitted_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)

    loss_history = []
    first_gradients = {}
    for iteration in range(arguments.iterations):
        dr.clear_grad(column_optimizer.variables)
        predicted_density = EvaluateDensity(fitted_columns, training_points, gaussian_kernel)
        loss = dr.mean(dr.square(predicted_density - target_training_density))
        dr.backward(loss)

        if iteration == 0:
            first_gradients = {name: float(dr.sum(dr.abs(dr.grad(values)))[0]) for name, values in column_optimizer.items()}
        loss_history.append({"step": iteration, "training_mse": float(loss[0])})
        column_optimizer.step()
        if iteration == 0 or (iteration + 1) % 100 == 0:
            print(f"Step {iteration + 1:4d}/{arguments.iterations}: training MSE={loss_history[-1]['training_mse']:.8g}", flush=True)

    final_training_density = EvaluateDensity(fitted_columns, training_points, gaussian_kernel)
    loss_history.append({"step": arguments.iterations, "training_mse": float(dr.mean(dr.square(final_training_density - target_training_density))[0])})
    fitted_grid = EvaluateDensity(fitted_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)
    validation_error = EvaluateDensity(fitted_columns, validation_points, gaussian_kernel) - target_validation_density
    validation_rmse = math.sqrt(float(dr.mean(dr.square(validation_error))[0]))
    final_parameters = ReadParameters(column_optimizer)

    initial_mse = float(np.mean((initial_grid.astype(np.float64) - target_grid) ** 2))
    final_mse = float(np.mean((fitted_grid.astype(np.float64) - target_grid) ** 2))
    covariance_error = np.linalg.norm(HorizontalCovariance(final_parameters) - HorizontalCovariance(target_parameters)) / np.linalg.norm(HorizontalCovariance(target_parameters))
    metrics = {"initial_grid_rmse": math.sqrt(initial_mse), "final_grid_rmse": math.sqrt(final_mse), "validation_rmse": validation_rmse,
               "initial_grid_psnr_db": -10 * math.log10(initial_mse), "final_grid_psnr_db": -10 * math.log10(final_mse) if final_mse > 0 else None,
               "zero_grid_error_at_float32_precision": final_mse == 0,
               "mse_reduction_factor": initial_mse / final_mse if final_mse > 0 else None, "horizontal_covariance_relative_error": float(covariance_error),
               "psnr_peak": 1.0, "grid_axis_order": "z,y,x", "grid_shape": grid_shape, "seed": 20261109, "first_step_gradient_l1": first_gradients}

    center_error = max(abs(final_parameters[name][0] - target_parameters[name][0]) for name in ("center_x", "center_y"))
    component_order = np.argsort(final_parameters["axial_centers"])
    axial_center_error = float(np.max(abs(np.asarray(final_parameters["axial_centers"])[component_order] - target_parameters["axial_centers"])))
    axial_scale_error = float(np.max(abs(np.asarray(final_parameters["axial_scales"])[component_order] / target_parameters["axial_scales"] - 1)))
    weight_error = float(np.max(abs(np.asarray(final_parameters["density_weights"])[component_order] / target_parameters["density_weights"] - 1)))
    metrics.update(horizontal_center_max_error=center_error, axial_center_max_error=axial_center_error, axial_scale_max_relative_error=axial_scale_error, weight_max_relative_error=weight_error)

    expanded_centers, expanded_scales, expanded_quaternions, expanded_weights = fitted_columns.Expand()
    np.testing.assert_allclose(expanded_centers.numpy()[:2, 0], expanded_centers.numpy()[:2, 1])
    np.testing.assert_allclose(expanded_scales.numpy()[:2, 0], expanded_scales.numpy()[:2, 1])
    np.testing.assert_allclose(expanded_quaternions.numpy()[:, 0], expanded_quaternions.numpy()[:, 1])
    passed = all(value > 1e-12 for value in first_gradients.values()) and final_mse < initial_mse / 100 and validation_rmse < 0.002
    passed = passed and center_error < 0.02 and covariance_error < 0.05 and axial_center_error < 0.02 and axial_scale_error < 0.05 and weight_error < 0.05
    metrics["passed"] = bool(passed)

    output_path = arguments.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_path / "parameters.json").write_text(json.dumps({"target": target_parameters, "initial": initial_parameters, "fitted": final_parameters}, indent=2), encoding="utf-8")
    with (output_path / "loss_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "training_mse"])
        writer.writeheader()
        writer.writerows(loss_history)

    SaveImages(output_path, target_grid, initial_grid, fitted_grid)
    SaveLossCurve(output_path, loss_history)
    final_psnr_label = f"{metrics['final_grid_psnr_db']:.3f} dB" if final_mse > 0 else "infinite (zero float32 error)"
    print(f"Grid PSNR: {metrics['initial_grid_psnr_db']:.3f} dB -> {final_psnr_label}")
    print(f"Validation RMSE: {validation_rmse:.8g}; covariance relative error: {covariance_error:.8g}")
    print(f"Outputs: {output_path}")
    assert passed, "Density fit did not meet convergence/parameter-recovery checks; inspect metrics.json"
    print("PASS: shared XY footprint and independent axial Gaussians recover the known target density.")


if __name__ == "__main__":
    Main()
