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


def evaluate_density(gaussian_columns, query_points, gaussian_kernel):
    centers, scales, quaternions, primitive_opacity = gaussian_columns.expand()
    density = mi.Float(0.0)
    for primitive_index in range(dr.width(primitive_opacity)):
        parameter_index = mi.UInt32(primitive_index)
        quaternion = dr.gather(mi.Quaternion4f, quaternions, parameter_index)
        ellipsoid = Ellipsoid(center=dr.gather(mi.Point3f, centers, parameter_index), scale=dr.gather(mi.Vector3f, scales, parameter_index), quat=quaternion, rot=dr.quat_to_matrix(mi.Matrix3f, quaternion))
        density += dr.gather(mi.Float, primitive_opacity, parameter_index) * gaussian_kernel.pdf(query_points, ellipsoid, mi.Bool(True))
    return density


def create_optimizer(parameter_values):
    column_optimizer = gfields.optimizers.BoundedAdam(lr=0.01, noise_init=0.0)
    for parameter_name, values in parameter_values.items():
        parameter_type = mi.Point2f if parameter_name == 'column_center_xy' else mi.Vector2f if parameter_name == 'column_scale_xy' else mi.Float
        column_optimizer[parameter_name] = parameter_type(values)
    for parameter_name in ("column_center_xy", "primitive_center_z"):
        column_optimizer.set_bounds(parameter_name, lower=-0.8, upper=0.8)

    for parameter_name in ("column_scale_xy", "primitive_scale_z"):
        column_optimizer.set_bounds(parameter_name, lower=0.04, upper=0.65)
        column_optimizer.set_learning_rate({parameter_name: 0.005})
    column_optimizer.set_bounds("primitive_opacity", lower=1e-5, upper=1.0)
    column_optimizer.set_learning_rate({"primitive_opacity": 0.003, "column_quaternions": 0.02})
    return column_optimizer


def read_parameters(column_optimizer):
    return {parameter_name: values.numpy().tolist() for parameter_name, values in column_optimizer.items()}


def get_font(size=18):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_images(output_path, target_grid, initial_grid, fitted_grid):
    depth, height, width = target_grid.shape
    slice_positions = [int(np.argmin(abs(np.linspace(-1, 1, depth) - value))) for value in (-0.38, 0.0, 0.36)]
    contact_image = Image.new("RGB", (1300, 80 + 310 * len(slice_positions)), "white")
    drawing = ImageDraw.Draw(contact_image)
    drawing.text((15, 15), "One column / two axial Gaussians: fitting a known 3D density", fill="black", font=get_font(22))
    drawing.text((15, 45), "Density range: 0..1; error range: 0..0.1 (white >= 0.1).", fill="black", font=get_font(17))

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
            drawing.text((column_index * 260 + 10, row_index * 310 + 90), f"z={slice_index}: {label}", fill="black", font=get_font(16))

    contact_image.save(output_path / "density_fit__target_initial_fitted_errors.png")
    middle_y = height // 2
    middle_x = width // 2
    for label, grid in (("target", target_grid), ("initial", initial_grid), ("fitted", fitted_grid)):
        image_values = np.uint8(np.rint(np.clip(grid[:, middle_y, :], 0.0, 1.0) * 255))
        Image.fromarray(image_values).save(output_path / f"xz_section_y{middle_y:03d}__{label}.png")
        image_values = np.uint8(np.rint(np.clip(grid[:, :, middle_x], 0.0, 1.0) * 255))
        Image.fromarray(image_values).save(output_path / f"yz_section_x{middle_x:03d}__{label}.png")


def save_loss_curve(output_path, loss_history):
    image = Image.new("RGB", (1000, 600), "white")
    drawing = ImageDraw.Draw(image)
    drawing.text((25, 15), "Gaussian column density fitting: training MSE", fill="black", font=get_font(22))
    drawing.text((25, 50), "Log display floor: 1e-16; CSV retains actual zero values.", fill="black", font=get_font(16))
    logarithmic_losses = np.log10(np.maximum([row["training_mse"] for row in loss_history], 1e-16))
    lower_bound = math.floor(float(logarithmic_losses.min()))
    upper_bound = max(lower_bound + 1, math.ceil(float(logarithmic_losses.max())))

    for exponent in range(lower_bound, upper_bound + 1):
        position_y = 500 - (exponent - lower_bound) / (upper_bound - lower_bound) * 400
        drawing.line((100, position_y, 950, position_y), fill="#dddddd")
        drawing.text((15, position_y - 9), f"1e{exponent}", fill="black", font=get_font(16))
    points = [(100 + index / max(len(loss_history) - 1, 1) * 850, 500 - (value - lower_bound) / (upper_bound - lower_bound) * 400) for index, value in enumerate(logarithmic_losses)]
    drawing.line(points, fill="#1769aa", width=3)
    drawing.line((100, 90, 100, 500, 950, 500), fill="black", width=2)

    for step in (0, len(loss_history) // 2, len(loss_history) - 1):
        drawing.text((100 + step / max(len(loss_history) - 1, 1) * 850 - 12, 515), str(step), fill="black", font=get_font(16))
    drawing.text((460, 555), "Optimizer steps", fill="black", font=get_font(19))
    image.save(output_path / "training_mse__log_scale.png")


def horizontal_covariance(parameters):
    column_quaternions = parameters["column_quaternions"][0]
    rotation_matrix = np.array([[math.cos(column_quaternions), -math.sin(column_quaternions)], [math.sin(column_quaternions), math.cos(column_quaternions)]])
    return rotation_matrix @ np.diag([parameters["column_scale_xy"][0][0] ** 2, parameters["column_scale_xy"][1][0] ** 2]) @ rotation_matrix.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--sample_count", type=int, default=32768)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "gaussian_column_fit_results")
    arguments = parser.parse_args()
    if arguments.iterations <= 0 or arguments.sample_count < 1024:
        parser.error("iterations must be positive and sample_count must be at least 1024")

    target_parameters = {"column_center_xy": [[0.12], [-0.08]], "column_scale_xy": [[0.32], [0.18]], "column_quaternions": [0.5],
                         "primitive_center_z": [-0.38, 0.36], "primitive_scale_z": [0.13, 0.17], "primitive_opacity": [0.08, 0.12]}
    initial_parameters = {"column_center_xy": [[-0.16], [0.12]], "column_scale_xy": [[0.4], [0.24]], "column_quaternions": [-0.15],
                          "primitive_center_z": [-0.22, 0.19], "primitive_scale_z": [0.21, 0.24], "primitive_opacity": [0.07, 0.09]}
    parameter_types = {'column_center_xy': mi.Point2f, 'column_scale_xy': mi.Vector2f}
    target_arrays = {name: parameter_types.get(name, mi.Float)(values) for name, values in target_parameters.items()}
    target_columns = GaussianColumn(target_arrays, primitives_per_column=2)
    column_optimizer = create_optimizer(initial_parameters)
    fitted_columns = GaussianColumn(column_optimizer, primitives_per_column=2)

    gaussian_kernel = GaussianKernel(mi.Properties())
    random_generator = np.random.default_rng(20261109)
    training_points = mi.Point3f(random_generator.uniform(-1, 1, (3, arguments.sample_count)).astype(np.float32))
    validation_points = mi.Point3f(random_generator.uniform(-1, 1, (3, 16384)).astype(np.float32))
    target_training_density = evaluate_density(target_columns, training_points, gaussian_kernel)
    target_validation_density = evaluate_density(target_columns, validation_points, gaussian_kernel)
    dr.eval(target_training_density, target_validation_density)

    # 固定规则网格只用于检查与出图，不参与梯度更新。
    grid_shape = (48, 64, 64)
    coordinates_z, coordinates_y, coordinates_x = np.meshgrid(np.linspace(-1, 1, grid_shape[0]), np.linspace(-1, 1, grid_shape[1]), np.linspace(-1, 1, grid_shape[2]), indexing="ij")
    grid_points = mi.Point3f([coordinates_x.ravel().astype(np.float32), coordinates_y.ravel().astype(np.float32), coordinates_z.ravel().astype(np.float32)])
    target_grid = evaluate_density(target_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)
    initial_grid = evaluate_density(fitted_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)

    loss_history = []
    first_gradients = {}
    for iteration in range(arguments.iterations):
        dr.clear_grad(column_optimizer.variables)
        predicted_density = evaluate_density(fitted_columns, training_points, gaussian_kernel)
        loss = dr.mean(dr.square(predicted_density - target_training_density))
        dr.backward(loss)

        if iteration == 0:
            first_gradients = {name: float(dr.sum(dr.abs(dr.grad(values)), axis=None)[0]) for name, values in column_optimizer.items()}
        loss_history.append({"step": iteration, "training_mse": float(loss[0])})
        column_optimizer.step()
        if iteration == 0 or (iteration + 1) % 100 == 0:
            print(f"Step {iteration + 1:4d}/{arguments.iterations}: training MSE={loss_history[-1]['training_mse']:.8g}", flush=True)

    final_training_density = evaluate_density(fitted_columns, training_points, gaussian_kernel)
    loss_history.append({"step": arguments.iterations, "training_mse": float(dr.mean(dr.square(final_training_density - target_training_density))[0])})
    fitted_grid = evaluate_density(fitted_columns, grid_points, gaussian_kernel).numpy().reshape(grid_shape)
    validation_error = evaluate_density(fitted_columns, validation_points, gaussian_kernel) - target_validation_density
    validation_rmse = math.sqrt(float(dr.mean(dr.square(validation_error))[0]))
    final_parameters = read_parameters(column_optimizer)

    initial_mse = float(np.mean((initial_grid.astype(np.float64) - target_grid) ** 2))
    final_mse = float(np.mean((fitted_grid.astype(np.float64) - target_grid) ** 2))
    covariance_error = np.linalg.norm(horizontal_covariance(final_parameters) - horizontal_covariance(target_parameters)) / np.linalg.norm(horizontal_covariance(target_parameters))
    metrics = {"initial_grid_rmse": math.sqrt(initial_mse), "final_grid_rmse": math.sqrt(final_mse), "validation_rmse": validation_rmse,
               "initial_grid_psnr_db": -10 * math.log10(initial_mse), "final_grid_psnr_db": -10 * math.log10(final_mse) if final_mse > 0 else None,
               "zero_grid_error_at_float32_precision": final_mse == 0,
               "mse_reduction_factor": initial_mse / final_mse if final_mse > 0 else None, "horizontal_covariance_relative_error": float(covariance_error),
               "psnr_peak": 1.0, "grid_axis_order": "z,y,x", "grid_shape": grid_shape, "seed": 20261109, "first_step_gradient_l1": first_gradients}

    center_error = float(np.max(np.abs(np.asarray(final_parameters["column_center_xy"]) - target_parameters["column_center_xy"])))
    primitive_order = np.argsort(final_parameters["primitive_center_z"])
    axial_center_error = float(np.max(abs(np.asarray(final_parameters["primitive_center_z"])[primitive_order] - target_parameters["primitive_center_z"])))
    axial_scale_error = float(np.max(abs(np.asarray(final_parameters["primitive_scale_z"])[primitive_order] / target_parameters["primitive_scale_z"] - 1)))
    weight_error = float(np.max(abs(np.asarray(final_parameters["primitive_opacity"])[primitive_order] / target_parameters["primitive_opacity"] - 1)))
    metrics.update(horizontal_center_max_error=center_error, axial_center_max_error=axial_center_error, axial_scale_max_relative_error=axial_scale_error, weight_max_relative_error=weight_error)

    expanded_centers, expanded_scales, expanded_quaternions, expanded_weights = fitted_columns.expand()
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

    save_images(output_path, target_grid, initial_grid, fitted_grid)
    save_loss_curve(output_path, loss_history)
    final_psnr_label = f"{metrics['final_grid_psnr_db']:.3f} dB" if final_mse > 0 else "infinite (zero float32 error)"
    print(f"Grid PSNR: {metrics['initial_grid_psnr_db']:.3f} dB -> {final_psnr_label}")
    print(f"Validation RMSE: {validation_rmse:.8g}; covariance relative error: {covariance_error:.8g}")
    print(f"Outputs: {output_path}")
    assert passed, "Density fit did not meet convergence/parameter-recovery checks; inspect metrics.json"
    print("PASS: shared XY footprint and independent axial Gaussians recover the known target density.")


if __name__ == "__main__":
    main()
