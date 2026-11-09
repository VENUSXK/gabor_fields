"""Run from the repository root: python -m gfields.tests.test_gaussian_column."""
import numpy as np
import drjit as dr
import mitsuba as mi
import gfields
from gfields.integrators.common import GaussianColumn


def Main():
    # 一根柱只有一套横向参数；两个轴向分量位于不同高度。
    column_optimizer = gfields.optimizers.BoundedAdam(lr=0.1, noise_init=0.0)
    column_optimizer["center_x"] = mi.Float([0.0])
    column_optimizer["center_y"] = mi.Float([0.1])
    column_optimizer["scale_x"] = mi.Float([0.3])
    column_optimizer["scale_y"] = mi.Float([0.15])
    column_optimizer["rotation_angle"] = mi.Float([0.25])

    column_optimizer["axial_centers"] = mi.Float([-0.4, 0.4])
    column_optimizer["axial_scales"] = mi.Float([0.1, 0.2])
    column_optimizer["density_weights"] = mi.Float([0.5, 0.8])
    gaussian_columns = GaussianColumn(column_optimizer, components_per_column=2)
    centers, scales, quaternions, density_weights = gaussian_columns.Expand()

    print("1. Stored parameters: one XY footprint, two axial components.")
    print("   Shared center_x:", column_optimizer["center_x"].numpy())
    print("   Expanded centers (rows=x,y,z):\n", centers.numpy())
    print("   Expanded scales (rows=x,y,z):\n", scales.numpy())
    print("   Expanded rotations (rows=quaternion components):\n", quaternions.numpy())
    print("   Independent density weights:", density_weights.numpy())

    np.testing.assert_allclose(centers.x.numpy(), [0.0, 0.0])
    np.testing.assert_allclose(centers.y.numpy(), [0.1, 0.1])
    np.testing.assert_allclose(centers.z.numpy(), [-0.4, 0.4])
    np.testing.assert_allclose(scales.numpy()[:2, 0], scales.numpy()[:2, 1])
    np.testing.assert_allclose(quaternions.numpy()[:, 0], quaternions.numpy()[:, 1])

    # 两个分量分别希望移动到不同的x位置，检验梯度能否累加回同一个参数。
    target_centers_x = mi.Float([0.5, 1.0])
    loss = dr.sum(dr.square(centers.x - target_centers_x))
    expected_component_gradients = 2.0 * (centers.x.numpy() - target_centers_x.numpy())
    dr.backward(loss)
    shared_gradient = dr.grad(column_optimizer["center_x"]).numpy()

    print("2. Diagnostic loss: sum((expanded_center_x - target_x)^2).")
    print("   Target x values:", target_centers_x.numpy())
    print("   Expected component gradients:", expected_component_gradients)
    print("   Gradient on the single shared center_x:", shared_gradient)
    np.testing.assert_allclose(shared_gradient, [expected_component_gradients.sum()])
    np.testing.assert_allclose(shared_gradient, [-3.0])

    column_optimizer.step()
    updated_centers, updated_scales, updated_quaternions, updated_weights = gaussian_columns.Expand()
    print("3. After one Adam step:")
    print("   Shared center_x:", column_optimizer["center_x"].numpy())
    print("   Both expanded x values:", updated_centers.x.numpy())
    print("   Independent z values:", updated_centers.z.numpy())

    np.testing.assert_allclose(updated_centers.x.numpy(), [0.1, 0.1], atol=1e-5)
    np.testing.assert_allclose(updated_centers.z.numpy(), [-0.4, 0.4])
    np.testing.assert_allclose(updated_scales.numpy(), scales.numpy())
    np.testing.assert_allclose(updated_quaternions.numpy(), quaternions.numpy())
    np.testing.assert_allclose(updated_weights.numpy(), density_weights.numpy())
    print("PASS: the two Gaussians share XY parameters and accumulate gradients into one footprint.")


if __name__ == "__main__":
    Main()
