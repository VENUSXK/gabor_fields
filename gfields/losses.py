# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Image losses / metrics for the training loop (Dr.Jit / Mitsuba).

A native Dr.Jit ``SSIMLoss`` plus the scalar ``l1`` / ``l2`` / ``psnr`` helpers.
Split out of ``gfields.optimizers`` (which now holds only the ``BoundedAdam``
optimizer) so the loss math has its own home.
"""

from __future__ import annotations  # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi


# ------------------------------------------------------------------------------
# Native DrJit SSIM implementation

def _make_gaussian_filter(sigma: float):
    """
    Create a Gaussian filter function using DrJit operations.

    Args:
        sigma: Standard deviation of the Gaussian in pixels

    Returns:
        A callable filter function for dr.convolve
    """
    def gaussian_filter(x: float) -> float:
        return dr.exp(-x*x * dr.rcp(2 * sigma*sigma))
    return gaussian_filter


def _convolve_channels(tensor: mi.TensorXf, filter_fn, filter_radius: float) -> mi.TensorXf:
    """
    Apply Gaussian convolution to an [H, W, C] tensor over spatial dimensions.

    dr.convolve applies the filter to all axes by default, but we want to
    convolve only the spatial dimensions (H, W), not the channel dimension.

    Args:
        tensor: Input tensor of shape [H, W, C]
        filter_fn: Filter function for dr.convolve
        filter_radius: Radius of the filter in pixels

    Returns:
        Convolved tensor of shape [H, W, C]
    """
    # Convolve spatial dimensions only (axis 0 and 1), leaving channels (axis 2) untouched
    return dr.convolve(tensor, filter_fn, filter_radius, axis=(0, 1))


class SSIMLoss:
    """
    Native DrJit implementation of SSIM (Structural Similarity Index) loss.

    Args:
        window_size: Size of the Gaussian window (default: 11)
        sigma: Standard deviation of the Gaussian filter (default: 1.5)
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2  # Stability constant for luminance
        self.C2 = 0.03 ** 2  # Stability constant for contrast
        # Filter radius covers approximately 3*sigma for good Gaussian approximation
        self.filter_radius = max(window_size // 2, sigma * 3)
        self._gaussian_filter = _make_gaussian_filter(sigma)

        # Reference image cache for precomputation
        self.ref_conv_cache = {}

    def reset_img_cache(self):
        """Clear the reference image convolution cache."""
        self.ref_conv_cache = {}

    def precompute_reference_convs(self, ref_images, num_cams: int = None):
        """
        Precompute convolutions for reference images to speed up repeated evaluations.

        Args:
            ref_images: Reference image tensor [H, W, C], [H, W*num_cams, C], or a list of such tensors
            num_cams: Number of cameras (used to split images horizontally)
        """
        # Clear old cache before computing new reference convolutions
        self.reset_img_cache()

        # Handle list of reference images
        if isinstance(ref_images, list):
            for ref_img in ref_images:
                self._precompute_single_ref(ref_img, num_cams)
        else:
            self._precompute_single_ref(ref_images, num_cams)

    def _precompute_single_ref(self, ref_images: mi.TensorXf, num_cams: int = None):
        """Precompute convolutions for a single reference image tensor."""
        # Validate multi-camera case (if applicable)
        if num_cams is not None:
            nW = ref_images.shape[1]
            assert nW % num_cams == 0, "Image width must be divisible by number of cameras"

        # Compute mu1 and sigma1_sq
        mu1 = _convolve_channels(ref_images, self._gaussian_filter, self.filter_radius)
        sigma1_sq = _convolve_channels(ref_images * ref_images, self._gaussian_filter, self.filter_radius) - mu1 * mu1

        # Evaluate and cache keyed by tensor id
        dr.eval(mu1, sigma1_sq)

        self.ref_conv_cache[id(ref_images)] = {
            'mu1': mu1,
            'sigma1_sq': sigma1_sq,
        }

    def __call__(self, ref: mi.TensorXf, pred: mi.TensorXf, num_cams: int = None) -> mi.Float:
        """
        Compute SSIM loss between reference and predicted images.

        Args:
            ref: Reference image tensor [H, W, C]
            pred: Predicted image tensor [H, W, C]
            num_cams: Number of cameras (used to split images horizontally)

        Returns:
            1 - mean(SSIM), as a scalar DrJit Float
        """
        return self.forward(ref, pred, num_cams)

    def forward(self, ref: mi.TensorXf, pred: mi.TensorXf, num_cams: int = None) -> mi.Float:
        """
        Compute SSIM loss between reference and predicted images.

        The SSIM formula is:
        SSIM(x,y) = ((2*μx*μy + C1) * (2*σxy + C2)) / ((μx² + μy² + C1) * (σx² + σy² + C2))

        Args:
            ref: Reference image tensor [H, W, C]
            pred: Predicted image tensor [H, W, C]
            num_cams: Number of cameras (used to split images horizontally)

        Returns:
            1 - mean(SSIM), as a scalar DrJit Float
        """
        # Compute local means using Gaussian convolution
        # Check if we have cached reference convolutions (keyed by tensor id)
        ref_id = id(ref)
        if ref_id in self.ref_conv_cache:
            mu1 = self.ref_conv_cache[ref_id]['mu1']
            sigma1_sq = self.ref_conv_cache[ref_id]['sigma1_sq']
        else:
            mu1 = _convolve_channels(ref, self._gaussian_filter, self.filter_radius)
            sigma1_sq = _convolve_channels(ref * ref, self._gaussian_filter, self.filter_radius) - mu1 * mu1

        mu2 = _convolve_channels(pred, self._gaussian_filter, self.filter_radius)

        # Local statistics
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        # Local variances and covariance
        sigma2_sq = _convolve_channels(pred * pred, self._gaussian_filter, self.filter_radius) - mu2_sq
        sigma12 = _convolve_channels(ref * pred, self._gaussian_filter, self.filter_radius) - mu1_mu2

        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)

        ssim_map = numerator / denominator

        # Return 1 - mean(SSIM) as the loss (dissimilarity)
        return 1 - dr.mean(ssim_map, axis=None)


def l1(reference, image) -> mi.Float:
    '''
    L1 loss function
    '''
    return dr.mean(dr.abs(reference - image), axis=None)


def l2(reference, image) -> mi.Float:
    '''
    L2 loss function
    '''
    return dr.mean(dr.power(reference - image, 2), axis=None)


def psnr(reference, image) -> mi.Float:
    '''PSNR metric (unfrozen - for use where gradients needed)'''
    return 20 * dr.log(1.0 * dr.rsqrt(l2(reference, image))) * dr.rcp(dr.log(10))


@dr.freeze
def psnr_frozen(reference, image) -> mi.Float:
    '''PSNR metric (frozen - for use in dr.suspend_grad() or logging)'''
    mse = dr.mean(dr.power(reference - image, 2), axis=None)
    return - 10 * dr.log(mse) * dr.rcp(dr.log(10))
