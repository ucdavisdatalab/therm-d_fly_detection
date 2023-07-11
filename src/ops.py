"""Low-level image transformations and operations.
"""

import cv2


def unsharp_mask(img, sigma, size = (0, 0)):
    """Apply an unsharp mask to an image.

    An unsharp mask is a kind of high-pass filter, meaning it preserves
    abrupt changes in the pixels of an image. Unsharp masking is usually used
    to sharpen images that will be viewed by people rather than for machine
    processing.

    Arguments
    ---------
    img:
        The image to which to apply the filter.

    sigma: float
        The standard deviation for the Gaussian kernel.

    size: tuple of ints
        The width and height of the kernel. Computed from sigma if both are 0;
        otherwise both must be odd.
    """
    blurred = cv2.GaussianBlur(img, size, sigma)
    return cv2.addWeighted(img, 2.0, blurred, -1.0, 0.0)
