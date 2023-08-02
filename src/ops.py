"""Low-level image transformations and operations.
"""

import cv2
import numpy as np


def rotate(image, angle, flags = cv2.INTER_LINEAR):
    """Rotate an image counterclockwise by an arbitrary angle.

    Arguments
    ---------
    image: numpy.ndarray
        The image to rotate.

    angle: float
        The angle to rotate counterclockwise in degrees.
    """
    # Based on https://stackoverflow.com/a/9042907/1166039
    # Get center of image.
    dims = image.shape[1::-1]  # cols, rows
    center = np.array(dims) / 2

    # Third rotation matrix argument is scale.
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation, dims, flags = flags)


def unsharp_mask(image, sigma, size = (0, 0)):
    """Apply an unsharp mask to an image.

    An unsharp mask is a kind of high-pass filter, meaning it preserves
    abrupt changes in the pixels of an image. Unsharp masking is usually used
    to sharpen images that will be viewed by people rather than for machine
    processing.

    Arguments
    ---------
    image: numpy.ndarray
        The image to which to apply the filter.

    sigma: float
        The standard deviation for the Gaussian kernel.

    size: tuple of ints
        The width and height of the kernel. Computed from sigma if both are 0;
        otherwise both must be odd.
    """
    blurred = cv2.GaussianBlur(image, size, sigma)
    return cv2.addWeighted(image, 2.0, blurred, -1.0, 0.0)


def rescale(image, max_size):
    """Rescale an image to a maximum size, preserving the aspect ratio.
    """
    # Determine which dimension to resize.
    h, w = image.shape[:2]
    m = max(w, h)
    if m <= max_size:
        return image

    scaling = max_size / max(w, h)
    w = int(round(w * scaling))
    h = int(round(h * scaling))
    return cv2.resize(image, (w, h))
