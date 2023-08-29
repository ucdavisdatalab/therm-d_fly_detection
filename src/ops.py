"""Low-level image transformations and operations.
"""

import cv2
import numpy as np


def gamma_transform(
    image, gamma = None, gamma_quantile = 0.25, verbose = False
):
    """Adjust the gamma of an image.

    Arguments
    ---------
    image: np.ndarray
        The image to transform.

    gamma: float or None
        The gamma exponent. If this is None, it will be estimated automatically
        so that a quantile of the image gets a 50% value.

    gamma_quantile: float
        If `gamma` is None, the quantile to assign to a 50% value.

    verbose: bool
        Print diagnostic messages?
    """
    if gamma is None:
        # Estimate gamma.
        gamma = np.log(np.quantile(image, gamma_quantile)) / np.log(128)

    if verbose:
        print(f"Gamma: {gamma:.2f}")

    # Use a lookup table to quickly compute the new pixel values.
    lookup_table = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookup_table[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookup_table)


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


def get_perspective_transform(contour):
    """Get perspective transformation that makes the area inside of the given
    4-point contour into a rectangular image.

    Arguments
    ---------
    contour: np.ndarray
        A contour with 4 points corresponding to the top left, top right,
        bottom right, and bottom left.

    Returns
    -------
    out
        A 3-tuple that contains an OpenCV perspective transformation, the new
        width, and the new height.
    """
    # Compute size of new image based on max length of each pair of parallel
    # sides.
    lengths = np.linalg.norm(contour - contour[[1, 2, 3, 0], :], axis = 1)
    lengths = np.rint(lengths).astype(int)
    width = lengths[::2].max()
    height = lengths[1::2].max()

    # The `getPerspectiveTransform` function requires float32 arguments.
    contour = np.float32(contour)
    output_xy = np.float32([
        [0, 0]
        , [width - 1, 0]
        , [width - 1, height - 1]
        , [0, height - 1]
    ])

    return cv2.getPerspectiveTransform(contour, output_xy), width, height


def rotate_contour(contour, shape, rotation):
    """Rotate a contour within an image by a given multiple of 90 degrees.

    Arguments
    ---------
    box: np.ndarray
        A contour (a n-by-2 matrix where the columns are x and y coordinates,
        respectively).

    shape: tuple
        Shape of the image (after rotation).

    rotation:
        How to rotate the contour, from OpenCV's rotation constants such as
        `cv2.ROTATE_180`.
    """
    h, w, _ = shape
    match rotation:
        case cv2.ROTATE_90_CLOCKWISE:
            contour = contour[[3, 0, 1, 2], ::-1].copy()
            contour[:, 0] = w - contour[:, 0]
            # Fix the corner sort.
            return contour
        case cv2.ROTATE_180:
            contour = contour[[2, 3, 0, 1], :].copy()
            contour[:, 0] = w - contour[:, 0]
            contour[:, 1] = h - contour[:, 1]
            return contour
        case cv2.ROTATE_90_COUNTERCLOCKWISE:
            contour = contour[[1, 2, 3, 0], ::-1].copy()
            contour[:, 1] = h - contour[:, 1]
        case _:
            return contour
