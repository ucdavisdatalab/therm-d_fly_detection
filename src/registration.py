"""Functions for detecting registration marks.
"""

import cv2
import numpy as np


def orient_and_bound(alignment_mark, position_marks):
    """Compute orientation and corners of a fly arena from its (orange)
    alignment mark and (green) position marks.
    """
    if not isinstance(alignment_mark, list):
        alignment_mark = [alignment_mark]
    elif len(alignment_mark) != 1:
        raise ValueError("There must be only 1 alignment mark.")

    # Compute median of each mark.
    marks = alignment_mark + position_marks
    medians = np.stack([np.median(c, axis = 0) for c in marks])
    # Get indexes of top left, top right, bottom right, bottom left.
    mark_order = arg_corner_sort(medians)

    # Determine orientation by checking where the (orange) alignment mark is.
    match np.where(mark_order == 0)[0]:
        case 0:  # Top left (the correct orientation)
            orient = None
        case 1:  # Top right
            orient = cv2.ROTATE_90_COUNTERCLOCKWISE
        case 2:  # Bottom right
            orient = cv2.ROTATE_180
        case 3:  # Bottom left
            orient = cv2.ROTATE_90_CLOCKWISE

    # Find the four corners of the arena.
    # Get top-left corner of top-left mark, etc...
    arena = np.empty_like(marks[0])
    for i, ix in enumerate(mark_order):
        mark = marks[ix]
        corner_order = arg_corner_sort(mark)
        arena[i, :] = mark[corner_order[i]]

    return orient, arena


def arg_corner_sort(points):
    """Find top-left, top-right, bottom-right, bottom-left ordering of four
    points.
    """
    if points.shape[0] != 4 or points.shape[1] != 2:
        raise ValueError("Argument points must be a 4x2 array.")

    # Use medians to divide points into quadrants.
    m = np.median(points, axis = 0)
    order = np.concatenate([
        np.where((points[:, 0] < m[0]) & (points[:, 1] < m[1]))    # tl
        , np.where((points[:, 0] > m[0]) & (points[:, 1] < m[1]))  # tr
        , np.where((points[:, 0] > m[0]) & (points[:, 1] > m[1]))  # br
        , np.where((points[:, 0] < m[0]) & (points[:, 1] > m[1]))  # bl
    ])

    return np.squeeze(order)


def aspect_ratio(contour):
    """Compute the aspect ratio (width / height) of a given contour.
    """
    _, _, w, h = cv2.boundingRect(contour)
    return w / h


def find_k_similar(x, k, sort = False):
    """Find positions and values of the k most similar adjacent values in an
    array.

    Arguments
    ---------
    x
        The array to search.

    k
        The number of consecutive values for which to search.

    sort
        Sort x before searching?
    """
    if sort:
        x = sorted(x)
    std = [np.std(x[i:i + k]) for i in range(len(x) - k + 1)]
    ix = np.argmin(std)
    return ix, x[ix:ix + k]


def compute_squares(
    image, n_squares, min_area_proportion = 0.0001, max_area_proportion = 0.1
    , min_aspect = 0.8, max_aspect = 1.2
):
    """Compute square contours within a given mask.

    Arguments
    ---------
    image: numpy.ndarray
        The image in which to search for squares. This should be a mask (a
        2-color image).

    n_squares: int
        The number of squares for which to search.

    min_area_proportion: float
        Minimum proportion of image area for each square.

    max_area_proportion: float
        Maximum proportion of image area for each square.

    min_aspect: float
        Minimum aspect ratio for each square.

    max_aspect: float
        Maximum aspect ratio for each square.
    """
    # Find contours in the image.
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check that contour aspect ratios are near 1 and areas are not too small
    # or large.
    h, w = image.shape[:2]
    image_area = h * w
    min_area = image_area * min_area_proportion
    max_area = image_area * max_area_proportion
    # print(f"{h=}, {w=}")

    area = np.array([cv2.contourArea(c) for c in contours])
    ar = np.array([aspect_ratio(c) for c in contours])
    is_ok = (min_area < area) & (area < max_area)
    is_ok &= (min_aspect < ar) & (ar < max_aspect)
    contours = [x for i, x in enumerate(contours) if is_ok[i]]
    area = area[is_ok]

    if len(contours) < n_squares:
        raise ValueError(f"Only {len(contours)} contours found.")

    # Sort by area (largest to smallest) and get n_squares with the most
    # similar areas.
    if len(contours) > 1:
        ix = np.argsort(-area)
        area = area[ix]
        contours = [contours[i] for i in ix]

    if n_squares > 1:
        ix, area = find_k_similar(area, n_squares)
        contours = contours[ix:ix + n_squares]
    else:
        # Get largest contour.
        contours = contours[:1]

    # Find smallest rotated rectangle enclosing the contour.
    contours = [
        cv2.boxPoints(cv2.minAreaRect(x)).astype(int)
        for x in contours]

    return contours, area


def mask_hsv(
    # turqouise: (60, 30, 127), (120, 255, 255))
    image, lower = (60, 127, 127), upper = (120, 255, 255), close_kernel = 5,
    open_kernel = 5
):
    """Mask all pixels except those in a specific hue range.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)

    # Use close op to close holes.
    close_kernel = (close_kernel, close_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Use open op to remove noise.
    open_kernel = (open_kernel, open_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
