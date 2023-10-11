"""Functions for detecting registration marks.
"""

import itertools as it

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
    , tol_aspect_ratio = 0.2, tol_rect_error = 0.25
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

    tol_aspect_ratio: float
        Tolerance for how much each square's aspect ratio can deviate from 1.

    tol_rect_error: float
        Tolerance for percent error between squares and the contours on which
        they are based. Values closer to 0 require contours to be more
        rectangular, while values closer to 1 allow contours to be less
        rectangular.
    """
    # Find contours in the image.
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Eliminate contours that are too large.
    image_area = image.shape[0] * image.shape[1]
    min_area = min_area_proportion * image_area
    max_area = max_area_proportion * image_area

    areas = np.array([cv2.contourArea(c) for c in contours])
    ok_areas = (min_area <= areas) & (areas <= max_area)
    contours = list(it.compress(contours, ok_areas))
    areas = areas[ok_areas]

    # Eliminate contours that are not well approximated by a rect.
    rects = [cv2.boxPoints(cv2.minAreaRect(c)).astype(int) for c in contours]
    rect_areas = np.array([cv2.contourArea(r) for r in rects])
    ok_approx = (rect_areas - areas) / rect_areas <= tol_rect_error
    rects = list(it.compress(rects, ok_approx))
    rect_areas = rect_areas[ok_approx]

    # Eliminate rects that are not approximately square.
    aspects = np.array([rect_aspect_ratio(r) for r in rects])
    ok_aspects = np.abs(1.0 - aspects) <= tol_aspect_ratio
    rects = list(it.compress(rects, ok_aspects))
    rect_areas = rect_areas[ok_aspects]

    # Sort by area (largest to smallest) and get n_squares with the most
    # similar areas.
    if len(rects) > n_squares:
        ix = np.argsort(-rect_areas)
        rect_areas = rect_areas[ix]
        rects = [rects[i] for i in ix]

        if n_squares > 1:
            ix, rect_areas = find_k_similar(rect_areas, n_squares)
            rects = rects[ix:]

    return rects[:n_squares], rect_areas


def rect_aspect_ratio(rect):
    """Compute the aspect ratio of a rectangle.
    """
    # Compute distance from one corner to all others.
    dists = np.linalg.norm(rect[0, :] - rect[1:, :], axis = 1)
    # Aspect ratio is ratio of two shortest distances.
    dists.sort()
    return dists[0] / dists[1]


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
