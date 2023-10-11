"""Functions for detecting registration marks.
"""

import itertools as it

import cv2
import numpy as np

from . import ops


def detect_arenas(image_set, config, debug_dir):
    """Detect arenas in images based on registration marks.

    Arguments
    ---------
    image_set: list of tuples
        Images in which to detect arenas, as (path, image) pairs.

    config: dict
        Configuration, as read from a TOML file.

    debug_dir: Path or None
        Path to debug directory if in debug mode, or None if not in debug mode.

    Yields
    ------
    out: tuple
        (arena, rotation) pairs; arena is a matrix with columns x, y and
        rows top left, top right, bottom right, bottom left; rotation is an
        OpenCV rotation code.
    """
    # Find the registration marks.
    print("Finding registration marks...\n")

    # TODO: Expose these in TOML file.
    green_hsv_lower = np.array([70, 63, 95], np.uint8)
    green_v_quant = None
    green_hsv_upper = np.array([85, 255, 255], np.uint8)
    green_close = 21
    green_open = 7
    green_tol_aspect_ratio = 0.25
    green_tol_rect_error = 0.2

    orange_hsv_lower = np.array([0, 95, 0], np.uint8)
    orange_v_quant = 0.45
    orange_hsv_upper = np.array([15, 255, 255], np.uint8)
    orange_close = 21
    orange_open = 3
    orange_tol_aspect_ratio = 0.25
    orange_tol_rect_error = 0.2

    for path, img in image_set:
        print(f"Image: '{path}'")

        # Standardize brightness across images.
        adj_img = ops.adaptive_gamma_correction(img)

        hsv = cv2.cvtColor(adj_img, cv2.COLOR_RGB2HSV)[:, :, 1:]
        qv = np.quantile(hsv[:, :, 1].ravel(), orange_v_quant)
        orange_hsv_lower[2] = qv

        green_mask = mask_hsv(
            adj_img, green_hsv_lower, green_hsv_upper
            , close_kernel = green_close, open_kernel = green_open)
        green_squares, _ = compute_squares(
            green_mask, 3
            , tol_aspect_ratio = green_tol_aspect_ratio
            , tol_rect_error = green_tol_rect_error)
        if len(green_squares) < 3:
            raise RuntimeError("Could not find 3 green registration marks"
                               f" ({len(green_squares)} found).")

        orange_mask = mask_hsv(
            adj_img, orange_hsv_lower, orange_hsv_upper
            , close_kernel = orange_close, open_kernel = orange_open)
        orange_square, _ = compute_squares(
            orange_mask, 1
            , tol_aspect_ratio = orange_tol_aspect_ratio
            , tol_rect_error = orange_tol_rect_error)
        if len(orange_square) != 1:
            raise RuntimeError("Could not find orange registration mark.")

        if debug_dir is not None:
            # Save a diagnostic image.
            preview = cv2.drawContours(
                adj_img.copy(), green_squares, -1, (0, 255, 0), 3)
            preview = cv2.drawContours(
                preview, orange_square, -1, (255, 0, 0), 3)

            debug_path = debug_dir / path.name
            cv2.imwrite(str(debug_path), preview)
            print(f"  Wrote '{debug_path}'.")

        # Get rotation and arena bounds.
        rotation, arena = orient_and_bound(orange_square, green_squares)

        print(f"  {rotation=}\n  {arena=}\n")

        yield arena, rotation


def get_arenas(data, config):
    """Get arena and rotation info from a configuration.

    Arguments
    ---------
    image_set: list of tuples
        Images in which to detect arenas, as (path, image) pairs.

    config: dict
        Configuration, as read from a TOML file.

    Yields
    ------
    out: tuple
        (arena, rotation) pairs; arena is a matrix with columns x, y and
        rows top left, top right, bottom right, bottom left; rotation is an
        OpenCV rotation code.
    """
    arena_config = config["arena"]
    arena = np.array([
        arena_config["top_left"]
        , arena_config["top_right"]
        , arena_config["bottom_right"]
        , arena_config["bottom_left"]
    ])

    rotation = {
        "none": None
        , "180": cv2.ROTATE_180
        , "90 clockwise": cv2.ROTATE_90_CLOCKWISE
        , "90 counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE
    }[arena_config["rotate"]]

    for _ in data:
        yield arena, rotation


def rotate_images_with_arenas(image_set, arena_set):
    """Rotate images and arena bounds together.

    Arguments
    ---------
    image_set: list of tuples
        Images to rotate, as (path, image) pairs.

    arena_set: list of tuples
        Arenas to rotate, as (arena, rotation) pairs.

    Yields
    ------
    out: tuple
        (image_set, arena) pairs; image_set is a (path, image) pair; arena is a
        matrix with columns x, y and rows top left, top right, bottom right,
        bottom left.
    """
    for (path, image), (arena, rotation) in zip(image_set, arena_set):
        if rotation is not None:
            image = cv2.rotate(image, rotation)
            arena = ops.rotate_contour(arena, image.shape, rotation)
        yield (path, image), arena


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
