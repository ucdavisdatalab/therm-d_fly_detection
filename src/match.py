"""This module contains functions for template matching.
"""

import cv2
import numpy as np


def get_bounds(loc, template):
    l, t = loc
    h, w = template.shape[:2]
    return t, l, t + h, l + w


def extract_match(image, template, mode = cv2.TM_SQDIFF):
    """Extract the coordinates of a template match.
    """
    dists = cv2.matchTemplate(image, template, mode)
    _, _, min_loc, max_loc = cv2.minMaxLoc(dists)

    if mode == cv2.TM_SQDIFF or mode == cv2.TM_SQDIFF_NORMED:
        loc = min_loc
    else:
        loc = max_loc

    return get_bounds(loc, template)


def concatenate(matches):
    """Concatenate matches from the extract function.

    Arguments
    ---------
    matches: list
        A list of matches from the extract function.
    """
    return [np.concatenate(x) for x in zip(*matches)]


def extract(
    image, template, threshold = 0.95, n_max = 200, adaptive = False
    , metric = cv2.TM_CCOEFF_NORMED, nhood_scale = 0.5, verbose = False
):
    """Extract multiple matches of a template from an image.

    Arguments
    ---------
    image: numpy.ndarray
        The image from which to extract matches.

    template: numpy.ndarray
        The template to search for in the image.

    threshold: float
        The similarity threshold for matches.

    n_max: int
        The maximum number of matches to return.

    adaptive: bool
        Interpret the threshold as a percentile rather than a similarity score?

    metric: int
        Similarity metric to use for template matching. For example,
        cv2.TM_CCOEFF.

    nhood_scale: float
        Scaling factor for neighborhood to set to -1 around matched points.

    verbose: bool
        Print diagnostic information?

    Returns
    -------
    out: tuple
        A 2-tuple that contains the coordinates and scores for the
        highest-similarity boxes. The coordinates are a matrix with columns
        top, left, bottom, right.
    """
    # Compute similarity matrix.
    similarity = cv2.matchTemplate(image, template, metric)
    if metric in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        similarity = -similarity

    # Shift similarity to be in [0, 1].
    similarity -= np.min(similarity)
    similarity /= np.max(similarity)

    if adaptive:
        threshold = np.quantile(similarity, threshold)

    if verbose:
        print(f"min: {np.min(similarity):.4f} "
              f"median: {np.median(similarity):.4f} "
              f"99.9th: {np.quantile(similarity, 0.999):.4f} "
              f"max: {np.max(similarity):.4f} "
              f"threshold: {threshold:.4f}")

    # Compute neighborhood radius as 1/2 of average template side length.
    h, w = template.shape[:2]
    nhood = np.array([-0.5, 0.5]) * (h + w) * nhood_scale
    nhood = np.rint(nhood).astype(int)

    # Search for matches until n_max are found or best match is below
    # threshold.
    locs = np.zeros((n_max, 4), dtype = np.int32)
    scores = np.zeros(n_max)
    for i in range(n_max):
        # Find location (top left corner) of current best match.
        _, score, _, (x, y) = cv2.minMaxLoc(similarity)

        # Stop if highest score is below threshold.
        if score < threshold:
            if verbose:
                print(f"After {i} iterations, hit threshold with "
                      f"score {score:.4f}.")
            i -= 1  # no box added on this final iteration
            break

        scores[i] = score
        locs[i, :] = (y, x, y + h, x + w)
        # Set scores in the location's neighborhood to -1.
        x_nhood = np.clip(x + nhood, 0, similarity.shape[1])
        y_nhood = np.clip(y + nhood, 0, similarity.shape[0])
        similarity[slice(*y_nhood), slice(*x_nhood)] = -1

    return locs[:i + 1, :], scores[:i + 1]


def suppress_nonmax(boxes, scores, threshold = 0.6):
    """Remove redundant bounding boxes by keeping only the highest-scoring box
    in each group of boxes with excessive overlap.

    Caution: this function assumes all boxes have the same area!

    Arguments
    ---------
    boxes: np.ndarray
        A matrix where each row is one box and the columns are the coordinates
        for top, left, bottom, right.

    scores: np.ndarray
        An array of similarity scores, with one element for each row in
        `boxes`.

    threshold: float
        Maximum intersection-over-union (IoU) for a box to be kept.
    """
    # See:
    #    https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    #
    # Compute box area for IoU calculation.
    box_area = (boxes[0, 2] - boxes[0, 0]) * (boxes[0, 3] - boxes[0, 1])

    # Process the boxes until no boxes remain.
    ix_remaining = np.argsort(scores)  # smallest to largest
    kept = np.zeros_like(boxes)
    n_kept = 0
    while len(ix_remaining) > 0:
        # Of the remaining boxes, get the one with the highest similarity
        # score. This is the "best" remaining box.
        ix_best = ix_remaining[-1]
        best = boxes[ix_best, :]
        ix_remaining = ix_remaining[:-1]
        # Keep the best box.
        kept[n_kept] = best
        n_kept += 1

        # Compute intersection-over-union of best box with all remaining boxes.
        ious = _equal_area_box_iou(best, boxes[ix_remaining, :], box_area)
        # Only keep boxes with low IoU.
        ix_remaining = ix_remaining[ious < threshold]

    # Only n_boxes kept.
    return kept[:n_kept]


def _equal_area_box_iou(box, boxes, box_area):
    """Compute the intersection-over-union for one box against many boxes, with
    the assumption that every box has the same area.
    """
    # Compute intersection area.
    h = np.fmin(box[2], boxes[:, 2]) - np.fmax(box[0], boxes[:, 0])
    w = np.fmin(box[3], boxes[:, 3]) - np.fmax(box[1], boxes[:, 1])
    intersect_area = np.fmax(0, h) * np.fmax(0, w)
    # Boxes have same size, so denominator (below) is union area.
    return intersect_area / (2 * box_area - intersect_area)
