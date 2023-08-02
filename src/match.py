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
