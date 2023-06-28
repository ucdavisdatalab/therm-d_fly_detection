"""This module contains functions for template matching.
"""

import cv2


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
