"""This module handles reading and writing data.
"""

import numpy as np
from PIL import Image

from pathlib import Path


def rescale_image(image, max_size):
    """Rescale an image to a maximum size, preserving the aspect ratio.
    """
    # Determine which dimension to resize.
    w, h = image.size
    scaling = max_size / max(w, h)
    w = int(round(w * scaling))
    h = int(round(h * scaling))
    return image.resize((w, h))


def read_image(path, max_size = 0, grayscale = False):
    """Read a single image.
    """
    img = Image.open(path)

    # Default to RGB if grayscale is not requested.
    if grayscale and img.mode != "L":
        img = img.convert("L")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if max_size > 0:
        img = rescale_image(img, max_size)

    return np.asarray(img)


def image_set_reader(
    img_dir, blank_glob = "blank.JPG", fly_glob = "IMG*.JPG",
    verbose = False, **kwargs
):
    """Read and yield images from a set of images.
    """
    img_dir = Path(img_dir)

    # Find the blank image.
    blank_path = next(img_dir.glob(blank_glob))
    if verbose:
        print(f"Found blank image '{blank_path}'")

    # Find the other images.
    fly_paths = sorted(img_dir.glob(fly_glob))
    if verbose:
        print(f"Found {len(fly_paths)} fly images")

    # Read the images.
    blank_img = read_image(blank_path, **kwargs)
    if verbose:
        print(f"Read '{blank_path}'")
    yield blank_img

    for p in fly_paths:
        fly_img = read_image(p, **kwargs)
        if verbose:
            print(f"Read '{p}'")
        yield fly_img
