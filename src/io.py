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


class FlyDatasetReader:
    def __init__(
        self, data_dir, blank_glob = "blank*.JPG", fly_glob = "IMG*.JPG",
        max_size = 0
    ):
        # Make sure there's a `photos/` directory.
        photos_dir = Path(data_dir) / "photos"
        if not photos_dir.is_dir():
            raise FileNotFoundError(
                f"Directory '{photos_dir}' does not exist.")

        # Find the blank image.
        self.blank_paths = sorted(photos_dir.glob(blank_glob))
        self.fly_paths = sorted(photos_dir.glob(fly_glob))
        self.max_size = max_size

    def read_blank(self, **kwargs):
        n_blank = len(self.blank_paths)
        if n_blank > 1:
            print(f"Found {n_blank} paths, returning first.")

        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        return read_image(self.blank_paths[0], **kwargs)

    def read_fly(self, index, **kwargs):
        p = self.fly_paths[index]

        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        return read_image(p, **kwargs)
