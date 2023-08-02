"""This module handles reading and writing data.
"""

import cv2

from .ops import rescale

from pathlib import Path


def read_image(path, max_size = 0, grayscale = False):
    """Read a single image.
    """
    path = str(path)
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        # OpenCV defaults to BGR for color images, but other tools expect RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if max_size > 0:
        img = rescale(img, max_size)

    return img


class FlyDatasetReader:
    def __init__(
        self, data_dir, glob = "*", suffixes = [".jpg", ".jpeg"], max_size = 0
        , blank_pattern = "blank"
    ):
        """Create a new FlyDataSetReader.

        Arguments
        ---------
        data_dir: str or pathlib.Path
            Path to the data set directory. The data set directory should
            contain a directory named "photos/" which contains the images.

        glob: str
            Case-sensitive glob string to filter files in the `photos/`
            directory.

        suffixes: list of str
            Case-insensitive filename suffixes to further filter files in the
            `photos/` directory *after* globbing. These should include the dot,
            as in `.jpg`.

        max_size: int
            Maximum side length for images. Images larger than this will be
            rescaled so that the longest side has this length. If this is not
            positive (the default), images will not be rescaled.

        blank_pattern: str
            Case-insensitive pattern for "blank" no-flies images.
        """
        # Make sure there's a `photos/` directory.
        photos_dir = Path(data_dir) / "photos"
        if not photos_dir.is_dir():
            raise FileNotFoundError(
                f"Directory '{photos_dir}' does not exist.")

        # Get all images.
        suffixes = [s.lower() for s in suffixes]
        blank_paths = []
        fly_paths = []
        for p in photos_dir.glob(glob):
            # Ignore files with incorrect suffixes.
            if p.suffix.lower() not in suffixes:
                continue

            # Check whether filename contains pattern for "blank" no flies
            # images.
            if blank_pattern and blank_pattern in p.stem.lower():
                blank_paths.append(p)
            else:
                fly_paths.append(p)

        self.blank_paths = sorted(blank_paths)
        self.fly_paths = sorted(fly_paths)
        self.max_size = max_size

    def __len__(self):
        return len(self.fly_paths)

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
