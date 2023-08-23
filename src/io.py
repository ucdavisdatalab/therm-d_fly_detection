"""This module handles reading and writing data.
"""

import cv2
import numpy as np
import pandas as pd

from .ops import rescale

import os
from pathlib import Path
import warnings


def read_image(path, max_size = 0, grayscale = False):
    """Read a single image in RGB or grayscale format.

    Arguments
    ---------
    path: str or pathlib.Path
        The path to the image.

    max_size: int
        Maximum side length for the image. If the image is larger than this, it
        will be rescaled so that the longest side has this length. If this is
        not positive (the default), the image will not be rescaled.

    grayscale: bool
        Convert the image to grayscale?
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
        paths = []
        for p in photos_dir.glob(glob):
            # Ignore files with incorrect suffixes.
            if p.suffix.lower() not in suffixes:
                continue

            # Check whether filename contains pattern for "blank" no flies
            # images.
            if blank_pattern and blank_pattern in p.stem.lower():
                blank_paths.append(p)
            else:
                paths.append(p)

        file = os.listdir(data_dir)
        excel = [p for p in file if p.lower().endswith(('.xlsx', '.xls'))]
        if len(excel) == 0:
            warnings.warn("No Excel file found.")
            self.excel = None
        else:
            if len(excel) > 1:
                warnings.warn(
                    "More than one Excel file found. Using the first.")
            self.excel = data_dir + "/" + excel[0]

        self.blank_paths = sorted(blank_paths)
        self.paths = sorted(paths)
        self.max_size = max_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.paths[index]

    def iter_read(self, **kwargs):
        """Iterate over the data set, reading images and yielding (path, image)
        pairs.

        Yields
        ------
        path: pathlib.Path
            The path to the image.

        image: numpy.ndarray
            The image in RGB format.
        """
        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        for path in self:
            yield path, read_image(path, **kwargs)

    def read(self, index, **kwargs):
        """Read a single image from the data set.

        Arguments
        ---------
        index: int
            An index into the data set (that is, into the `.paths` attribute).

        **kwargs
            Additional arguments to read_image.

        Returns
        -------
        image: numpy.ndarray
            The image in RGB format.
        """
        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        p = self.paths[index]
        return read_image(p, **kwargs)

    def read_blank(self, index = 0, **kwargs):
        """Read a single "blank" no-flies image from the data set.

        Arguments
        ---------
        index: int
            An index into the data set (that is, into the `.paths` attribute).

        **kwargs
            Additional arguments to read_image.

        Returns
        -------
        image: numpy.ndarray
            The image in RGB format.
        """
        if len(self.blank_paths) == 0:
            raise RuntimeError("No blank images in this data set.")

        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        return read_image(self.blank_paths[index], **kwargs)

    def read_sheet_temperatures(self):
        if self.excel is None:
            raise RuntimeError("No Excel file found.")
        df = pd.read_excel(self.excel)
        ind = df[df.iloc[:, 0] == 'temperature probe'].index[0]
        df = df[ind:ind + 3].dropna(axis = 1)
        df.columns = df.iloc[0]
        df = df[1:3]
        df = df.set_index('temperature probe')
        df = df.to_numpy()
        return df


def write_yolo(path, boxes, scores, shape):
    '''Write the bounding boxes in YOLO format for a single image.

    Arguments
    ---------
    path: str or Path
        Path to where the YOLO file should be saved.

    boxes: np.ndarray
        Coordinates of the boxes, where each row is one box and the columns are
        top, left, bottom, right.

    scores: np.ndarray
        Similarity scores of the boxes, or None to skip writing scores.

    shape: np.ndarray
        The dimensions of the image, to standardize the coordinates.
    '''
    ylocs = (boxes[:, 2] + boxes[:, 0]) / (2 * shape[0])
    xlocs = (boxes[:, 3] + boxes[:, 1]) / (2 * shape[1])
    heights = (boxes[:, 2] - boxes[:, 0]) / shape[0]
    widths = (boxes[:, 3] - boxes[:, 1]) / shape[1]

    if scores is None:
        lines = [
            f"0 {x} {y} {w} {h}\n"
            for x, y, w, h in zip(xlocs, ylocs, widths, heights)
        ]
    else:
        lines = [
            f"0 {x} {y} {w} {h} {s}\n"
            for x, y, w, h, s in zip(xlocs, ylocs, widths, heights, scores)
        ]

    with open(path, 'wt') as f:
        f.writelines(lines)
            
