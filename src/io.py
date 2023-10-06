"""This module handles reading and writing data.
"""

import cv2
import numpy as np
import pandas as pd
import tomllib

from .ops import rescale

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
        data_dir = Path(data_dir)

        # Make sure there's a `photos/` directory.
        photos_dir = data_dir / "photos"
        if not photos_dir.is_dir():
            raise FileNotFoundError(
                f"Directory '{photos_dir}' does not exist.")

        self.max_size = max_size

        # Determine the apparatus.
        name_parts = data_dir.name.rsplit("_", 1)
        if len(name_parts) != 2:
            warnings.warn(
                f"Data directory name '{data_dir.name}' does not have format"
                " YYYY-MM-DD_APPARATUS.")
        self.apparatus = name_parts[-1].lower()

        # Locate photos.
        suffixes = [s.lower() for s in suffixes]
        blank_paths = []
        paths = []
        for p in photos_dir.glob(glob):
            # Ignore files with incorrect suffixes.
            if p.suffix.lower() not in suffixes:
                continue
            # Check whether filename contains "blank" (a no flies image).
            if blank_pattern and blank_pattern in p.stem.lower():
                blank_paths.append(p)
            else:
                paths.append(p)

        self.blank_paths = sorted(blank_paths)
        self.paths = sorted(paths)

        # Locate arenas.
        arenas_dir = data_dir / "arenas"
        arena_paths = []
        if not arenas_dir.is_dir():
            warnings.warn(f"Directory '{arenas_dir}' does not exist.")
        else:
            image_stems = [p.stem for p in paths]
            for p in arenas_dir.iterdir():
                if p.suffix.lower() not in suffixes:
                    continue
                # Check that the arena corresponds to a photo.
                if p.stem not in image_stems:
                    warnings.warn(f"No photo for arena file '{p}'.")
                    continue
                arena_paths.append(p)

        self.arena_paths = sorted(arena_paths)

        # Locate labels.
        labels_dir = data_dir / "labels"
        label_paths = []
        if not labels_dir.is_dir():
            warnings.warn(f"Directory '{labels_dir}' does not exist.")
        else:
            image_stems = [p.stem for p in paths]
            for p in labels_dir.iterdir():
                if p.suffix.lower() not in (".txt"):
                    continue
                # Check that the label corresponds to a photo.
                if p.stem not in image_stems:
                    warnings.warn(f"No photo for label file '{p}'.")
                    continue
                label_paths.append(p)

        self.label_paths = sorted(label_paths)

        # Locate the Excel file.
        self.excel_paths = [
            p for p in data_dir.iterdir()
            if p.suffix.lower() in ('.xlsx', '.xls')]
        if len(self.excel_paths) == 0:
            warnings.warn("No Excel file in this data set.")

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
        for i in range(len(self)):
            yield self.read(i, **kwargs)

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
        if len(self) == 0:
            raise RuntimeError("No photos in this data set.")

        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size
        path = self.paths[index]
        return path, read_image(path, **kwargs)

    def iter_read_arenas(self, **kwargs):
        """Iterate over the data set, reading arena images and yielding
        (path, image) pairs.

        Yields
        ------
        path: pathlib.Path
            The path to the image.

        image: numpy.ndarray
            The image in RGB format.
        """
        arena_paths = self.arena_paths
        n_arenas = len(arena_paths)
        if n_arenas == 0:
            raise RuntimeError("No arenas in this data set.")

        if "max_size" not in kwargs:
            kwargs["max_size"] = self.max_size

        for i in range(n_arenas):
            path = arena_paths[i]
            yield path, read_image(path, **kwargs)

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

    def read_sheet_temperatures(self, index = 0):
        """Read temperatures from an Excel file.

        Arguments
        ---------
        index: int
            An index into the list of Excel files.

        Returns
        -------
        """
        if len(self.excel_paths) == 0:
            raise RuntimeError("No Excel file in this data set.")

        excel_path = self.excel_paths[index]
        df = pd.read_excel(excel_path)
        ind = df[df.iloc[:, 0] == 'temperature probe'].index[0]
        df = df[ind:ind + 3].dropna(axis = 1)
        df.columns = df.iloc[0]
        df = df[1:3]
        df = df.set_index('temperature probe')
        df = df.to_numpy()
        self.temperatures = df
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


def read_fly_template(path):
    """Read a collection of saved fly templates.

    Arguments
    ---------
    path: str or Path
        Path to an `.npz` file.

    Returns
    -------
    out: tuple
        A 2-tuple that contains a list of templates and the dimensions of the
        image from which they were cropped.
    """
    npz = np.load(path)
    shape = npz.get("shape")
    templates = [v for k, v in npz.items() if k != "shape"]
    return templates, shape


def read_config(path, populate_data_path = ["register", "detect", "match"]):
    """Read a TOML configuration file.

    Arguments
    ---------
    path: str or Path
        Path to TOML configuration file.

    populate_data_path: list of str
        List of keys for subconfigurations (dicts) to which the top-level
        `data_path` entry should be propagated if no `data_path` entry is
        present.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)

    for key in populate_data_path:
        if key in config and "data_path" not in config[key]:
            config[key]["data_path"] = config["data_path"]

    return config


def get_distance(dict, name):
    return dict['distance'][name]
