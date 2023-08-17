"""Entry point for running the fly detection package on a data set.

Currently, this module only runs step 1 of these steps:

1. Orient and crop the fly images via registration marks.
2. Detect the flies.
3. Estimate the temperatures of the detected flies.
"""

import cv2
#import matplotlib.pyplot as plt
import numpy as np

from .io import FlyDatasetReader
from . import ops
from . import registration as reg
#from . import viz

from argparse import ArgumentParser
from pathlib import Path
import sys


def register_arena(args):
    """Register the fly arenas.
    """
    # Read the data set.
    path = args.data
    if not path.is_dir():
        raise IOError(f"'{path}' is not a valid path to a directory")

    dset = FlyDatasetReader(path)
    print(f"Found {len(dset)} images in data set '{path}'.")

    # Find the registration marks.
    arenas = []
    images = []
    for path, img in dset.iter_read():
        print(path)

        # FIXME: Standardize brightness across images.

        green_mask = reg.mask_hsv(
            img, (60, 50, 79), (120, 255, 255), close_kernel = 11)
        green_squares, _ = reg.compute_squares(green_mask, 3)

        orange_mask = reg.mask_hsv(
            img, (5, 79, 127), (20, 255, 255), close_kernel = 11)
        orange_square, _ = reg.compute_squares(orange_mask, 1)

        # preview = cv2.drawContours(
        #     img.copy(), green_squares, -1, (0, 255, 0), 3)
        # preview = cv2.drawContours(
        #     preview, orange_square, -1, (255, 0, 0), 3)
        #
        # out_path = Path("output/test/") / dset.fly_paths[i].name
        # cv2.imwrite(str(out_path), preview)
        # print(f"Wrote '{out_path}'.")

        # Get orientation and arena bounds.
        orient, arena = reg.orient_and_bound(orange_square, green_squares)
        print(f"{orient=}\n{arena=}")
        arenas.append(arena.ravel())

        # Orient the image.
        if orient:
            img = cv2.rotate(img, orient)
        images.append(img)

    # Compute median arena position. No need to convert to int since the
    # perspective transformation requires float inputs.
    arenas = np.stack(arenas)
    m_arena = np.median(arenas, axis = 0).reshape(-1, 2)
    print(f"{m_arena=}")

    transform, width, height = ops.get_perspective_transform(m_arena)
    print(f"{width=}, {height=}")

    # Make sure the output directory exists.
    out_dir = args.out
    print(f"Output directory: {out_dir}")
    out_dir.mkdir(parents = True, exist_ok = True)
    if next(out_dir.iterdir(), None):
        msg = (f"Directory '{out_dir}' contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)

    # Extract and save the arenas.
    for path, img in zip(dset, images):
        img = cv2.warpPerspective(
            img, transform, (width, height), flags = cv2.INTER_CUBIC)

        print(f"{path=}, {img.shape}")

        # Save the new image (or save metadata about the orientation and
        # perspective).
        out_path = out_dir / path.name
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), img)
        print(f"Wrote '{out_path}'.")


def prompt_yes(prompt):
    """Prompt the user with a yes or no question and return True if they
    respond yes.
    """
    while True:
        match input(prompt).lower():
            case "y" | "yes":
                return True
            case "n" | "no":
                return False


def main():
    # Parse command line arguments.
    parser = ArgumentParser()
    parser.add_argument(
        "data", type = Path, help = "path to the data set directory")
    parser.add_argument(
        "out", type = Path, help = "path to directory to save output"
        , default = Path("output/test/"), nargs = "?")
    args = parser.parse_args()

    # Read config file if there is one.

    register_arena(args)


if __name__ == "__main__":
    main()
