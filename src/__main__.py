"""Entry point for running the fly detection package on a data set.

Currently, this module only runs step 1 of these steps:

1. Orient and crop the fly images via registration marks.
2. Detect the flies.
3. Estimate the temperatures of the detected flies.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .io import FlyDatasetReader
from . import io
from . import match
from . import ops
from . import registration as reg
from . import viz

from argparse import ArgumentParser
from pathlib import Path
import sys


def register_arenas(args):
    """Register the fly arenas.
    """
    # TODO:
    #   + Standardize image brightness
    #   + Expose mask_hsv parameters

    # Make sure the output directory exists.
    out_dir = args.out
    print(f"Output directory: {out_dir}")
    out_dir.mkdir(parents = True, exist_ok = True)
    if next(out_dir.iterdir(), None):
        msg = (f"Directory '{out_dir}' contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)

    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok = True)

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

        # Standardize brightness across images.
        img = ops.gamma_transform(img, gamma_quantile = 0.05)

        green_mask = reg.mask_hsv(
            img, (60, 50, 79), (120, 255, 255), close_kernel = 11)
        green_squares, _ = reg.compute_squares(green_mask, 3)

        orange_mask = reg.mask_hsv(
            img, (5, 79, 95), (20, 255, 255), close_kernel = 11)
        orange_square, _ = reg.compute_squares(orange_mask, 1)

        if args.debug:
            # Save a diagnostic image.
            preview = cv2.drawContours(
                img.copy(), green_squares, -1, (0, 255, 0), 3)
            preview = cv2.drawContours(
                preview, orange_square, -1, (255, 0, 0), 3)

            debug_path = debug_dir / path.name
            cv2.imwrite(str(debug_path), preview)
            print(f"Wrote '{debug_path}'.")

        # Get orientation and arena bounds.
        orient, arena = reg.orient_and_bound(orange_square, green_squares)

        # Orient the image and the arena.
        if orient is not None:
            img = cv2.rotate(img, orient)
            arena = ops.rotate_contour(arena, img.shape, orient)

        print(f"{orient=}\n{arena=}")

        images.append(img)
        arenas.append(arena)

    # Compute median arena position. No need to convert to int since the
    # perspective transformation requires float inputs.
    arenas = np.stack(arenas)
    m_arena = np.median(arenas, axis = 0).reshape(-1, 2)
    print(f"{m_arena=}")

    transform, width, height = ops.get_perspective_transform(m_arena)
    print(f"{width=}, {height=}")

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


def match_flies(args):
    """Apply template matching to fly images.
    """
    # TODO:
    #   + Scale templates according to size of image
    #   + Standardize image brightness
    #   + Expose match.extract parameters

    # Make sure the output directory exists.
    out_dir = args.out
    print(f"Output directory: {out_dir}")
    out_dir.mkdir(parents = True, exist_ok = True)
    if next(out_dir.iterdir(), None):
        msg = (f"Directory '{out_dir}' contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)

    debug_dir = out_dir / "debug"
    debug_dir.mkdir(exist_ok = True)

    # Read the data set.
    dset_path = args.data
    if not dset_path.is_dir():
        raise IOError(f"'{dset_path}' is not a valid path to a directory")

    dset = FlyDatasetReader(dset_path)
    print(f"Found {len(dset)} images in data set '{dset_path}'.")

    # Load the templates.
    template_path = Path("outputs/fly_template.npz")
    templates, template_shape = io.read_fly_template(template_path)
    print(f"Found {len(templates)} templates at '{template_path}'.")

    # TODO: Resize templates based on relative arena size. This requires info
    # about the physical size of the arena.

    # Process each image.
    for path, img in dset.iter_read():
        print(f"\nProcessing '{path}'.")
        # Run template matching.
        boxes, scores = match.concatenate(
            [match.extract(img, t, verbose = args.debug) for t in templates])
        boxes = match.suppress_nonmax(boxes, scores)
        print(f"Found {boxes.shape[0]} boxes.")

        if args.debug:
            # Output a diagnostic image.
            ax = viz.plot_image(img, figsize = (15, 15))

            for i in range(boxes.shape[0]):
                viz.plot_box(boxes[i, :], ax)
            debug_path = debug_dir / path.name
            plt.savefig(debug_path, bbox_inches = "tight")
            print(f"Wrote '{debug_path}'.")

        # Save to a YOLO file.
        out_path = out_dir / (str(path.stem) + ".txt")
        io.write_yolo(out_path, boxes, None, img.shape)
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
        "--debug", action = "store_true"
        , help = "output diagnostic information")

    subparsers = parser.add_subparsers(title = "subcommands")

    register_parser = subparsers.add_parser(
        "register", help = "(step 1) register arenas in data set")
    register_parser.set_defaults(func = register_arenas)

    match_parser = subparsers.add_parser(
        "match", help = "apply template matching to data set")
    match_parser.set_defaults(func = match_flies)

    for name, subparser in subparsers.choices.items():
        subparser.add_argument(
            "data", type = Path, help = "path to the data set directory")
        subparser.add_argument(
            "out", type = Path, help = "path to directory to save output"
            , default = Path("outputs/test/"), nargs = "?")

    args = parser.parse_args()

    # TODO: Read config file if there is one.

    args.func(args)


if __name__ == "__main__":
    main()
