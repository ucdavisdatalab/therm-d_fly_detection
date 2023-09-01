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
    #   + Expose mask_hsv parameters

    # Read the data set.
    data_dir = args.data
    print(f"Data set directory: '{data_dir}'")
    if not data_dir.is_dir():
        raise IOError(f"'{data_dir}' is not a directory.")

    dset = FlyDatasetReader(data_dir)
    print(f"  Found {len(dset)} images.\n")

    # Make sure the output directory exists.
    out_dir = args.out
    if out_dir is None:
        out_dir = Path("outputs") / data_dir.name / "photos"
    print(f"Output directory: '{out_dir}'")

    out_dir.mkdir(parents = True, exist_ok = True)
    if next(out_dir.iterdir(), None):
        msg = ("Output directory contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)
    print()

    if args.debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok = True)
        print(f"Debug directory: '{debug_dir}'\n")

    # Find the registration marks.
    print("Finding registration marks...\n")

    arenas = []
    images = []
    for path, img in dset.iter_read():
        print(path)

        # Standardize brightness across images.
        adj_img = ops.adaptive_gamma_correction(img)

        hsv = cv2.cvtColor(adj_img, cv2.COLOR_RGB2HSV)[:, :, 1:]
        qv = np.quantile(hsv[:, :, 1].ravel(), 0.45)

        green_mask = reg.mask_hsv(
            adj_img, (70, 63, 95), (85, 255, 255), close_kernel = 21
            , open_kernel = 7)
        green_squares, _ = reg.compute_squares(
            green_mask, 3, tol_aspect_ratio = 0.25, tol_rect_error = 0.2)
        if len(green_squares) < 3:
            raise RuntimeError("Could not find 3 green registration marks"
                               f" ({len(green_squares)} found).")

        orange_mask = reg.mask_hsv(
            adj_img, (0, 95, qv), (15, 255, 255), close_kernel = 21
            , open_kernel = 3)
        orange_square, _ = reg.compute_squares(
            orange_mask, 1, tol_aspect_ratio = 0.25, tol_rect_error = 0.2)
        if len(orange_square) != 1:
            raise RuntimeError("Could not find orange registration mark.")

        if args.debug:
            # Save a diagnostic image.
            preview = cv2.drawContours(
                adj_img.copy(), green_squares, -1, (0, 255, 0), 3)
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

        print(f"{orient=}\n{arena=}\n")

        images.append(img)
        arenas.append(arena)

    # Compute median arena position. No need to convert to int since the
    # perspective transformation requires float inputs.
    print("Computing median arena boundary...\n")

    arenas = np.stack(arenas)
    m_arena = np.median(arenas, axis = 0).reshape(-1, 2)
    print(f"{m_arena=}")

    transform, width, height = ops.get_perspective_transform(m_arena)
    print(f"{width=}, {height=}\n")

    # Extract and save the arenas.
    print("Extracting and saving arenas...\n")

    for path, img in zip(dset, images):
        img = cv2.warpPerspective(
            img, transform, (width, height), flags = cv2.INTER_CUBIC)

        #print(f"{path=}, {img.shape}")
        img = ops.adaptive_gamma_correction(img)

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
    #   + Expose match.extract parameters

    # Read the data set.
    data_dir = args.data
    print(f"Data set directory: '{data_dir}'")
    if not data_dir.is_dir():
        raise IOError(f"'{data_dir}' is not a directory.")

    dset = FlyDatasetReader(data_dir)
    print(f"  Found {len(dset)} images.\n")

    # Make sure the output directory exists.
    out_dir = args.out
    if out_dir is None:
        out_dir = Path("outputs") / data_dir.name / "labels"
    print(f"Output directory: '{out_dir}'")

    out_dir.mkdir(parents = True, exist_ok = True)
    if next(out_dir.iterdir(), None):
        msg = ("Output directory contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not prompt_yes(msg):
            sys.exit(1)
    print()

    if args.debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(exist_ok = True)
        print(f"Debug directory: '{debug_dir}'\n")

    # Load the templates.
    template_path = Path("outputs/fly_template.npz")
    print(f"Template path: '{template_path}'")
    templates, template_shape = io.read_fly_template(template_path)
    print(f"  Found {len(templates)} template images.\n")

    # TODO: Resize templates based on relative arena size. This requires info
    # about the physical size of the arena.

    # Process each image.
    print("Processing images...\n")

    for path, img in dset.iter_read():
        print(f"Processing '{path}'.")
        # Run template matching.
        boxes, scores = match.concatenate(
            [match.extract(img, t, verbose = args.debug) for t in templates])
        boxes = match.suppress_nonmax(boxes, scores)
        print(f"  Found {boxes.shape[0]} boxes.")

        if args.debug:
            # Output a diagnostic image.
            ax = viz.plot_image(img, figsize = (15, 15))

            for i in range(boxes.shape[0]):
                viz.plot_box(boxes[i, :], ax)
            debug_path = debug_dir / path.name
            plt.savefig(debug_path, bbox_inches = "tight")
            print(f"  Wrote '{debug_path}'.")

        # Save to a YOLO file.
        out_path = out_dir / (str(path.stem) + ".txt")
        io.write_yolo(out_path, boxes, None, img.shape)
        print(f"  Wrote '{out_path}'.\n")

    fly_label = f'{out_dir}/labels.txt'
    with open(fly_label, mode = 'wt') as file:
        file.write('fly')
    

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
            , nargs = "?")

    args = parser.parse_args()

    # TODO: Read config file if there is one.

    args.func(args)


if __name__ == "__main__":
    main()
