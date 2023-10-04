"""Entry point for running the fly detection package on a data set.

Currently, this module only runs step 1 of these steps:

1. Orient and crop the fly images via registration marks.
2. Detect the flies.
3. Estimate the temperatures of the detected flies.
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import cli
from . import detect
from . import io
from . import match
from . import ops
from . import registration as reg
from . import train
from . import viz


def register_arenas(config):
    """Register the fly arenas.
    """
    # TODO:
    #   + Expose mask_hsv parameters
    config = config["register"]

    # Read the data set.
    data_dir = Path(config["data_path"])
    print(f"Data set directory: '{data_dir}'")
    if not data_dir.is_dir():
        raise IOError(f"'{data_dir}' is not a directory.")

    dset = io.FlyDatasetReader(data_dir)
    print(f"  Found {len(dset)} images.\n")

    # Make sure the output directory exists.
    output_dir = config.get(
        "output_path", Path("outputs") / data_dir.name / "photos")
    cli.check_dir_exists_empty(output_dir, "Output directory")

    is_debug = config.get("debug", False)
    if is_debug:
        debug_dir = output_dir / "debug"
        cli.check_dir_exists_empty(debug_dir, "Debug directory")

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

        if is_debug:
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
        output_path = output_dir / path.name
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img)
        print(f"Wrote '{output_path}'.")


def match_flies(config):
    """Apply template matching to fly images.
    """
    # TODO:
    #   + Scale templates according to size of image
    #   + Expose match.extract parameters

    apparatuses = config["apparatuses"]
    config = config["register"]

    # Read the data set.
    data_dir = Path(config["data_path"])
    print(f"Data set directory: '{data_dir}'")
    if not data_dir.is_dir():
        raise IOError(f"'{data_dir}' is not a directory.")

    dset = io.FlyDatasetReader(data_dir)
    print(f"  Found {len(dset)} images.\n")

    # Make sure the output directory exists.
    output_dir = config.get(
        "output_path", Path("outputs") / data_dir.name / "labels")
    cli.check_output_dir_exists_empty(output_dir, "Output directory")

    is_debug = config.get("debug", False)
    if is_debug:
        debug_dir = output_dir / "debug"
        cli.check_dir_exists_empty(debug_dir, "Debug directory")

    # Load the templates.
    template_path = Path("outputs/fly_template.npz")
    #template_path = Path("outputs/test_rotations.npz")
    print(f"Template path: '{template_path}'")
    templates, template_shape = io.read_fly_template(template_path)
    print(f"  Found {len(templates)} template images.\n")

    # Resize templates based on relative arena size.
    arena_w_cm = apparatuses[dset.apparatus]["horizontal"]
    # FIXME:
    arena_w_px = cv2.imread(str(dset[0])).shape[1]
    arena_scale = arena_w_px / arena_w_cm
    template_scale = 4280 / 28.6
    scale = arena_scale / template_scale
    width = int(round(templates[0].shape[1] * scale))
    height = int(round(templates[0].shape[0] * scale))
    dim = (width, height)

    c = 0
    for i in templates:
        templates[c] = cv2.resize(i, dim, interpolation = cv2.INTER_AREA)
        c += 1

    # Process each image.
    print("Processing images...\n")

    for path, img in dset.iter_read():
        print(f"Processing '{path}'.")
        # Run template matching.
        boxes, scores = match.concatenate(
            [match.extract(img, t, verbose = is_debug) for t in templates])
        boxes = match.suppress_nonmax(boxes, scores)
        print(f"  Found {boxes.shape[0]} boxes.")

        if is_debug:
            # Output a diagnostic image.
            ax = viz.plot_image(img, figsize = (15, 15))

            for i in range(boxes.shape[0]):
                viz.plot_box(boxes[i, :], ax)
            debug_path = debug_dir / path.name
            plt.savefig(debug_path, bbox_inches = "tight")
            print(f"  Wrote '{debug_path}'.")

        # Save to a YOLO file.
        output_path = output_dir / (str(path.stem) + ".txt")
        io.write_yolo(output_path, boxes, None, img.shape)
        print(f"  Wrote '{output_path}'.\n")

    fly_label = f"{output_dir}/labels.txt"
    with open(fly_label, mode = "wt") as file:
        file.write("fly")


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

    detect_parser = subparsers.add_parser(
        "detect", help = "apply a fly detection model to  a dataset")
    detect_parser.set_defaults(func = detect.main)

    match_parser = subparsers.add_parser(
        "match", help = "apply template matching to data set")
    match_parser.set_defaults(func = match_flies)

    assemble_parser = subparsers.add_parser(
        "assemble", help = "assemble a YOLO object detection training set "
        "from multiple annotated data sets")
    assemble_parser.set_defaults(func = train.assemble_training_set)

    # Add `config` argument to all subparsers.
    for name, subparser in subparsers.choices.items():
        subparser.add_argument(
            "config", type = Path, help ="path to config file")

    args = parser.parse_args()

    # Read the config file.
    print(f"Config path: '{args.config}'")
    config = io.read_config(args.config)

    args.func(config)


if __name__ == "__main__":
    main()
