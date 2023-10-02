"""Functions to predict fly bounding boxes.
"""

from pathlib import Path
import sys
import tomllib

import cv2
import numpy as np
import pandas as pd

from . import cli
from . import io
from . import ops
from . import extract_extropolate


def main(args):
    import onnxruntime as ort

    # Read the config file.
    print(f"Config path: '{args.config}'")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    config = config["detect"]

    data_path = Path(config["data_path"])
    print(f"Data path: '{data_path}'")
    model_path = config["model_path"]
    print(f"Model path: '{model_path}'")

    # Set up the output path.
    as_parquet = config.get("as_parquet", False)
    out_path = config.get("output_path")
    if out_path is None:
        out_path = (data_path / "boxes").with_suffix(
            ".parquet" if as_parquet else ".csv")
    out_path = Path(out_path)
    if out_path.exists():
        msg = f"'{out_path}' exists. Continue and overwrite (y/n)? "
        if not cli.prompt_yes(msg):
            sys.exit(1)
    print(f"Output path: '{out_path}'\n")

    # Load the model.
    model = ort.InferenceSession(model_path)

    # Load the images.
    data = io.FlyDatasetReader(data_path)

    # TODO: Can we just run with an entire list of images?
    # Apply the model to each image.
    results = []
    for path, image in data.iter_read():
        print(f"Detecting flies in '{path}'")

        # Preprocess the image.
        h, w = image.shape[:2]
        image = preprocess_image(image)
        new_h, new_w = image.shape[-2:]

        # Apply the model.
        result = model.run(None, {"images": image})
        result = result[0][0, :, :].transpose()

        # FIXME: Remove boxes that are in the padding area.

        # Convert results to a data frame.
        result = pd.DataFrame(
            result, columns = ["xc", "yc", "w", "h", "confidence"])
        result["path"] = str(path)
        result["arena_width"] = w
        result["arena_height"] = h

        # Correct for image rescaling.
        result[["xc", "w"]] *= w / new_w
        result[["yc", "h"]] *= h / new_h

        min_conf = config["minimum_confidence"]
        result = result.loc[min_conf <= result["confidence"], :]

        # FIXME: Non-maximum suppression.

        # TODO: Option to save boxes as images.

        results.append(result)

    results = pd.concat(results)

    # Converting px to cm for coordinates
    temperature_table = data.read_sheet_temperatures()
    lis_x = results.loc[:, 'xc']
    lis_y = results.loc[:, 'yc']
    arena_name = data_path.name.rsplit('_', 1)[1]
    table = io.read_config('configs/test.toml')
    arena_cm_horizontal = io.get_distance(table, arena_name)['horizontal']
    arena_cm_vertical = io.get_distance(table, arena_name)['vertical']

    x_px = results.loc[:, 'arena_width']
    x_cm = lis_x * arena_cm_horizontal / x_px
    y_px = results.loc[:, 'arena_height']
    y_cm = lis_y * arena_cm_vertical / y_px

    results['x_cm'] = x_cm
    results['y_cm'] = y_cm

    # Temperature
    temp_col = x_cm.map(
        lambda x: extract_extropolate.extropolate(x, temperature_table))
    results['temperatures'] = temp_col

    if as_parquet:
        results.to_parquet(out_path, index = False)
    else:
        results.to_csv(out_path)
    print(f"Wrote '{out_path}'")


def preprocess_image(image, shape = (384, 640)):
    """Preprocess an image to prepare it for ONNX.
    """
    # Rescale and pad.
    image = ops.rescale(image, shape)
    image = cv2.copyMakeBorder(
        image, 0, shape[0] - image.shape[0], 0, shape[1] - image.shape[1]
        , cv2.BORDER_CONSTANT, 0)

    # Convert image to float32.
    image = image.astype(np.float32)
    image /= 255.

    # Move channels to first dimension.
    image = np.moveaxis(image, -1, 0)
    image = image[np.newaxis, ...]

    return image
