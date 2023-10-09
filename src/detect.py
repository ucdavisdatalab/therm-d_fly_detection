"""Functions to predict fly bounding boxes.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd

from . import cli
from . import io
from . import ops
from . import temperature as tmp
from . import viz


def main(config):
    apparatuses = config["apparatuses"]
    config = config["detect"]

    data_path = Path(config["data_path"])
    print(f"Data path: '{data_path}'")
    model_path = config["model_path"]
    print(f"Model path: '{model_path}'")

    # Set up the output path.
    as_parquet = config.get("as_parquet", False)
    output_path = config.get("output_path")
    if output_path is None:
        output_path = (data_path / "predictions").with_suffix(
            ".parquet" if as_parquet else ".csv")
    output_path = Path(output_path)
    cli.prompt_overwrite(output_path, "Output path")

    is_debug = config.get("debug", False)
    if is_debug:
        debug_dir = data_path / "debug/predictions"
        cli.prompt_overwrite(debug_dir, "Debug directory", mkdir = True)

    # Load the model.
    model = ort.InferenceSession(model_path)

    # Load the images and temperatures.
    data = io.FlyDatasetReader(data_path)
    temperature_table = data.read_sheet_temperatures()

    # TODO: Can we just run with an entire list of images?
    # Apply the model to each image.
    results = []
    for path, image in data.iter_read_arenas():
        print(f"Detecting flies in '{path}'")

        # Preprocess the image.
        orig = image.copy()
        arena_h, arena_w = image.shape[:2]
        image, pad_x, pad_y = preprocess_image(image)
        padded_h, padded_w = image.shape[-2:]

        # Apply the model.
        result = model.run(None, {"images": image})
        result = result[0][0, :, :].transpose()

        # Convert results to a data frame.
        result = pd.DataFrame(
            result, columns = [
                "x_px", "y_px", "width_px", "height_px", "confidence"])
        result["path"] = str(path)
        result["arena_width_px"] = arena_w
        result["arena_height_px"] = arena_h

        # Remove boxes that are in the padding area.
        model_arena_w = padded_w - pad_x
        model_arena_h = padded_h - pad_y
        result.query(
            "x_px <= @model_arena_w and y_px <= @model_arena_h"
            , inplace = True)

        # Correct for image rescaling.
        result[["x_px", "width_px"]] *= arena_w / model_arena_w
        result[["y_px", "height_px"]] *= arena_h / model_arena_h

        min_conf = config["minimum_confidence"]
        result = result.loc[min_conf <= result["confidence"], :]

        # FIXME: Non-maximum suppression.

        results.append(result)

        # TODO: Option to save boxes as images.
        # debug then run the following

        if is_debug:
            temp = result.loc[:, ['x_px', 'y_px', 'width_px', 'height_px']]
            temp = temp.to_numpy()
            img_name = path.name.rsplit("/", 1)[-1]

            # Flipping temperatrue table to iterate through every degree

            flipped = temperature_table[1, :]
            flipped = np.vstack([flipped, temperature_table[0, :]])

            lb = int(flipped[0, 0])
            ub = int(-(-flipped[0, -1] // 1))

            x_lines = []
            degrees = []
            for i in range(lb, ub + 1):
                x_lines.append(tmp.estimate(i, flipped))
                degrees.append(i)

            vert_lines = [
                orig.shape[1] * i / apparatuses[data.apparatus]['horizontal']
                for i in x_lines]
            arr = np.array([vert_lines, degrees])
            indices_to_keep = np.where(
                (arr[0] >= 0) & (arr[0] <= orig.shape[1]))
            mat = arr[:, indices_to_keep[0]]
            x_positions = mat[0]
            text_values = mat[1]
            text_height = int(arena_w * .0175)

            # plotting boxes and degree lines and saving into folder

            fig, ax = plt.subplots(1, 1, figsize = (20, 20))
            viz.plot_image(orig, ax = ax)

            for i in range(temp.shape[0]):
                xc, yc, w, h = temp[i, :4]
                viz.plot_box(
                    (yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2), ax)

            for x_pos, text_val in zip(x_positions, text_values):
                ax.axvline(x=x_pos, color='r', linestyle='--', alpha = .15)
                ax.text(
                    x_pos, text_height, f'{text_val}', ha='right', va='bottom'
                    , fontsize=12, color = 'b')

            plt.savefig(debug_dir / img_name)
            plt.close()

    results = pd.concat(results)

    # Compute coordinates in centimeters, converting from pixels.
    apparatus = data.apparatus
    arena_w_cm = apparatuses[apparatus]["horizontal"]
    arena_h_cm = apparatuses[apparatus]["vertical"]

    arena_w_px = results["arena_width_px"]
    arena_h_px = results["arena_height_px"]

    x_px = results["x_px"]
    y_px = results["y_px"]

    x_cm = x_px * arena_w_cm / arena_w_px
    y_cm = y_px * arena_h_cm / arena_h_px
    results["x_cm"] = x_cm
    results["y_cm"] = y_cm

    # Compute temperature estimates.
    results["temperature"] = x_cm.map(
        lambda x: tmp.estimate(x, temperature_table))

    if as_parquet:
        results.to_parquet(output_path, index = False)
    else:
        results.to_csv(output_path, index = False)
    print(f"Wrote '{output_path}'")


def preprocess_image(image, shape = (384, 640)):
    """Preprocess an image to prepare it for ONNX.
    """
    # Rescale and pad.
    image = ops.rescale(image, shape)
    pad_y = shape[0] - image.shape[0]
    pad_x = shape[1] - image.shape[1]
    image = cv2.copyMakeBorder(
        image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, 0)

    # Convert image to float32.
    image = image.astype(np.float32)
    image /= 255.

    # Move channels to first dimension.
    image = np.moveaxis(image, -1, 0)
    image = image[np.newaxis, ...]

    return image, pad_x, pad_y
