"""Functions to predict fly bounding boxes.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import cli
from . import io
from . import ops
from . import temperature as tmp


def main(config):
    import onnxruntime as ort

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
        h, w = image.shape[:2]
        image = preprocess_image(image)
        new_h, new_w = image.shape[-2:]

        # Apply the model.
        result = model.run(None, {"images": image})
        result = result[0][0, :, :].transpose()

        # FIXME: Remove boxes that are in the padding area.

        # Convert results to a data frame.
        result = pd.DataFrame(
            result, columns = [
                "x_px", "y_px", "width_px", "height_px", "confidence"])
        result["path"] = str(path)
        result["arena_width_px"] = w
        result["arena_height_px"] = h

        # Correct for image rescaling.
        result[["x_px", "width_px"]] *= w / new_w
        result[["y_px", "height_px"]] *= h / new_h

        min_conf = config["minimum_confidence"]
        result = result.loc[min_conf <= result["confidence"], :]

        # FIXME: Non-maximum suppression.

        # TODO: Option to save boxes as images.

        results.append(result)

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
