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
    """Use the fly detection model to detect flies for all images in a data
    set.
    """
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
        print(f"Image: '{path}'")

        # Preprocess the image.
        orig = image.copy()
        arena_h, arena_w = image.shape[:2]
        image, pad_x, pad_y = preprocess_image(image)
        padded_h, padded_w = image.shape[-2:]

        # Apply the model.
        print("  Detecting flies...")
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
        print(f"  Found {result.shape[0]} flies above {min_conf}"
              " minimum confidence.")

        # Non-maximum suppression.
        print("  Suppressing redundant detections...")
        max_iou = config.get("maximum_iou", 0.25)
        ix = suppress_nonmax(
            result[["x_px", "y_px", "width_px", "height_px"]].to_numpy()
            , result["confidence"].to_numpy()
            , max_iou)
        result = result.iloc[ix, :]
        print(f"  Kept {result.shape[0]} flies below {max_iou}"
              " maximum IoU.")

        result["id"] = range(1, len(result) + 1)

        results.append(result)

        # Option to save boxes as images.
        if is_debug:
            temp = result.loc[:, ['x_px', 'y_px', 'width_px', 'height_px']]
            temp = temp.to_numpy()
            img_name = path.name.rsplit("/", 1)[-1]

            # Flip temperature table to iterate through every degree

            flipped = temperature_table[1, :]
            flipped = np.vstack([flipped, temperature_table[0, :]])

            lb = int(flipped[0, 0])
            ub = int(-(-flipped[0, -1] // 1))

            x_lines = []
            degrees = []

            i = lb
            while i <= ub:
                x_lines.append(tmp.estimate(i, flipped))
                degrees.append(i)
                i += .5

            vert_lines = [
                orig.shape[1] * i / apparatuses[data.apparatus]['horizontal']
                for i in x_lines]
            arr = np.array([vert_lines, degrees])
            indices_to_keep = np.where(
                (arr[0] >= 0) & (arr[0] <= orig.shape[1]))
            mat = arr[:, indices_to_keep[0]]
            x_positions = mat[0]
            text_values = mat[1]

            # Plot boxes and degree lines
            fig, ax = plt.subplots(1, 1, figsize = (20, 20))
            viz.plot_image(orig, ax = ax)

            for i in range(temp.shape[0]):
                xc, yc, w, h = temp[i, :4]
                viz.plot_box(
                    (yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2), ax)
                ax.text(
                    xc - w / 2, yc + h / 2, i + 1
                    , ha='right', va='bottom', fontsize = 'x-small'
                    , color = 'black')

            for x_pos, text_val in zip(x_positions, text_values):
                ax.axvline(x=x_pos, color='r', linestyle='--', alpha = .15)
                ax.text(
                    x_pos, round(arena_h * .025), text_val
                    , ha='right', va='bottom', fontsize = 'small', color = 'b')

            # Save into folder
            plt.savefig(debug_dir / img_name)
            plt.close()
            print(f"  Wrote '{debug_dir / img_name}'")
            print()

    results = pd.concat(results)

    column_names = results.columns
    id_col = results.iloc[:, -1]
    results = results.drop(results.columns[-1], axis = 1)
    results.insert(0, column_names[-1], id_col)

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

    Arguments
    ---------
    image: np.ndarray
        Image to preprocess.

    shape: tuple of ints
        Required image shape (height, width) for input to the model.
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


def suppress_nonmax(boxes, confidence, maximum_iou = 0.25):
    """Remove redundant bounding boxes by keeping only the highest-confidence
    box in each group of similar boxes.

    Arguments
    ---------
    boxes: np.ndarray
        A matrix where each row corresponds to one box and there are 4 columns:
        x center, y center, width, and height.

    confidence: np.ndarray
        An array of confidence scores, with one element for each row in
        `boxes`.

    maximum_iou: float
        Maximum intersection-over-union (IoU) ratio for a lower-confidence box
        to be kept. Two boxes have a high IoU ratio if most of their areas
        overlap.

    Returns
    -------
    out: np.ndarray
        Indexes of rows to keep.
    """
    # Convert format from (x, y, w, h) to (left, right, top, bottom, area).
    data = boxes
    boxes = np.full((data.shape[0], 5), np.NaN)
    boxes[:, 0] = data[:, 0] - 0.5 * data[:, 2]
    boxes[:, 1] = data[:, 0] + 0.5 * data[:, 2]
    boxes[:, 2] = data[:, 1] - 0.5 * data[:, 3]
    boxes[:, 3] = data[:, 1] + 0.5 * data[:, 3]
    boxes[:, 4] = data[:, 2] * data[:, 3]

    ix_remaining = np.argsort(confidence)
    ix_kept = np.full_like(ix_remaining, -1)
    n_kept = 0
    while len(ix_remaining) > 0:
        # Of the remaining boxes, get the one with the highest similarity
        # score. This is the "best" remaining box.
        ix_best = ix_remaining[-1]
        ix_remaining = ix_remaining[:-1]

        # Keep the best box.
        ix_kept[n_kept] = ix_best
        n_kept += 1

        # Compute intersection-over-union of best box with all remaining boxes.
        ious = intersection_over_union(
            boxes[ix_best, :], boxes[ix_remaining, :])
        ix_remaining = ix_remaining[ious <= maximum_iou]

    return ix_kept[:n_kept]


def intersection_over_union(box, boxes):
    """Compute the intersection-over-union for one box against many boxes.

    Note: for efficiency, this function does not accept the standard YOLO box
    format (x, y, w, h) as input.

    Arguments
    ---------
    box: np.ndarray
        An array which corresponds to one box, where the elements are the left,
        right, top, bottom, and area.

    boxes: np.ndarray
        A matrix where each row corresponds to one box and there are 5 columns:
        left, right, top, bottom, and area.
    """
    # Compute intersection area.
    w = np.fmin(box[1], boxes[:, 1]) - np.fmax(box[0], boxes[:, 0])
    h = np.fmin(box[3], boxes[:, 3]) - np.fmax(box[2], boxes[:, 2])

    intersect_area = np.fmax(0, w) * np.fmax(0, h)
    union_area = box[4] + boxes[:, 4] - intersect_area
    return intersect_area / union_area
