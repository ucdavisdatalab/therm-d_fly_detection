"""Functions to train a YOLO model for fly detection.
"""

from pathlib import Path
import sys
import tomllib

import pandas as pd
import yaml

from . import cli
from . import io


def train_yolo(args):
    """Train a YOLO object detection model.
    """
    import ultralytics as ult

    # Read the config file.
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Use global config `name` if `name` isn't set in the train section.
    model_name = config["train"].get("name", config["name"])
    config = config["train"]

    # FIXME:
    yaml_path = ""

    # TODO: handle case where this is the first time the model is being
    # fine-tuned.
    model = ult.YOLO(config["pretrain_path"])

    model.train(
        data = str(yaml_path)
        # From config --------
        , name = model_name
        , epochs = config["epochs"]
        # Epochs to wait for no observable improvement for early stopping
        , patience = config["patience"]
        # --------------------
        # initial and final learning rate
        #, lr0, lrf
        #, optimizer
        # random seed
        #, seed
        # Top-level directory where results will be saved
        , project = "models"
        , pretrained = True
        , device = 0
        , workers = 1
        , exist_ok = False)

    # Do something with the model.


def assemble_training_set(args):
    """Assemble a (YOLO) training set from multiple annotated data sources.
    """
    # Read the config file.
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Use global config `name` if `name` isn't set in the assemble section.
    training_set_name = config["assemble"].get("name", config["name"])
    config = config["assemble"]

    data_dir = Path(config["data_dir"])

    # Make directories for the training data set.
    out_dir = config.get("out_dir", Path("outputs") / training_set_name)
    out_dir = Path(out_dir)
    print(f"Output directory: '{out_dir}'")
    if out_dir.is_dir() and next(out_dir.iterdir(), None):
        msg = ("Output directory contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not cli.prompt_yes(msg):
            sys.exit(1)

    out_images_dir = out_dir / "images" / "train"
    out_images_dir.mkdir(parents = True, exist_ok = True)
    out_labels_dir = out_dir / "labels" / "train"
    out_labels_dir.mkdir(parents = True, exist_ok = True)

    print("\nLinking files:")

    # Link in the images and labels.
    count = 0
    crosswalk = []
    for name in sorted(config["datasets"]):
        dataset = io.FlyDatasetReader(data_dir / name)

        for i, p in enumerate(dataset):
            # Link to image file.
            out_path = out_images_dir / (f"{count:04}" + p.suffix.lower())
            out_path.unlink(missing_ok = True)
            out_path.hardlink_to(p)
            print(f"  '{p}' -> '{out_path}'")

            # Check for and link to label file.
            label_path = dataset.label_paths[i]
            metadata = {
                "id": count
                , "source_photo": p, "source_label": label_path
                , "linked_photo": out_path
            }

            out_path = out_labels_dir / (
                f"{count:04}" + label_path.suffix.lower())
            out_path.unlink(missing_ok = True)
            out_path.hardlink_to(label_path)
            print(f"  '{label_path}' -> '{out_path}'\n")

            metadata["linked_label"] = out_path
            crosswalk.append(metadata)
            count += 1

    print(f"Linked {count} files.")

    # Create a YAML file.
    yolo_metadata = {
        "path": ""
        , "train": "images/train"
        , "val": "images/val"
        , "test": ""
        , "names": {0: "fly"}
    }
    out_path = out_dir / (training_set_name + ".yaml")
    with open(out_path, "wt") as f:
        yaml.dump(yolo_metadata, f)
    print(f"Wrote YAML file '{out_path}'.")

    # Save the crosswalk for the linked files as a CSV.
    out_path = out_dir / "crosswalk.csv"
    crosswalk = pd.DataFrame(crosswalk)
    crosswalk.to_csv(out_path, index = False)
    print(f"Wrote crosswalk '{out_path}'.")
