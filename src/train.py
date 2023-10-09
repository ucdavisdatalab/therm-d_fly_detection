"""Functions to train a YOLO model for fly detection.
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import tomllib

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from . import cli
from . import io


def train_yolo(args):
    """Train a YOLO object detection model.
    """
    import ultralytics as ult

    # Read the config file.
    print(f"Config path: '{args.config}'")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Use global config `name` if `name` isn't set in the train section.
    model_name = config["train"].get("name", config["name"])
    config = config["train"]

    # FIXME:
    yaml_path = config["data_path"]

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


def assemble_training_set(config):
    """Assemble a (YOLO) training set from multiple annotated data sources.
    """
    # Use global config `name` if `name` isn't set in the assemble section.
    training_set_name = config["assemble"].get("name", config["name"])
    config = config["assemble"]

    data_dir = Path(config["data_dir"])

    # Assemble file metadata.
    count = 0
    metadata = []
    for name in sorted(config["datasets"]):
        dataset = io.FlyDatasetReader(data_dir / name)
        apparatus = dataset.apparatus

        # FIXME: image and labels could be out of correspondence if there are
        # no labels for some images.
        for image_path, label_path in zip(dataset, dataset.label_paths):
            metadata.append({
                "id": count
                , "source_image": image_path, "source_label": label_path
                , "apparatus": apparatus
            })
            count += 1

    metadata = pd.DataFrame(metadata)
    print(f"Found {metadata.shape[0]} image and label file pairs.")

    # Assign each file to train set or validation set.
    train, vdate = train_test_split(
        metadata["id"], shuffle = True
        , test_size = config["test_size"]
        , random_state = config["seed"]
        , stratify = metadata["apparatus"])

    metadata["train_set"] = None
    metadata.loc[metadata["id"].isin(train), "train_set"] = "train"
    metadata.loc[metadata["id"].isin(vdate), "train_set"] = "val"

    print(pd.crosstab(metadata["apparatus"], metadata["train_set"]))

    # Make directories for the training data set.
    out_dir = config.get("out_dir", Path("outputs") / training_set_name)
    out_dir = Path(out_dir)
    cli.prompt_overwrite(out_dir, "Output directory")

    out_images_dir = out_dir / "images"
    (out_images_dir / "train").mkdir(exist_ok = True, parents = True)
    (out_images_dir / "val").mkdir(exist_ok = True, parents = True)

    out_labels_dir = out_dir / "labels"
    (out_labels_dir / "train").mkdir(exist_ok = True, parents = True)
    (out_labels_dir / "val").mkdir(exist_ok = True, parents = True)

    # Generate paths.
    def make_path(row, source, base):
        name = f"{row['id']:04}" + Path(row[source]).suffix.lower()
        return base / row["train_set"] / name

    metadata["linked_image"] = metadata.apply(
        make_path, axis = 1, args = ("source_image", out_images_dir))
    metadata["linked_label"] = metadata.apply(
        make_path, axis = 1, args = ("source_label", out_labels_dir))

    print("\nLinking files:")

    # Link in the images and labels.
    for _, row in metadata.iterrows():
        out_path = Path(row["linked_image"])
        out_path.unlink(missing_ok = True)
        out_path.hardlink_to(row["source_image"])
        print(f"  '{row['source_image']}' -> '{out_path}'")

        out_path = Path(row["linked_label"])
        out_path.unlink(missing_ok = True)
        out_path.hardlink_to(row["source_label"])
        print(f"  '{row['source_label']}' -> '{out_path}'\n")

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
    out_path = out_dir / "metadata.csv"
    metadata.to_csv(out_path, index = False)
    print(f"Wrote metadata '{out_path}'.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type = Path, help = "path to config file")
    args = parser.parse_args()
    train_yolo(args)
