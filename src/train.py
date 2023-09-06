"""Functions to train a YOLO model for fly detection.
"""

from pathlib import Path
import sys
import tomllib

import pandas as pd

from . import cli
from . import io


def main():
    # Read the config file.
    PATH = "configs/test.toml"
    with open(PATH, "rb") as f:
        config = tomllib.load(f)

    config = config["training"]
    data_dir = Path(config["data_dir"])

    # Make a directory for the training data.
    out_dir = Path(config["assembly_dir"])
    if next(out_dir.iterdir(), None):
        msg = ("Output directory contains files. "
               "Continue and possibly overwrite (y/n)? ")
        if not cli.prompt_yes(msg):
            sys.exit(1)

    out_photos_dir = out_dir / "photos"
    out_photos_dir.mkdir(parents = True, exist_ok = True)
    out_labels_dir = out_dir / "labels"
    out_labels_dir.mkdir(parents = True, exist_ok = True)

    print("\nLinking files:")

    # Link in the images and labels.
    count = 0
    crosswalk = []
    for name in sorted(config["datasets"]):
        dataset = io.FlyDatasetReader(data_dir / name)

        for i, p in enumerate(dataset):
            # Link to image file.
            out_path = out_photos_dir / (f"{count:04}" + p.suffix)
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

            out_path = out_labels_dir / (f"{count:04}" + label_path.suffix)
            out_path.unlink(missing_ok = True)
            out_path.hardlink_to(label_path)
            print(f"  '{label_path}' -> '{out_path}'\n")

            metadata["linked_label"] = out_path
            crosswalk.append(metadata)
            count += 1

    print(f"Linked {count} files.")
    # Save the crosswalk for the linked files as a CSV.
    out_path = out_dir / "crosswalk.csv"
    crosswalk = pd.DataFrame(crosswalk)
    crosswalk.to_csv(out_path, index = False)
    print(f"Wrote crosswalk '{out_path}'.")


if __name__ == "__main__":
    main()
