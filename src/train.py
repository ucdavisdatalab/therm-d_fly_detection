"""Functions to train a YOLO model for fly detection.
"""

from pathlib import Path
import tomllib

from . import io


def fly_dataset_subdirs(path):
    path = Path(path)

    photos_path = path / "photos"
    labels_path = Path("staging").joinpath(*path.parts[1:]) / "labels"

    print(f"{photos_path=}")
    print(f"{labels_path=}")
    assert photos_path.is_dir()
    assert labels_path.is_dir()

    return photos_path, labels_path


def main():
    # Read the config file.
    PATH = "configs/test.toml"
    with open(PATH, "rb") as f:
        config = tomllib.load(f)

    config = config["training"]

    # Check that the data sets exist.
    dataset_dirs = [Path(p) for p in sorted(config["datasets"])]

    # Make a directory for the training data.
    out_dir = Path(config["assembly_dir"])

    out_photos_dir = out_dir / "photos"
    out_photos_dir.mkdir(parents = True, exist_ok = True)
    out_labels_dir = out_dir / "labels"
    out_labels_dir.mkdir(parents = True, exist_ok = True)

    print()

    # Link in the images and labels.
    count = 1
    for dataset_path in dataset_dirs:
        dataset = io.FlyDatasetReader(dataset_path)
        labels_dir = Path("staging").joinpath(*dataset_path.parts[1:])
        labels_dir /= "labels"

        for p in dataset:
            # Link to image file.
            out_path = out_photos_dir / (f"{count:04}" + p.suffix)
            out_path.hardlink_to(p)
            print(f"  '{p}' -> '{out_path}'")

            # Check for and link to label file.
            label_path = labels_dir / (p.stem + ".txt")
            assert label_path.exists()

            out_path = out_labels_dir / (f"{count:04}" + label_path.suffix)
            out_path.hardlink_to(label_path)
            print(f"  '{label_path}' -> '{out_path}'\n")
            count += 1

    # Set up the training data by linking the images.
    print(config)


if __name__ == "__main__":
    main()
