# Script to test orientation function on every image in a dataset.

from src.io import *
from src.tsops import *


def main():
    exp_dir = "data/2023-03-10_shiny"
    dset = FlyDatasetReader(exp_dir)

    for i in range(len(dset)):
        img = dset.read_fly(i)
        orient_image(img)


if __name__ == "__main__":
    main()
