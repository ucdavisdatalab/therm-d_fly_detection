"""Image transformations and operations that use Tesseract.

Note: this file is outdated and NOT used by any of the fly detection tools.
It's just here as a record of attempted methods.
"""

import cv2
import pytesseract as pyt

from . import io
from . import ops


def orient_image(image):
    """Use Tesseract to orient an image.

    Arguments
    ---------
    image: numpy.ndarray
        The image to orient.
    """
    # Improve contrast and denoise.
    # TODO: Likely not all of these steps are necessary.
    transformed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    transformed = cv2.GaussianBlur(transformed, (0, 0), 1)
    # TODO: How to set the CLAHE parameter?
    clahe = cv2.createCLAHE(clipLimit = 10)
    transformed = clahe.apply(transformed)
    transformed = ops.unsharp_mask(transformed, 1)

    # The 'rotate' field is the clockwise rotation angle to correctly orient
    # the text.
    osd = pyt.image_to_osd(
        transformed, lang = "eng", output_type = pyt.Output.DICT)
    rotate = osd["rotate"]
    print(f"Image requires {rotate} degree clockwise rotation.")
    if rotate == 0:
        return image

    rotate = {
        90: cv2.ROTATE_90_CLOCKWISE
        , 180: cv2.ROTATE_180
        , 270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }[rotate]

    return cv2.rotate(image, rotate)


def main():
    exp_dir = "data/2023-03-10_shiny"
    dset = io.FlyDatasetReader(exp_dir)

    for i in range(len(dset)):
        img = dset.read_fly(i)
        orient_image(img)


if __name__ == "__main__":
    main()
