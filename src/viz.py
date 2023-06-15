"""This module handles visualization.
"""

from PIL import Image
import matplotlib.pyplot as plt

from src.io import rescale_image


def show_image(
    image, ax = None, grayscale = False, max_resolution = 1_000, **kwargs
):
    """Display an image.

    Arguments
    ---------
    image: numpy.ndarray
        The image to display.

    ax: matplotlib.axes._axes.Axes
        Matplotlib axes on which to display the image.

    grayscale: bool
        Whether to convert the image to grayscale before displaying.

    max_resolution: int
        Maximum resolution for displayed image. If positive, the image is
        resized so that this is the length of its longest side. If
        non-positive, the image is displayed at full resolution. Using lower
        resolutions can speed up plotting.

    **kwargs
        Additional arguments passed on to matplotlib.pyplot.subplots.
    """
    if ax is None:
        _, ax = plt.subplots(**kwargs)

    image = Image.fromarray(image)
    if max_resolution > 0:
        image = rescale_image(image, max_resolution)

    if grayscale:
        image = image.convert("L")
        ax.imshow(image, cmap = "gray")
    else:
        ax.imshow(image)

    return ax
