"""This module handles visualization.
"""

import cv2
import matplotlib.pyplot as plt

from src.io import rescale_image


def show_image(
    image, ax = None, grayscale = False, max_size = 0, **kwargs
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

    max_size: int
        Maximum resolution for displayed image. If positive, the image is
        resized so that this is the length of its longest side. If
        non-positive, the image is displayed at full resolution. Using lower
        resolutions can speed up plotting.

    **kwargs
        Additional arguments passed on to matplotlib.pyplot.subplots.
    """
    if ax is None:
        _, ax = plt.subplots(**kwargs)

    print(image.shape)
    if max_size > 0:
        print(max_size)
        image = rescale_image(image, max_size)

    n_channels = 1
    if len(image.shape) == 3:
        n_channels = image.shape[-1]

    if grayscale and n_channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        n_channels = 1

    if n_channels == 1:
        ax.imshow(image, cmap = "gray")
    else:
        ax.imshow(image)

    return ax
