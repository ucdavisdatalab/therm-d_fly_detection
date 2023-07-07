"""This module handles visualization.
"""

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.io import rescale_image


def plot_image(
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


def plot_image_grid(*args, nrow = None, ncol = None, axs = None, **kwargs):
    """Plot a grid of images.

    Parameters
    ----------
    *args: lists of images
        Lists of images to plot; each argument is plotted on a separate row.
    axs: list of Axes
        Axes on which to plot the images.
    """
    nrow = len(args) if nrow is None else nrow
    ncol = max(len(a) for a in args) if ncol is None else ncol

    if axs is None:
        _, axs = plt.subplots(
            nrow, ncol, squeeze = False, layout = "constrained", **kwargs)

    for i in range(nrow):
        row = args[i]
        for j in range(len(row)):
            img = row[j]
            ax = axs[i][j]

            match img:
                case dict() as img:
                    plot_image(img.img, ax)
                    ax.set_title(img.title)
                case _:
                    plot_image(img, ax)
            ax.set_xticks([])
            ax.set_yticks([])

    return axs


def plot_box(coordinates, ax = None, **kwargs):
    """Draw a box on a plot.

    Arguments
    ---------
    coordinates: tuple
        The box coordinates as top, left, bottom, right.

    ax: matplotlib.axes._axes.Axes
        Matplotlib axes on which to display the box.

    **kwargs
        Additional arguments passed on to matplotlib.patches.Rectangle.
    """
    if ax is None:
        _, ax = plt.subplots()

    if "facecolor" not in kwargs:
        kwargs["facecolor"] = "none"
    if "edgecolor" not in kwargs:
        kwargs["edgecolor"] = "#00FF00"

    t, l, b, r = coordinates
    box = mpl.patches.Rectangle(
        (l, t), width = r - l, height = b - t, **kwargs)
    ax.add_patch(box)

    return ax
