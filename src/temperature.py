"""Functions to extract and estimate temperatures.
"""

import numpy as np
import pandas as pd


def estimate(x, mat):
    """Estimate temperature based on position in centimeters, using
    measurements from temperature probes.

    Arguments
    ---------
    x: float
        Position at which to estimate temperature.

    mat: np.ndarray
        Matrix with 2 rows, where the first row contains probe positions in
        centimeters and the second row contains temperature measurements at
        those positions.
    """
    # TODO: Make sure the temperature matrix is always sorted.
    # Get index of right endpoint of interval for estimation.
    if x < mat[0, 0]:
        # Position is left of leftmost probe, so extrapolate from leftmost
        # interval (right endpoint at index 1).
        ix = 1
    elif x > mat[0, -1]:
        # Position is right of rightmost probe, so extrapolate from rightmost
        # interval (right endpoint at index -1).
        ix = -1
    else:
        # Position is between probes, so interpolate.
        ix = np.searchsorted(mat[0, :], x)

    m = (mat[1, ix] - mat[1, ix - 1]) / (mat[0, ix] - mat[0, ix - 1])
    return m * (x - mat[0, ix - 1]) + mat[1, ix - 1]


def get_matrix(path):
    """Turns excel sheet into np array.
    """
    df = pd.read_excel(path)
    ind = df[df.iloc[:, 0] == 'temperature probe'].index[0]
    df = df[ind:ind + 3].dropna(axis = 1)
    df.columns = df.iloc[0]
    df = df[1:3]
    df = df.set_index('temperature probe')
    df = df.to_numpy()
    return df


