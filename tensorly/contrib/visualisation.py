"""
This module contains plotting utilities
"""

# Author: Ilya Kisil <ilyakisil@gmail.com>

import numpy as np
import matplotlib.pyplot as plt


def plot_kruskal_factors(factors, components=None, max_default_length=5):
    """ Plot factor vectors of the CP-like decomposition.

    Each row illustrates factor vectors from different modes.

    Parameters
    ----------
    factors : list[np.ndarray]
        List of kryskal factors.
    components : list[int], optional
        List of factor vectors to be plotted. Its length should not be greater then
        the number of columns of factor matrices ``length(components) <= factors[0].shape[1]``.
    max_default_length : int, optional
        Defines maximum number of components to be plotted when the
        optional parameter ``components`` is provided. Suppressed otherwise.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects

    Notes
    -----
    In the kruskal representation of a tensor, there is one to one relation
    between factor vectors that belong to different modes.
    """

    if components is None:
        if factors[0].shape[1] >= max_default_length:
            default_length = max_default_length
        else:
            default_length = factors[0].shape[1]
        components = [i for i in range(default_length)]

    combinations = [tuple([i] * len(factors)) for i in components]
    fig, axis = _plot_component_groups(factors, combinations)
    return fig, axis


def plot_tucker_factors(factors, combinations=None):
    """ Plot factor vectors of the Tucker-like decomposition.

    Parameters
    ----------
    factors : list[np.ndarray]
        List of tucker factors.
    combinations : list[tuple], optional
        List of factor vectors combinations to be plotted. Its length should not be greater then
        the product of columns number of all factor matrices.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects

    Notes
    -----
    In the tucker representation of a tensor, any factor vector from any factor matrix is related
    to all factor vectors from all other factor matrices.
    """
    if combinations is None:
        combinations = [tuple([0] * len(factors))]

    fig, axis = _plot_component_groups(factors, combinations)
    return fig, axis


def _plot_component_groups(factors, combinations):
    """ Generalised interface for plotting factor vectors

    Parameters
    ----------
    factors : list[np.ndarray]
        List of factor matrices obtained through a tensor decomposition algorithm.
    combinations : list[tuple]
        List of combinations of factor vectors from different modes to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects
    """
    n_rows = len(combinations)
    n_cols = len(factors)
    axis_width = 4
    axis_height = 4
    fig, axis = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(n_cols * axis_width, n_rows * axis_height)
                             )
    # Always keep axis as 2-D array
    if axis.ndim == 1:
        axis = np.array([axis])

    for row, group in enumerate(combinations):
        for mode, n_component in enumerate(group):
            component = factors[mode][:, n_component]
            title = "Mode-{}, Component #{}".format(mode, n_component)
            col = mode
            axis[row, col].plot(component)
            axis[row, col].set_title(title)
    plt.tight_layout()
    return fig, axis

