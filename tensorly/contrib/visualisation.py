"""
This module contains plotting utilities
"""

# Author: Ilya Kisil <ilyakisil@gmail.com>

import numpy as np
import matplotlib.pyplot as plt


def plot_kruskal_factors(factors, components=None, max_default_length=5, custom_plots=None):
    """ Plot factor vectors of the CP-like decomposition.

    Each row illustrates combinations of factor vectors from different modes.

    Parameters
    ----------
    factors : list[np.ndarray]
        List of kryskal factors.
    components : list[int], optional
        List of factor vectors to be plotted. Its length should not be greater then
        the number of columns of factor matrices ``length(components) <= factors[0].shape[1]``.
    max_default_length : int, optional
        Defines maximum number of components to be plotted when the
        optional parameter `components` is provided. Suppressed otherwise.
    custom_plots : dict[int, callable], optional
        Dictionary with values being custom plotting functions and keys being the mode
        numbers these functions correspond to. If not specified, then line plots will be used.
        Signature of the custom functions must be as follows: ``custom_plotting_function(ax, data)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects
        2D array (row, col) of ``Axes`` objects each of which illustrates a
        particular component from `factors`.

    Notes
    -----
    1)  In the kruskal representation of a tensor, there is one to one relation
        between factor vectors that belong to different modes.
    2)  Dictionary `custom_plots` is used to update the dictionary with default plotting functions.
        Therefore, main use if for changing default behaviour.
    """
    # # # We can hardcode `max_default_length` if needed
    # # if components is None:
    # #     default_length = 5
    # #     if factors[0].shape[1] < default_length:
    # #         default_length = factors[0].shape[1]
    # #     components = [i for i in range(default_length)]

    if components is None:
        if factors[0].shape[1] >= max_default_length:
            default_length = max_default_length
        else:
            default_length = factors[0].shape[1]
        components = [i for i in range(default_length)]

    combinations = [tuple([i] * len(factors)) for i in components]

    fig, axis = _plot_component_groups(factors=factors,
                                       combinations=combinations,
                                       custom_plots=custom_plots
                                       )
    return fig, axis


def plot_tucker_factors(factors, combinations=None, custom_plots=None):
    """ Plot factor vectors of the Tucker-like decomposition.

    Each row illustrates combinations of factor vectors from different modes.

    Parameters
    ----------
    factors : list[np.ndarray]
        List of tucker factors.
    combinations : list[tuple], optional
        List of factor vectors combinations to be plotted. Its length should not be greater then
        the product of columns number of all factor matrices.
    custom_plots : dict[int, callable], optional
        bla bla

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects
        2D array (row, col) of ``Axes`` objects each of which illustrates a
        particular component from `factors`.

    Notes
    -----
    In the tucker representation of a tensor, any factor vector from any factor matrix is related
    to all factor vectors from all other factor matrices.
    """
    if combinations is None:
        combinations = [tuple([0] * len(factors))]

    fig, axis = _plot_component_groups(factors=factors,
                                       combinations=combinations,
                                       custom_plots=custom_plots
                                       )
    return fig, axis


def _plot_component_groups(factors, combinations, custom_plots):
    """ Generalised interface for plotting factor vectors

    Parameters
    ----------
    factors : list[np.ndarray]
        List of factor matrices obtained through a tensor decomposition algorithm.
    combinations : list[tuple]
        List of combinations of factor vectors from different modes to be plotted.
    custom_plots : dict[int, callable]
        Dictionary with custom plotting functions for different modes.
        If None, then all components are represented through a line plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Top level container for all plot elements.
    axis : np.ndarray of matplotlib.axes.Axes objects
        2D array (row, col) of ``Axes`` objects each of which illustrates a
        particular component from `factors`.
    """
    n_rows = len(combinations)
    n_cols = len(factors)
    axis_width = 4
    axis_height = 4
    fig, axis = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(n_cols * axis_width, n_rows * axis_height)
                             )

    # Always keep axis as 2-D array [n_rows, n_cols] which helps with further generalisation
    if n_rows == 1 or n_cols == 1:  # can replace with if axis.ndim == 1:
        axis = np.array([axis])
        if n_cols == 1:
            axis = axis.T

    plot_bank = {i: _line_plot for i in range(n_cols)}
    if custom_plots is not None:
        plot_bank.update(custom_plots)

    for row, group in enumerate(combinations):
        for mode, n_component in enumerate(group):
            col = mode
            plot_function = plot_bank[mode]
            component = factors[mode][:, n_component]
            title = "Mode-{}, Component #{}".format(mode, n_component)
            plot_function(ax=axis[row, col],
                          data=component)
            axis[row, col].set_title(title)
    plt.tight_layout()
    return fig, axis


def _line_plot(ax, data):
    """ Default plotting function for each mode

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object which is used to illustrate `data`
    data : np.ndarray
        Array of data to be plotted. Shape of such array is ``(N, 1)``
    """
    ax.plot(data)
