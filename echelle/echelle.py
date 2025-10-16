from __future__ import print_function, division

import warnings

import numpy as np
import matplotlib.pyplot as plt

from .utils import smooth_power

__all__ = ["echelle", "plot_echelle"]


def echelle(freq, power, dnu, fmin=0.0, fmax=None, offset=0.0):
    """Calculates the echelle diagram. Use this function if you want to do
    some more custom plotting.

    Parameters
    ----------
    freq : array-like
        Frequency values
    power : array-like
        Power values for every frequency
    dnu : float
        Value of deltanu
    fmin : float, optional
        Minimum frequency to calculate the echelle at, by default 0.
    fmax : float, optional
        Maximum frequency to calculate the echelle at. If none is supplied,
        will default to the maximum frequency passed in `freq`, by default None
    offset : float, optional
        An offset to apply to the echelle diagram, by default 0.0

    Returns
    -------
    z : array-like
        The 2D array of the echelle
    """
    if fmax == None:
        fmax = freq[-1]

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    # trim data
    m = (freq >= fmin) & (freq <= fmax)
    freq, power = freq[m], power[m]

    # wack ass speedup
    split_indices = np.where(np.diff(freq % dnu / dnu) < 0)[0] + 1
    a_split = np.split(power, split_indices)
    max_length = max(len(row) for row in a_split)
    z = np.zeros((len(a_split), max_length))
    for i, row in enumerate(a_split):
        z[i, : len(row)] = row

    return z


def plot_echelle(
    freq,
    power,
    dnu,
    mirror=False,
    ax=None,
    cmap="Blues",
    scale=None,
    interpolation=None,
    smooth=False,
    smooth_filter_width=50,
    **kwargs,
):
    """Plots the echelle diagram.

    Parameters
    ----------
    freq : numpy array
        Frequency values
    power : array-like
        Power values for every frequency
    dnu : float
        Value of deltanu
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        A matplotlib axes to plot into. If no axes is provided, a new one will
        be generated, by default None
    cmap : str, optional
        A matplotlib colormap, by default 'BuPu'
    scale : str, optional
        either 'sqrt' or 'log' or None. Scales the echelle to bring out more
        features, by default 'sqrt'
    interpolation : str, optional
        Type of interpolation to perform on the echelle diagram through
        matplotlib.pyplot.imshow, by default 'none'
    smooth_filter_width : float, optional
        Amount by which to smooth the power values, using a Box1DKernel
    **kwargs : dict
        Dictionary of arguments to be passed to `echelle.echelle`

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The plotted echelle diagram on the axes
    """
    if smooth:
        power = smooth_power(power, smooth_filter_width)
    echz = echelle(freq, power, dnu, **kwargs)

    if scale is not None:
        if scale == "log":
            echz = np.log10(echz)
        elif scale == "sqrt":
            echz = np.sqrt(echz)

    if ax is None:
        fig, ax = plt.subplots()

    if "fmin" in kwargs:
        fmin = kwargs["fmin"]
    else:
        fmin = freq[0]
    if "fmax" in kwargs:
        fmax = kwargs["fmax"]
    else:
        fmax = freq[-1]

    ax.imshow(
        echz,
        aspect="auto",
        extent=(0, dnu, fmin, fmax),
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
    )

    # It's much cheaper just to replot the data we already have
    # and mirror it.
    # if mirror:
    #     ax.imshow(
    #         echz,
    #         aspect="auto",
    #         extent=(
    #             (echx.min() + dnu),
    #             (echx.max() + dnu),
    #             (echy.min() - dnu),
    #             (echy.max()) - dnu,
    #         ),
    #         origin="lower",
    #         cmap=cmap,
    #         interpolation=interpolation,
    #     )

    ax.set_xlabel(r"Frequency" + " mod " + str(dnu))
    ax.set_ylabel(r"Frequency")
    return ax
