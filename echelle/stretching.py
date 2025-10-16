#!/usr/bin/env python3
# Yaguang Li

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def make_fold(
    nu, ps, period, n_stack, n_element, idx, echelle_type="single", reverse=False
):
    if echelle_type == "single":
        z = np.zeros([n_stack, n_element])
        base = np.linspace(0, period, n_element, endpoint=False)
        for istack in range(n_stack):
            z[-istack - 1, :] = np.interp(
                base,
                nu[idx[istack] : idx[istack + 1]],
                ps[idx[istack] : idx[istack + 1]],
                period=period,
            )
    else:
        z = np.zeros([n_stack, 2 * n_element])
        base = np.linspace(0, 2 * period, 2 * n_element, endpoint=False)
        for istack in range(n_stack - 1):
            if reverse:
                z[-istack - 1, :] = np.r_[
                    np.interp(
                        base[:n_element],
                        nu[idx[istack + 1] : idx[istack + 2]],
                        ps[idx[istack + 1] : idx[istack + 2]],
                        period=period,
                    ),
                    np.interp(
                        base[:n_element],
                        nu[idx[istack] : idx[istack + 1]],
                        ps[idx[istack] : idx[istack + 1]],
                        period=period,
                    ),
                ]
            else:
                z[-istack - 1, :] = np.r_[
                    np.interp(
                        base[:n_element],
                        nu[idx[istack] : idx[istack + 1]],
                        ps[idx[istack] : idx[istack + 1]],
                        period=period,
                    ),
                    np.interp(
                        base[:n_element],
                        nu[idx[istack + 1] : idx[istack + 2]],
                        ps[idx[istack + 1] : idx[istack + 2]],
                        period=period,
                    ),
                ]
    return base, z


def period_echelle(
    nu,
    ps,
    ΔΠ,
    tau=None,
    fmin=None,
    fmax=None,
    echelle_type="single",
    plot_with="imshow",
):
    """
    Generate a (stretched) period echelle plot used in asteroseismology.

    Parameters
    ----------
    nu : 1D array-like
        Frequencies in microhertz (μHz).
    ps : 1D array-like
        Power spectrum.
    ΔΠ : float
        Length of each stack in seconds (s).
    tau : 1D array-like, optional
        Stretched period in seconds (s). Defaults to unstretched period echelle.
    fmin : float, optional
        Minimum frequency to be plotted. Defaults to None.
    fmax : float, optional
        Maximum frequency to be plotted. Defaults to None.
    echelle_type : str, optional
        Type of echelle diagram, either 'single' or 'replicated'. Defaults to 'single'.
    plot_with : str, optional
        Plotting method, either 'imshow' or 'contour'. Defaults to 'imshow'.


    Returns
    -------
    z : 2D numpy.array
        Folded power spectrum.
    extent : list
        Edges of the plot in the format [left, right, bottom, top].
    x : 1D numpy.array
        x-coordinates. Only return when 'plot_with="contour"'.
    y : 1D numpy.array
        y-coodinates. Only return when 'plot_with="contour"'.

    Notes
    -----
    The function supports two types of plotting methods: 'imshow' for a fast rendering
    suitable for interactive use, and 'contour' for a more accurate rendering suitable
    for publication-quality plots. 'imshow' is also accurate if no stretching involved.

    Examples
    --------
    Example Usage 1:
    Using 'plt.imshow' for fast interactive plotting:
        z, ext = period_echelle(nu, ps, ΔΠ, tau=tau, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
        plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')

    Example Usage 2:
    Using 'plt.contour' for accurate, publication-quality plotting:
        z, x, y = period_echelle(nu, ps, ΔΠ, tau=tau, fmin=numax-4*Dnu, fmax=numax+4*Dnu, plot_with='contour')
        plt.contour(x, y, z, cmap='gray_r', levels=500)

    """

    if fmin is None:
        fmin = np.nanmin(nu)
    if fmax is None:
        fmax = np.nanmax(nu)

    if tau is None:
        tau = np.copy(1 / (nu * 1e-6))

    # trimming
    m = (nu > fmin) & (nu < fmax)
    nu, ps, tau = nu[m], ps[m], tau[m]

    # find the loci (index) of turning points to define stack
    idx = np.unique(
        np.concatenate([[0], np.where(np.diff((tau) % ΔΠ) > 0)[0], [len(tau) - 2]])
    )

    # define plotting elements
    resolution = np.median(np.abs(np.diff((1 / (nu * 1e-6)))))
    # number of vertical stacks
    n_stack = len(idx) - 1
    # number of point per stack
    n_element = int(np.ceil(ΔΠ / resolution))

    # make z
    base, z = make_fold(
        tau, ps, ΔΠ, n_stack, n_element, idx, echelle_type=echelle_type, reverse=True
    )

    # format output
    if plot_with == "imshow":
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu))
        return z, extent
    elif plot_with == "contour":
        x = base
        y = np.array(
            [np.median(nu[idx[istack] : idx[istack + 1]]) for istack in range(n_stack)]
        )
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0]) / 2
        yu = y + np.diff(y, append=y[-1]) / 2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y
    else:
        return None


def frequency_echelle(
    nu, ps, Δν, f=None, fmin=None, fmax=None, echelle_type="single", plot_with="imshow"
):
    """
    Generate a (stretched) frequency echelle plot used in asteroseismology.

    Parameters
    ----------
    nu : 1D array-like
        Frequencies in microhertz (μHz).
    ps : 1D array-like
        Power spectrum.
    Δν : float
        Length of each stack in microhertz (μHz).
    f : 1D array-like, optional
        Stretched frequency in seconds (μHz). Defaults to unstretched frequency echelle.
    fmin : float, optional
        Minimum frequency to be plotted. Defaults to None.
    fmax : float, optional
        Maximum frequency to be plotted. Defaults to None.
    echelle_type : str, optional
        Type of echelle diagram, either 'single' or 'replicated'. Defaults to 'single'.
    plot_with : str, optional
        Plotting method, either 'imshow' or 'contour'. Defaults to 'imshow'.


    Returns
    -------
    z : 2D numpy.array
        Folded power spectrum.
    extent : list
        Edges of the plot in the format [left, right, bottom, top].
    x : 1D numpy.array
        x-coordinates. Only return when 'plot_with="contour"'.
    y : 1D numpy.array
        y-coodinates. Only return when 'plot_with="contour"'.

    Notes
    -----
    The function supports two types of plotting methods: 'imshow' for a fast rendering
    suitable for interactive use, and 'contour' for a more accurate rendering suitable
    for publication-quality plots. 'imshow' is also accurate if no stretching involved.

    Examples
    --------
    Example Usage 1:
    Using 'plt.imshow' for fast interactive plotting:
        z, ext = frequency_echelle(nu, ps, Δν, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
        plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')

    Example Usage 2:
    Using 'plt.contour' for accurate, publication-quality plotting:
        z, x, y = frequency_echelle(nu, ps, Δν, fmin=numax-4*Dnu, fmax=numax+4*Dnu, plot_with='contour')
        plt.contour(x, y, z, cmap='gray_r', levels=500)

    """

    if fmin is None:
        fmin = np.nanmin(nu)
    if fmax is None:
        fmax = np.nanmax(nu)

    if f is None:
        f = np.copy(nu)

    fmin = 1e-4 if fmin < Δν else fmin - (fmin % Δν)

    # trimming
    m = (nu > fmin) & (nu < fmax)
    nu, ps, f = nu[m], ps[m], f[m]

    # find the loci (index) of turning points to define stack
    idx = np.unique(
        np.concatenate([[0], np.where(np.diff((f) % Δν) < 0)[0], [len(f) - 2]])
    )

    # define plotting elements
    resolution = np.median(np.abs(np.diff(nu)))
    # number of vertical stacks
    n_stack = len(idx) - 1
    # number of point per stack
    n_element = int(np.ceil(Δν / resolution))

    # make z
    base, z = make_fold(
        f, ps, Δν, n_stack, n_element, idx, echelle_type=echelle_type, reverse=False
    )

    # format output
    if plot_with == "imshow":
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu))
        return z, extent
    elif plot_with == "contour":
        x = base
        y = np.array(
            [np.median(nu[idx[istack] : idx[istack + 1]]) for istack in range(n_stack)]
        )
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0]) / 2
        yu = y + np.diff(y, append=y[-1]) / 2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y
    else:
        return None


def ε_p(ν, params, constant_ε_p=False):
    if constant_ε_p:
        return params["ε_p"]
    else:
        return (
            params["α_p"] * ((ν - params["ν_max"]) / params["Δν"]) ** 2.0
            + params["ε_p"]
        )


def q(ν, params, constant_q=False):
    if constant_q:
        return params["q"]
    else:
        return params["q_k"] * (ν - params["ν_max"]) + params["q"]


def make_f(ν, params, constant_q=False):
    Theta_g = np.pi * (params["ε_g"] - 1 / (ν * 1e-6) / params["ΔΠ1"])
    return ν - params["Δν"] / np.pi * np.arctan(
        q(ν, params, constant_q=constant_q) / np.tan(Theta_g)
    )


def make_τ(ν, params, constant_q=False, constant_ε_p=False, constant_d01=True):
    Theta_p = np.pi * (
        ν / params["Δν"]
        - (
            1 / 2
            + ε_p(ν, params, constant_ε_p=constant_ε_p)
            + d01(ν, params, constant_d01=constant_d01)
        )
    )
    return 1 / (ν * 1e-6) + params["ΔΠ1"] / np.pi * np.arctan(
        q(ν, params, constant_q=constant_q) / np.tan(Theta_p)
    )


def d01(ν, params, constant_d01=True):
    if constant_d01:
        return params["d01"]
    else:
        return params["d01"] * (params["ν_max"] / ν)


def plot_frequency_echelle(
    freq, power, Dnu, f=None, plot_with="imshow", ax=None, cmap="gray_r", **kwargs
):

    if ax is None:
        fig, ax = plt.subplots()

    pack = frequency_echelle(freq, power, Dnu, f=f, plot_with=plot_with, **kwargs)

    if plot_with == "imshow":
        z, extent = pack
        ax.imshow(z, extent=extent, aspect="auto", interpolation="nearest", cmap=cmap)
    elif plot_with == "contour":
        z, x, y = pack
        ax.contour(x, y, z, levels=500, cmap=cmap)
    else:
        raise ValueError("Invalid plot_with option. Choose 'imshow' or 'contour'.")

    ax.set_xlabel(r"Frequency" + " mod " + str(Dnu))
    ax.set_ylabel(r"Frequency")

    return ax


def plot_period_echelle(
    freq, power, DP, tau=None, plot_with="imshow", ax=None, cmap="gray_r", **kwargs
):

    if ax is None:
        fig, ax = plt.subplots()

    pack = period_echelle(freq, power, DP, tau=tau, plot_with=plot_with, **kwargs)

    if plot_with == "imshow":
        z, extent = pack
        ax.imshow(z, extent=extent, aspect="auto", interpolation="nearest", cmap=cmap)
    elif plot_with == "contour":
        z, x, y = pack
        ax.contour(x, y, z, levels=500, cmap=cmap)
    else:
        raise ValueError("Invalid plot_with option. Choose 'imshow' or 'contour'.")

    ax.set_xlabel(r"Period" + " mod " + str(DP))
    ax.set_ylabel(r"Frequency")

    return ax
