from __future__ import print_function, division

import warnings

import numpy as np
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

__all__ = ["echelle", "plot_echelle", "interact_echelle", "smooth_power"]


def echelle(freq, power, dnu, fmin=0.0, fmax=None, offset=0.0, sampling=0.1):
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
    array-like
        The x, y, and z values of the echelle diagram.
    """
    if fmax is None:
        fmax = freq[-1]

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    # trim data
    index = (freq >= fmin) & (freq <= fmax)
    trimx = freq[index]

    samplinginterval = np.median(trimx[1:-1] - trimx[0:-2]) * sampling
    xp = np.arange(fmin, fmax + dnu, samplinginterval)
    yp = np.interp(xp, freq, power)

    n_stack = int((fmax - fmin) / dnu)
    n_element = int(dnu / samplinginterval)

    morerow = 2
    arr = np.arange(1, n_stack) * dnu
    arr2 = np.array([arr, arr])
    yn = np.reshape(arr2, len(arr) * 2, order="F")
    yn = np.insert(yn, 0, 0.0)
    yn = np.append(yn, n_stack * dnu) + fmin + offset

    xn = np.arange(1, n_element + 1) / n_element * dnu
    z = np.zeros([n_stack * morerow, n_element])
    for i in range(n_stack):
        for j in range(i * morerow, (i + 1) * morerow):
            z[j, :] = yp[n_element * (i) : n_element * (i + 1)]
    return xn, yn, z


def plot_echelle(
    freq,
    power,
    dnu,
    mirror=False,
    ax=None,
    cmap="BuPu",
    scale=None,  # "sqrt",
    interpolation=None,
    smooth=False,
    smooth_filter_width=50,
    **kwargs
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
    echx, echy, echz = echelle(freq, power, dnu, **kwargs)

    if scale is not None:
        if scale is "log":
            echz = np.log10(echz)
        elif scale is "sqrt":
            echz = np.sqrt(echz)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(
        echz,
        aspect="auto",
        extent=(echx.min(), echx.max(), echy.min(), echy.max()),
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
    )

    # It's much cheaper just to replot the data we already have
    if mirror:
        ax.imshow(
            echz,
            aspect="auto",
            extent=(
                (echx.min() + dnu),
                (echx.max() + dnu),
                (echy.min() - dnu),
                (echy.max()) - dnu,
            ),
            origin="lower",
            cmap=cmap,
            interpolation=interpolation,
        )

    ax.set_xlabel(r"Frequency" + " mod " + str(dnu))
    ax.set_ylabel(r"Frequency")
    return ax


def interact_echelle(
    freq,
    power,
    dnu_min,
    dnu_max,
    step=0.01,
    cmap="BuPu",
    ax=None,
    interpolation=None,
    smooth=False,
    smooth_filter_width=50.0,
    scale=None,  # "sqrt",
    return_coords=False,
    **kwargs
):
    """Creates an interactive echelle environment with a variable deltanu 
    slider. If you're working in a Jupyter notebook/lab environment, you must
    call `%matplotlib notebook` before running this.
    
    Parameters
    ----------
    freq : np.array
        Array of frequencies in the amplitude or power spectrum
    power : np.array
        Corresponding array of power values
    dnu_min : float
        Minimum deltanu value for the slider
    dnu_max : float
        Maximum deltanu value for the slider
    step : float, optional
        Step size by which to increment or decrement the slider, by default 
        0.01
    cmap : matplotlib.colormap, optional
        Colormap for the echelle diagram, by default 'BuPu'
    ax : matplotlib.axis, optional
        axis object on which to plot. If none is passed, one will be created, 
        by default None
    interpolation : str, optional
        Type of interpolation to perform on the echelle diagram through 
        matplotlib.pyplot.imshow. This is very expensive in an interactive 
        environment, so use with caution, by default 'none'
    smooth_filter_width : float, optional
        Size of the Box1DKernel which is convolved with the power to smooth the
        spectrum. 1 performs no smoothing, by default 50.
    scale : str, optional
        either 'sqrt' or 'log' or None. Scales the echelle to bring out more 
        features, by default 'sqrt'
    return_coords : bool, optional
        If True, this will bind mouseclick events to the interactive plot. 
        Clicking on the plot will store the values of the frequencies
        at the click event, and return them in a list object, by default False
    **kwargs : dict
        Dictionary of arguments to be passed to `echelle.echelle`
    
    Returns
    -------
    list
        A list of clicked frequencies if `return_coords=True`.
    """

    if dnu_max < dnu_min:
        raise ValueError("Maximum range can not be less than minimum")

    if smooth_filter_width < 1:
        raise ValueError("The smooth filter width can not be less than 1!")

    if ax is None:
        fig, ax = plt.subplots()
    if smooth:
        power = smooth_power(power, smooth_filter_width)

    x, y, z = echelle(freq, power, (dnu_max + dnu_min) / 2.0, sampling=1, **kwargs)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    if scale is "sqrt":
        z = np.sqrt(z)
    elif scale is "log":
        z = np.log10(z)

    line = ax.imshow(
        z,
        aspect="auto",
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        cmap=cmap,
        interpolation=interpolation,
    )

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    valfmt = "%1." + str(len(str(step).split(".")[-1])) + "f"
    slider = Slider(
        axfreq,
        u"\u0394\u03BD",
        dnu_min,
        dnu_max,
        valinit=(dnu_max + dnu_min) / 2.0,
        valstep=step,
        valfmt=valfmt,
    )

    def update(dnu):
        x, y, z = echelle(freq, power, dnu, **kwargs)
        if scale is not None:
            if scale is "sqrt":
                z = np.sqrt(z)
            elif scale is "log":
                z = np.log10(z)
        line.set_array(z)
        line.set_extent((x.min(), x.max(), y.min(), y.max()))
        ax.set_xlim(0, dnu)
        fig.canvas.blit(ax.bbox)

    def on_key_press(event):
        if event.key == "left":
            new_dnu = slider.val - slider.valstep
        elif event.key == "right":
            new_dnu = slider.val + slider.valstep
        else:
            new_dnu = slider.val

        slider.set_val(new_dnu)
        update(new_dnu)

    def on_click(event):
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    slider.on_changed(update)

    ax.set_xlabel(u"Frequency mod \u0394\u03BD")
    ax.set_ylabel("Frequency")
    plt.show()

    if return_coords:
        coords = []
        fig.canvas.mpl_connect("button_press_event", on_click)
        return coords


def smooth_power(power, smooth_filter_width):
    """Smooths the input power array with a Box1DKernel from astropy
    
    Parameters
    ----------
    power : array-like
        Array of power values
    smooth_filter_width : float
        filter width
    
    Returns
    -------
    array-like
        Smoothed power
    """
    return convolve(power, Box1DKernel(smooth_filter_width))


def plot_echelle_old(
    freq,
    power,
    dnu,
    ax=None,
    nlevels=32,
    cmap="Greys",
    scale="sqrt",
    offset=0.0,
    xmin=None,
    xmax=None,
    rasterized=True,
    **kwargs
):
    echx, echy, echz = echelle(freq, power, dnu, offset=offset, **kwargs)
    echx += offset

    if scale is "log":
        echz = np.log10(echz)
    elif scale is "sqrt":
        echz = np.sqrt(echz)

    if ax is None:
        fig, ax = plt.subplots()

    levels = np.linspace(np.min(echz), np.max(echz), nlevels)
    ax.contourf(echx, echy, echz, cmap=cmap, levels=levels, rasterized=rasterized)
    if xmax is not None:
        ax.contourf(
            echx + dnu,
            echy - dnu,
            echz,
            cmap=cmap,
            levels=levels,
            rasterized=rasterized,
        )
        ax.axis([xmin, xmax, np.min(echy), np.max(echy)])
    else:
        ax.axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])

    ax.set_xlabel(r"Frequency" + " mod " + str(dnu))
    ax.set_ylabel(r"Frequency")

    return ax
