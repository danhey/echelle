from __future__ import division, print_function

from .echelle import echelle
from .utils import smooth_power

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets import FloatSlider, VBox, HBox, Label, interactive_output
from IPython.display import display
import numpy as np
from . import stretching as st

__all__ = ["interact_echelle", "interact_stretched_echelle"]


def make_spectrum_echelles(nu, ps, params):
    echelle_kwargs = {"echelle_type": "replicated", "plot_with": "imshow"}
    τ = st.make_τ(nu, params)
    f = st.make_f(nu, params)
    return (
        st.frequency_echelle(nu, ps, params["Δν"], **echelle_kwargs),
        st.period_echelle(nu, ps, params["ΔΠ1"], tau=τ, **echelle_kwargs),
        st.frequency_echelle(nu, ps, params["Δν"], f=f, **echelle_kwargs),
    )


def make_ν_m_echelles(ν_m, params):
    f = st.make_f(ν_m, params)
    τ = st.make_τ(ν_m, params)
    return (
        (
            np.r_[ν_m % params["Δν"], ν_m % params["Δν"] + params["Δν"]],
            np.r_[ν_m, ν_m - params["Δν"]],
        ),
        (np.r_[τ % params["ΔΠ1"], τ % params["ΔΠ1"] + params["ΔΠ1"]], np.r_[ν_m, ν_m]),
        (
            np.r_[f % params["Δν"], f % params["Δν"] + params["Δν"]],
            np.r_[ν_m, ν_m - params["Δν"]],
        ),
    )


def make_ν_p_echelles(ν_p, params):
    return (
        np.r_[ν_p % params["Δν"], ν_p % params["Δν"] + params["Δν"]],
        np.r_[ν_p, ν_p - params["Δν"]],
    )


def make_ν_g_echelles(ν_g, params):
    return (
        np.r_[(1e6 / ν_g) % params["ΔΠ1"], (1e6 / ν_g) % params["ΔΠ1"] + params["ΔΠ1"]],
        np.r_[ν_g, ν_g],
    )


def make_ν_0_echelles(ν_0, params):
    return (
        np.r_[ν_0 % params["Δν"], ν_0 % params["Δν"] + params["Δν"]],
        np.r_[ν_0, ν_0 - params["Δν"]],
    )


def make_vlines(params):
    return (
        ((params["ε_p"] % 1) * params["Δν"], (params["ε_p"] % 1 + 1.0) * params["Δν"]),
        (
            (params["ε_g"] % 1) * params["ΔΠ1"],
            ((params["ε_g"] % 1) + 1.0) * params["ΔΠ1"],
        ),
        (
            ((params["ε_p"] + 0.5 + params["d01"]) % 1) * params["Δν"],
            ((params["ε_p"] + 0.5 + params["d01"]) % 1 + 1) * params["Δν"],
        ),
    )


def interact_stretched_echelle(
    params,
    nu=None,
    ps=None,
    ν_m=None,
    ν_p=None,
    ν_g=None,
    ν_0=None,
    show_vlines=False,
):

    params = params  # AttrDict({'ΔΠ1': 84.84, 'q': 0.31, 'q_k':0., 'Δν':17.277, 'ε_p':0.289, 'ε_g':0.7, 'd01': 0.00, 'α_p': 0., 'ν_max':0.})

    flag_spectra = False  # an array of l=1 π mode frequencies
    flag_ν_m = False  # an array of l=1 mixed mode frequencies
    flag_ν_p = False  # an array of l=1 π mode frequencies
    flag_ν_g = False  # an array of l=1 γ mode frequencies
    flag_ν_0 = False  # an array of l=0 mode frequencies

    if (nu is not None) and (ps is not None):
        nu = nu
        ps = ps
        flag_spectra = True

    if ν_m is not None:
        ν_m = ν_m
        flag_ν_m = True

    if ν_p is not None:
        ν_p = ν_p
        flag_ν_p = True

    if ν_g is not None:
        ν_g = ν_g
        flag_ν_g = True

    if ν_0 is not None:
        ν_0 = ν_0
        flag_ν_0 = True

    fig, ax = plt.subplots(figsize=[10, 5], nrows=1, ncols=3, squeeze=False)
    ax = ax.reshape(-1)

    # initialize plots
    if flag_spectra:
        images = []
        for i, (z, ext) in enumerate(make_spectrum_echelles(nu, ps, params)):
            images.append(
                ax[i].imshow(
                    z, extent=ext, aspect="auto", interpolation="nearest", cmap="gray_r"
                )
            )

    if flag_ν_m:
        scm = []
        for i, (x, y) in enumerate(make_ν_m_echelles(ν_m, params)):
            scm.append(ax[i].scatter(x, y, label="mixed", marker="^", color="C0"))

    if flag_ν_g:
        scγ = ax[1].scatter(
            *make_ν_g_echelles(ν_g, params), label="γ", marker="^", color="C1"
        )

    if flag_ν_p:
        scπ = ax[2].scatter(
            *make_ν_p_echelles(ν_p, params), label="π", marker="^", color="C2"
        )

    if flag_ν_0:
        sc0 = []
        for i in [0, 2]:
            sc0.append(
                ax[i].scatter(
                    *make_ν_0_echelles(ν_0, params),
                    label="l=0",
                    marker="o",
                    edgecolor="k",
                    facecolor="none",
                )
            )

    # add lines (predictions based on current parameters)
    if show_vlines:
        vlines = []
        for i, vals in enumerate(make_vlines(params)):
            vlines.append(
                [
                    ax[i].axvline(val, color="b", linestyle=":", alpha=0.5)
                    for val in vals
                ]
            )

    # labels and stuff
    ax[0].set_xlabel(r"$\nu$ mod $\Delta\nu$ [$\mu$Hz]")
    ax[0].set_ylabel(r"$\nu$ [$\mu$Hz]")
    ax[1].set_xlabel(r"$\tau$ mod $\Delta\Pi_1$ [s]")
    # ax[1].set_ylabel(r'$\nu$ [$\mu$Hz]')
    ax[2].set_xlabel(r"$f$ mod $\Delta\nu$ [$\mu$Hz]")
    # ax[2].set_ylabel(r'$\nu$ [$\mu$Hz]')

    ax[0].set_xlim(0, params["Δν"] * 2.0)
    ax[1].set_xlim(0, params["ΔΠ1"] * 2.0)
    ax[2].set_xlim(0, params["Δν"] * 2.0)

    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(bottom=0.40)

    # make horizontal sliders to control the parameters
    xstart, width, height, ystart = 0.15, 0.7, 0.025, 0.05

    ΔΠ1_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 0 * 0.04, width, height]),
        label=r"$\Delta\Pi_1$",
        valmin=40,
        valmax=360,
        valinit=params["ΔΠ1"],
    )
    q_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 1 * 0.04, width, height]),
        label=r"$q$",
        valmin=0.0,
        valmax=1.0,
        valinit=params["q"],
    )
    q_k_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 2 * 0.04, width, height]),
        label=r"$q_k$",
        valmin=-1e-2,
        valmax=1e-2,
        valinit=params["q_k"],
    )
    d01_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 3 * 0.04, width, height]),
        label=r"$d_{01}$",
        valmin=-0.25,
        valmax=0.25,
        valinit=params["d01"],
    )
    Δν_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 4 * 0.04, width, height]),
        label=r"$\Delta\nu$",
        valmin=params["Δν"] * 0.8,
        valmax=params["Δν"] * 1.2,
        valinit=params["Δν"],
    )
    ε_p_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 5 * 0.04, width, height]),
        label=r"$\epsilon_p$",
        valmin=0.0,
        valmax=2.0,
        valinit=params["ε_p"],
    )
    ε_g_slider = Slider(
        ax=fig.add_axes([xstart, ystart + 6 * 0.04, width, height]),
        label=r"$\epsilon_g$",
        valmin=0.0,
        valmax=1.0,
        valinit=params["ε_g"],
    )

    # define the function to be called anytime a slider's value changes
    def update(val):
        pu = {
            "ΔΠ1": ΔΠ1_slider.val,
            "q": q_slider.val,
            "q_k": q_k_slider.val,
            "d01": d01_slider.val,
            "ε_g": ε_g_slider.val,
            "Δν": Δν_slider.val,
            "ε_p": ε_p_slider.val,
            "α_p": params["α_p"],
            "ν_max": params["ν_max"],
        }

        # modify plots
        if flag_spectra:
            for i, (z, ext) in enumerate(make_spectrum_echelles(nu, ps, pu)):
                images[i].set(array=z, extent=ext)

        if flag_ν_m:
            for i, (x, y) in enumerate(make_ν_m_echelles(ν_m, pu)):
                scm[i].set_offsets(np.c_[(x, y)])

        if flag_ν_g:
            scγ.set_offsets(np.c_[*make_ν_g_echelles(ν_g, pu)])

        if flag_ν_p:
            scπ.set_offsets(np.c_[*make_ν_p_echelles(ν_p, pu)])

        if flag_ν_0:
            for i in range(2):
                sc0[i].set_offsets(np.c_[*make_ν_0_echelles(ν_0, pu)])

        # modify lines
        if show_vlines:
            for i, vals in enumerate(make_vlines(pu)):
                vlines[i][0].set_xdata([vals[0], vals[0]])
                vlines[i][1].set_xdata([vals[1], vals[1]])

        ax[0].set_xlim(0, pu.Δν * 2.0)
        ax[1].set_xlim(0, pu.ΔΠ1 * 2.0)
        ax[2].set_xlim(0, pu.Δν * 2.0)

        fig.canvas.draw_idle()

    # register the update function with each slider
    for slider in [
        ΔΠ1_slider,
        q_slider,
        q_k_slider,
        d01_slider,
        Δν_slider,
        ε_p_slider,
        ε_g_slider,
    ]:
        slider.on_changed(update)

    plt.show()


def interact_echelle(
    freq,
    power,
    dnu_min,
    dnu_max,
    step=None,
    cmap="BuPu",
    ax=None,
    smooth=False,
    stepsize=10,
    smooth_filter_width=50.0,
    scale=None,
    return_coords=False,
    backend="matplotlib",
    notebook_url="localhost:8888",
    plot_method="fast",
    **kwargs,
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
        the frequency spacing
    cmap : matplotlib.colormap, optional
        Colormap for the echelle diagram, by default 'BuPu'
    ax : matplotlib.axis, optional
        axis object on which to plot. If none is passed, one will be created,
        by default None
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
    plot_method : str, 'fast' or 'slow'
        Uses either the fast `pcolormesh` function, or the slower `imshow`
        function. pcolormesh is much faster, but does not support
        interpolation.
    sampling: float
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

    if smooth:
        power = smooth_power(power, smooth_filter_width)

    if "fmin" in kwargs:
        fmin = kwargs["fmin"]
    else:
        fmin = freq[0]
    if "fmax" in kwargs:
        fmax = kwargs["fmax"]
    else:
        fmax = freq[-1]

    dnu_initial = (dnu_max + dnu_min) / 2.0
    z = echelle(
        freq,
        power,
        dnu_initial,
        **kwargs,
    )

    if scale == "sqrt":
        z = np.sqrt(z)
    elif scale == "log":
        z = np.log10(z)

    if step == None:
        step = stepsize * np.median(np.diff(freq))
    if backend == "matplotlib":
        # Create the matplotlib version of the interactive
        # form.
        if ax is None:
            fig, ax = plt.subplots(figsize=[4, 6])
        else:
            fig = plt.gcf()

        if plot_method == "fast":
            line = ax.pcolorfast((0, dnu_initial), (fmin, fmax), z, cmap=cmap)
        else:
            line = ax.imshow(
                z,
                aspect="auto",
                extent=(0, dnu_initial, fmin, fmax),
                origin="lower",
                cmap=cmap,
            )
        line.set_extent((0, dnu_initial, fmin, fmax))
        ax.set_xlim(0, dnu_initial)

        axfreq = plt.axes([0.1, 0.025, 0.8, 0.02])
        valfmt = -int(np.floor(np.log10(abs(step))))
        slider = Slider(
            axfreq,
            "\u0394\u03bd",
            dnu_min,
            dnu_max,
            valinit=(dnu_max + dnu_min) / 2.0,
            valstep=step,
            valfmt=f"%1.{valfmt}f",
        )

        def update(dnu):
            z = echelle(freq, power, dnu, **kwargs)
            if scale is not None:
                if scale == "sqrt":
                    z = np.sqrt(z)
                elif scale == "log":
                    z = np.log10(z)
            line.set_array(z)
            line.set_extent((0, dnu, fmin, fmax))
            ax.set_xlim(0, dnu)
            fig.canvas.draw_idle()
            # fig.canvas.blit(ax.bbox)

        def on_key_press(event):
            if event.key == "left":
                new_dnu = slider.val - slider.valstep
            elif event.key == "right":
                new_dnu = slider.val + slider.valstep
            else:
                new_dnu = slider.val

            slider.set_val(new_dnu)
            update(new_dnu)

        # def on_click(event):
        #     ix, iy = event.xdata, event.ydata
        #     coords.append((ix, iy))

        fig.canvas.mpl_connect("key_press_event", on_key_press)
        slider.on_changed(update)

        ax.set_xlabel("Frequency mod \u0394\u03bd")
        ax.set_ylabel("Frequency")
        plt.subplots_adjust(
            left=0.1,
            right=0.95,
            bottom=0.1,
            top=0.95,
        )
        plt.show()

        # if return_coords:
        #     coords = []
        #     fig.canvas.mpl_connect("button_press_event", on_click)
        #     return coords

    elif backend == "bokeh":
        # Otherwise we use Bokeh.
        try:
            import bokeh
        except:
            raise ImportError("You need to install the Bokeh package.")

        from bokeh.io import show, output_notebook
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.layouts import column
        from bokeh.models import ColumnDataSource, FreehandDrawTool
        from bokeh.models import Slider as b_Slider

        import warnings
        from bokeh.util.warnings import BokehUserWarning

        warnings.simplefilter("ignore", BokehUserWarning)

        fmin, fmax = freq.min(), freq.max()

        def create_interact_ui(doc):
            source = ColumnDataSource(
                data={
                    "image": [z],
                    "x": x,
                    "y": y,
                    "dw": [x.max() - x.min()],
                    "dh": [y.max() - y.min()],
                }
            )

            plot = figure(
                x_range=(x.min(), x.max()),
                y_range=(y.min(), y.max()),
                width=600,
                height=700,
                resizable=True,
                # plot_width=550,
                # plot_height=600,
            )

            palette = get_bokeh_palette(cmap)

            full_plot = plot.image(
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh=y.max() - y.min(),
                source=source,
                palette=palette,
            )

            plot.xaxis.axis_label = "Frequency mod \u0394\u03bd"
            plot.yaxis.axis_label = "Frequency"

            slider = b_Slider(
                start=dnu_min,
                end=dnu_max,
                value=(dnu_min + dnu_max) / 2,
                step=step,
                title="\u0394\u03bd",
                format="0.000",
            )

            # Slider callback
            def update_upon_dnu_change(attr, old, new):
                x, y, z = echelle(
                    freq,
                    power,
                    new,
                    fmin=fmin,
                    fmax=fmax,
                    sampling=sampling,
                )
                if scale is not None:
                    if scale == "sqrt":
                        z = np.sqrt(z)
                    elif scale == "log":
                        z = np.log10(z)
                full_plot.data_source.data["image"] = [z]
                full_plot.data_source.data["dw"] = [x.max() - x.min()]
                plot.x_range.start = x.min()
                plot.x_range.end = x.max()

            slider.on_change("value", update_upon_dnu_change)

            # Adjust some toolbar options
            r = plot.multi_line(line_width=15, alpha=0.2, color="red")
            plot.add_tools(FreehandDrawTool(renderers=[r]))
            plot.toolbar.logo = None
            plot.toolbar.active_drag = None

            # Layout all of the plots
            widgets_and_figures = column(slider, plot)
            doc.add_root(widgets_and_figures)

        output_notebook(verbose=False, hide_banner=True)
        return show(create_interact_ui, notebook_url=notebook_url)

    else:
        raise ValueError("'backend' must be either 'matplotlib' or 'bokeh")


def get_bokeh_palette(cmap):
    """Creates a color palette compatible with Bokeh
    from a matplotlib cmap name.

    Parameters
    ----------
    cmap : string
        Name of a matplotlib colormap

    Returns
    -------
    list
        A series of hex colour codes generated from
        the matplotlib colormap
    """
    from bokeh.colors import RGB
    from matplotlib import cm

    # Solution adapted from
    # https://stackoverflow.com/questions/31883097/elegant-way-to-match-a-string-to-a-random-color-matplotlib
    m_RGB = (255 * plt.get_cmap(cmap)(range(256))).astype("int")
    return [RGB(*tuple(rgb)).to_hex() for rgb in m_RGB]
