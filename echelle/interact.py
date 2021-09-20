from __future__ import division, print_function

from .echelle import echelle
from .utils import smooth_power

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

__all__ = ["interact_echelle"]


def interact_echelle(
    freq,
    power,
    dnu_min,
    dnu_max,
    step=None,
    cmap="BuPu",
    ax=None,
    smooth=False,
    smooth_filter_width=50.0,
    scale=None,
    return_coords=False,
    backend="bokeh",
    notebook_url="localhost:8888",
    plot_method='fast',
    sampling=2,
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

    x, y, z = echelle(freq, power, (dnu_max + dnu_min) / 2.0, sampling=sampling, **kwargs)

    if scale is "sqrt":
        z = np.sqrt(z)
    elif scale is "log":
        z = np.log10(z)

    if step is None:
        step = 5*np.median(np.diff(freq))
    if backend == "matplotlib":
        # Create the matplotlib version of the interactive
        # form. This should only really be done
        # if the user is working from the terminal.
        if ax is None:
            fig, ax = plt.subplots(figsize=[6.4, 9])
        else:
            fig = plt.gcf()
        
        if plot_method is 'fast':
            line = ax.pcolorfast((x.min(), x.max()),
                                (y.min(), y.max()), 
                                z, 
                                cmap=cmap)
        else:
            line = ax.imshow(
                z,
                aspect="auto",
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin="lower",
                cmap=cmap
            )
        
        axfreq = plt.axes([0.1, 0.025, 0.8, 0.02])
        valfmt = "%1." + str(len(str(step).split(".")[-1])) + "f"
        print(valfmt)
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
            x, y, z = echelle(freq, power, dnu, sampling=1, **kwargs)
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
        plt.subplots_adjust(
            left=0.1,
            right=0.95,
            bottom=0.1,
            top=0.95,
        )
        plt.show()

        if return_coords:
            coords = []
            fig.canvas.mpl_connect("button_press_event", on_click)
            return coords

    elif backend == "bokeh":
        # Otherwise we use Bokeh.
        try:
            import bokeh
        except:
            raise ImportError("You need to install the Bokeh package.")

        from bokeh.io import show, output_notebook, push_notebook
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.layouts import column
        from bokeh.models import CustomJS, ColumnDataSource, FreehandDrawTool
        from bokeh.models import Slider as b_Slider

        import warnings
        from bokeh.util.warnings import BokehUserWarning

        # This is a terrible hack and I hate Bokeh
        warnings.simplefilter("ignore", BokehUserWarning)

        from notebook import notebookapp

        # Make sure the user knows to pass in the right notebook_url.
        servers = list(notebookapp.list_running_servers())
        ports = [s["port"] for s in servers]
        if len(np.unique(ports)) > 1:
            warnings.warn(
                "You have multiple Jupyter servers open. \
            You will need to pass the current notebook to `notebook_url`. \
            i.e. interact_echelle(x,y,notebook_url='http://localhost:8888')",
                UserWarning,
            )

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
                plot_width=550,
                plot_height=600
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

            plot.xaxis.axis_label = u"Frequency mod \u0394\u03BD"
            plot.yaxis.axis_label = "Frequency"

            slider = b_Slider(
                start=dnu_min,
                end=dnu_max,
                value=(dnu_min + dnu_max) / 2,
                step=step,
                title=u"\u0394\u03BD",
                format="0.000",
            )

            # Slider callback
            def update_upon_dnu_change(attr, old, new):
                x, y, z = echelle(
                    freq,
                    power,
                    new,
                    sampling=sampling,
                )
                if scale is not None:
                    if scale is "sqrt":
                        z = np.sqrt(z)
                    elif scale is "log":
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