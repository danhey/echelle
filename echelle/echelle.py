from __future__ import print_function, division

import warnings

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["echelle", "plot_echelle", "interact_echelle"]

def echelle(freq, power, dnu, offset=0.0):

    if len(freq) != len(power): 
        raise ValueError("x and y must have equal size.")	

    fmin, fmax = freq[0], freq[-1]

    fmin = fmin - offset
    fmax = fmax - offset
    freq = freq - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % dnu)

    # trim data
    index = np.intersect1d(np.where(freq>=fmin)[0],np.where(freq<=fmax)[0])
    trimx = freq[index]

    samplinginterval = np.median(trimx[1:-1] - trimx[0:-2]) * 0.1
    xp = np.arange(fmin,fmax+dnu,samplinginterval)
    yp = np.interp(xp, freq, power)

    n_stack = int((fmax-fmin)/dnu)
    n_element = int(dnu/samplinginterval)

    morerow = 2
    arr = np.arange(1,n_stack) * dnu # + period/2.0
    arr2 = np.array([arr,arr])
    yn = np.reshape(arr2,len(arr)*2,order="F")
    yn = np.insert(yn,0,0.0)
    yn = np.append(yn,n_stack*dnu) + fmin + offset

    xn = np.arange(1,n_element+1)/n_element * dnu
    z = np.zeros([n_stack*morerow,n_element])
    # This should be vectorized for speed-up
    for i in range(n_stack):
        for j in range(i*morerow,(i+1)*morerow):
            z[j,:] = yp[n_element*(i):n_element*(i+1)]
    return xn, yn, z

def plot_echelle(freq, power, dnu, offset=0.0,
                 ax=None, nlevels=32, cmap='gray_r'):
    """
    Plots the echelle diagram for a given amplitude (or power) spectrum
    Args:
    freq (array-like): Frequencies of the spectrum
    power (array-like): Power or amplitude of the spectrum
    dnu (float): Large separation value
    offset (float): Amount by which to offset the echelle diagram
    """

    echx, echy, echz = echelle(freq, power, dnu, offset=0.0,)
    echz = np.sqrt(echz)
    if ax is None:
        fig, ax = plt.subplots()

    levels = np.linspace(np.min(echz),np.max(echz),nlevels)
    # Using levels is very slow for some reason.. 
    ax.contourf(echx,echy,echz,cmap=cmap,levels=levels)
    ax.axis([np.min(echx),np.max(echx),np.min(echy),np.max(echy)])
    
    ax.set_xlabel(r'Frequency' +' mod ' + str(dnu))
    ax.set_ylabel(r'Frequency')
    
    return ax

def interact_echelle(freq, power, dnu, notebook_url='localhost:8888'):
    """
    Plots an interactive echelle diagram with a variable dnu slider.
    This method requires Bokeh to run, and wil.
    Args:
    freq (array-like): Frequencies of the spectrum
    power (array-like): Power or amplitude of the spectrum
    dnu (float): Large separation value
    notebook_url (str): Current url of the notebook. Defaults to `localhost:8888`
    """
    try:
        import bokeh
    except:
        raise ImportError('Bokeh is definitely required for this.')

    from bokeh.io import show, output_notebook, push_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.palettes import grey
    from bokeh.layouts import column
    from bokeh.models import CustomJS, ColumnDataSource, Slider

    import warnings
    from bokeh.util.warnings import BokehUserWarning
    # This is a terrible hack and I hate Bokeh
    warnings.simplefilter("ignore", BokehUserWarning)

    from notebook import notebookapp
    servers = list(notebookapp.list_running_servers())
    ports = [s['port'] for s in servers]
    if len(np.unique(ports)) > 1:
        warnings.warn("You have multiple Jupyter servers open. \
        You will need to pass the current notebook to `notebook_url`. \
        i.e. interact_echelle(x,x,notebook_url='http://localhost:8888')",UserWarning)

    def create_interact_ui(doc):

        x, y, z = echelle(freq,power,dnu)
        source = ColumnDataSource(data={'image':[np.sqrt(z)] ,'x': x, 'y':y, 
                                        'dw':[x.max()-x.min()], 'dh':[y.max()-y.min()]})

        plot = figure(x_range=(x.min(), x.max()), y_range=(y.min(), y.max()))

        cmap = grey(256)[::-1]
        full_plot = plot.image(image='image', x='x', y='y', 
                dw='dw', dh=y.max()-y.min(), source=source,
                palette=cmap)

        plot.xaxis.axis_label=u'Frequency mod \u0394\u03BD'
        plot.yaxis.axis_label='Frequency'

        slider = Slider(start=dnu-2., end=dnu+2., value=dnu, step=.001, title=u"\u0394\u03BD")
            
        # Slider callback
        def update_upon_dnu_change(attr, old, new):
            x,y, z = echelle(freq,power,new)
            full_plot.data_source.data['image'] = [np.sqrt(z)]
            full_plot.data_source.data['dw'] = [x.max()-x.min()]
            #full_plot.data_source.data.update({'image' : [np.sqrt(z)],
            #                           'dw' : [x.max()-x.min()]})
            plot.x_range.start=x.min()
            plot.x_range.end=x.max()
        
        slider.on_change('value', update_upon_dnu_change)

        # Layout all of the plots
        widgets_and_figures = column(slider,plot)
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)