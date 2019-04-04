from __future__ import print_function, division

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import get_backend

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

def interact_echelle(freq, power, dnu):
    """
    Plots an interactive echelle diagram with a variable dnu slider.
    Args:
    freq (array-like): Frequencies of the spectrum
    power (array-like): Power or amplitude of the spectrum
    dnu (float): Large separation value
    """

    from ipywidgets import interact
    mpl_is_inline = 'nbAgg' in get_backend()
    if not mpl_is_inline:
        print("You should be using %matplotlib notebook otherwise this won't work")

    x,y,z=echelle(freq, power, dnu, offset=0.0)
    fig, ax = plt.subplots(figsize=[7,7])
    line = ax.imshow(np.sqrt(z), aspect='auto', 
                    extent=(x.min(), x.max(), y.min(), y.max()), 
                    origin='lower',
                    cmap='gray_r')

    def update(dnu):
        x,y,z=echelle(freq, power, dnu, offset=0.0)
        line.set_array(np.sqrt(z))
        line.set_extent((x.min(), x.max(), y.min(), y.max()))
        ax.set_xlim(0,dnu)
        fig.canvas.draw()

    ax.set_xlabel(u'Frequency mod \u0394\u03BD')
    ax.set_ylabel('Frequency')
    
    interact(update, dnu=(dnu-2.,dnu+2.,0.01))