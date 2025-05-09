#!/usr/bin/env python3
# Yaguang Li

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider

def make_fold(nu, ps, period, n_stack, n_element, idx, echelle_type="single", reverse=False):
    if echelle_type=='single':
        z = np.zeros([n_stack, n_element])
        base = np.linspace(0, period, n_element, endpoint=False) 
        for istack in range(n_stack):
            z[-istack-1,:] = np.interp(base, nu[idx[istack]:idx[istack+1]], ps[idx[istack]:idx[istack+1]], period=period)
    else:
        z = np.zeros([n_stack, 2*n_element])
        base = np.linspace(0, 2*period, 2*n_element, endpoint=False) 
        for istack in range(n_stack-1):
            if reverse:
                z[-istack-1,:] = np.r_[np.interp(base[:n_element], nu[idx[istack+1]:idx[istack+2]], ps[idx[istack+1]:idx[istack+2]], period=period),
                                       np.interp(base[:n_element], nu[idx[istack]:idx[istack+1]], ps[idx[istack]:idx[istack+1]], period=period,), ]
            else:
                z[-istack-1,:] = np.r_[np.interp(base[:n_element], nu[idx[istack]:idx[istack+1]], ps[idx[istack]:idx[istack+1]], period=period),
                                       np.interp(base[:n_element], nu[idx[istack+1]:idx[istack+2]], ps[idx[istack+1]:idx[istack+2]], period=period,), ]
    return base, z


def plot_period_echelle(nu, ps, ΔΠ, tau=None, fmin=None, fmax=None, echelle_type='single', plot_with='imshow'):
    '''
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

    '''

    if fmin is None: fmin=np.nanmin(nu)
    if fmax is None: fmax=np.nanmax(nu)

    if tau is None: tau=np.copy(1/(nu*1e-6))

    # trimming
    m = (nu > fmin) & (nu < fmax)
    nu, ps, tau = nu[m], ps[m], tau[m] 

    # find the loci (index) of turning points to define stack
    idx = np.unique(np.concatenate([[0], np.where(np.diff((tau)%ΔΠ) > 0)[0], [len(tau)-2]]))

    # define plotting elements
    resolution = np.median(np.abs(np.diff((1/(nu*1e-6)))))
    # number of vertical stacks
    n_stack = len(idx) - 1 
    # number of point per stack
    n_element = int(np.ceil(ΔΠ/resolution))

    # make z
    base, z = make_fold(tau, ps, ΔΠ, n_stack, n_element, idx, echelle_type=echelle_type, reverse=True)

    # format output
    if plot_with=='imshow':
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu)) 
        return z, extent
    elif plot_with=='contour':
        x = base
        y = np.array([np.median(nu[idx[istack]:idx[istack+1]]) for istack in range(n_stack)])
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0])/2
        yu = y + np.diff(y, append=y[-1])/2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y 
    else:
        return None



def plot_frequency_echelle(nu, ps, Δν, f=None, fmin=None, fmax=None, echelle_type='single', plot_with='imshow'):
    '''
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

    '''

    if fmin is None: fmin=np.nanmin(nu)
    if fmax is None: fmax=np.nanmax(nu)

    if f is None: f=np.copy(nu)

    fmin = 1e-4 if fmin<Δν else fmin - (fmin % Δν)

    # trimming
    m = (nu > fmin) & (nu < fmax)
    nu, ps, f = nu[m], ps[m], f[m] 

    # find the loci (index) of turning points to define stack
    idx = np.unique(np.concatenate([[0], np.where(np.diff((f)%Δν) < 0)[0], [len(f)-2]]))

    # define plotting elements
    resolution = np.median(np.abs(np.diff(nu)))
    # number of vertical stacks
    n_stack = len(idx) - 1 
    # number of point per stack
    n_element = int(np.ceil(Δν/resolution))
    
    # make z
    base, z = make_fold(f, ps, Δν, n_stack, n_element, idx, echelle_type=echelle_type, reverse=False)

    # format output
    if plot_with=='imshow':
        extent = (0, np.max(base), np.nanmin(nu), np.nanmax(nu)) 
        return z, extent
    elif plot_with=='contour':
        x = base
        y = np.array([np.median(nu[idx[istack]:idx[istack+1]]) for istack in range(n_stack)])
        z = np.repeat(z, 2, axis=0)
        yl = y - np.diff(y, prepend=y[0])/2
        yu = y + np.diff(y, append=y[-1])/2
        y = np.sort(np.array([yl, yu]).reshape(-1))[::-1]
        return z, x, y 
    else:
        return None

def circvar(x, w=None):
    if w is None:
        return 1 - np.abs(np.sum(np.exp(1j * x))) / len(x)
    else:
        return 1 - np.abs(np.sum(w * np.exp(1j * x))) / np.sum(w)

def circstd(x, w=None):
    if w is None:
        return np.sqrt(-2 * np.log( np.abs(np.sum(np.exp(1j * x))) / len(x) ) ) 
    else:
        return np.sqrt(-2 * np.log( np.abs(np.sum(w * np.exp(1j * x))) / np.sum(w) ) )


def ε_p(ν, params, constant_ε_p=False):
    if constant_ε_p:
        return params['ε_p']
    else:
        return params['α_p'] *( (ν - params['ν_max'])/params['Δν'] )**2.0 + params['ε_p']

def q(ν, params, constant_q=False):
    if constant_q:
        return params['q']
    else:
        return params['q_k'] * (ν - params['ν_max']) + params['q']

def make_f(ν, params, constant_q=False):
    Theta_g = np.pi*(params['ε_g'] - 1/(ν*1e-6)/params['ΔΠ1'])
    return ν - params['Δν']/np.pi * np.arctan( q(ν, params, constant_q=constant_q) / np.tan( Theta_g ))

def make_τ(ν, params, constant_q=False, constant_ε_p=False, constant_d01=True):
    Theta_p = np.pi*(ν/params['Δν'] - (1/2 + ε_p(ν, params, constant_ε_p=constant_ε_p) + d01(ν, params, constant_d01=constant_d01)))
    return 1/(ν*1e-6) + params['ΔΠ1']/np.pi * np.arctan( q(ν, params, constant_q=constant_q) / np.tan(Theta_p) )


def make_ζ(ν, params, constant_ε_p=False, constant_d01=True, constant_q=False):
    Theta_p = np.pi*(ν/params['Δν'] - (1/2 + ε_p(ν, params, constant_ε_p=constant_ε_p) + d01(ν, params, constant_d01=constant_d01)))
    Theta_g = np.pi*(params['ε_g'] - 1/(ν*1e-6)/params['ΔΠ1'])
    q1 = q(ν, params, constant_q=constant_q)
    return 1/(1 + params['ΔΠ1'] / (params['Δν']*1e-6) * (ν*1e-6)**2 / q1 * np.sin(Theta_g)**2 / np.cos(Theta_p)**2)


def dfdε_g(ν, params, constant_q=False):
    q1 = q(ν, params, constant_q=constant_q)
    Theta_g = np.pi*(params['ε_g'] - 1/(ν*1e-6)/params['ΔΠ1']) 
    return (q1 * params['Δν'] * np.sin(Theta_g)**-2. ) / (1 + q1**2.0 * np.tan(Theta_g)**-2.)


def d01(ν, params, constant_d01=True):
    if constant_d01:
        return params['d01']
    else:
        return params['d01'] * (params['ν_max'] / ν)
    
def f_echelle(ν, params, constant_q=False):
    f = make_f(ν, params, constant_q=constant_q)
    return f % params['Δν'], ν

def τ_echelle(ν, params, constant_q=False, constant_ε_p=False):
    τ = make_τ(ν, params, constant_q=constant_q, constant_ε_p=constant_ε_p)
    return τ % params['ΔΠ1'], ν

def ν_echelle(ν, params):
    return ν % params['Δν'], ν

def P_echelle(ν, params):
    return (1e6/ν) % params['ΔΠ1'], ν

def get_radial_as(ν, n, ν_max,):
    '''
        ν = Δν (n + α_p * (n-n_max)^2 ) , where n_max = ν_max / Δν

        c_0 = Δν * α_p  # n^2
        c_1 = Δν - 2 * α_p * ν_max  # n
        c_2 = α_p * ν_max**2./Δν + Δν * ε_p  # 1
    '''

    # nmax = np.interp(ν_max, ν, n)
    # coeff = np.polyfit(n, ν, 2)
    # α_p = coeff[0]/coeff[1] / (1 + 2*nmax * coeff[0] / coeff[1])
    # Δν = coeff[0] / α_p
    # ε_p = coeff[2] / Δν - α_p * nmax**2.0

    # width estimates based on Yu+2018, Lund+2017, Li+2020
    k, b = 0.9638, -1.7145
    width = np.exp(k*np.log(ν_max) + b)
    w = np.exp(-(ν-ν_max)**2./(2*width**2.))
    # idx = w>1e-100

    coeff = np.polyfit(n, ν, 2, w=w)
    Δν = (coeff[1] + (coeff[1]**2. + 8*coeff[0]*ν_max)**0.5) / 2.
    α_p = coeff[0] / Δν 
    ε_p = (coeff[2] - α_p * ν_max**2. / Δν ) / Δν

    return Δν, ε_p, α_p


def get_model_Δν(ν, n, ν_max):
    # width estimates based on Yu+2018, Lund+2017, Li+2020
    k, b = 0.9638, -1.7145
    width = np.exp(k*np.log(ν_max) + b)
    w = np.exp(-(ν-ν_max)**2./(2*width**2.))
    idx = w>1e-100

    if np.sum(idx)>2:
        p, _, _, _, _ = np.polyfit(n[idx], ν[idx], 1, w=w[idx], full=True)
        Δν, ε_p = p[0], p[1]/p[0]
    else:
        Δν, ε_p = np.nan, np.nan 

    return Δν, ε_p


def get_model_δν01(freqs, ls, numax, Dnu):

    # Dnu_freq, _ = get_model_Dnu(freqs, ls, ns, numax)
    freq0s = np.sort(freqs[ls==0])
    freq1s = np.zeros(len(freq0s))

    for ifreq, freq0 in enumerate(freq0s):
        idx1 = (freqs>(freq0)) & (freqs<(freq0+Dnu)) & (ls==1)
        freq1s[ifreq] = np.nan if np.sum(idx1)==0 else freqs[idx1][np.argmin(freqs[idx1])]

    # width estimates based on Yu+2018, Lund+2017, Li+2020
    k, b = 0.9638, -1.7145
    width = np.exp(k*np.log(numax) + b)
    weight = np.exp(-(freq1s-numax)**2./(2*width**2.))
    idx = (freq1s < numax+3*width) & (freq1s > numax-3*width) & np.isfinite(weight)
    if np.sum(weight[idx]) == 0.:
        δν01 = np.nan 
    else:
        δν01 = np.average(freq0s[idx] + 0.5*Dnu - freq1s[idx], weights=weight[idx])
    return δν01

def get_model_δν02(freqs, ls, numax, Dnu):

    # Dnu_freq, _ = get_model_Dnu(freqs, ls, ns, numax)
    freq0s = np.sort(freqs[ls==0])
    freq2s = np.zeros(len(freq0s))

    for ifreq, freq0 in enumerate(freq0s):
        idx2 = (freqs>(freq0-0.5*Dnu)) & (freqs<(freq0+0.3*Dnu)) & (ls==2)
        freq2s[ifreq] = np.nan if np.sum(idx2)==0 else freqs[idx2][np.argmax(freqs[idx2])]

    # width estimates based on Yu+2018, Lund+2017, Li+2020
    k, b = 0.9638, -1.7145
    width = np.exp(k*np.log(numax) + b)
    weight = np.exp(-(freq2s-numax)**2./(2*width**2.))
    idx = (freq2s < numax+3*width) & (freq2s > numax-3*width)  & np.isfinite(weight)
    if np.sum(weight[idx]) == 0.:
        δν02 = np.nan 
    else:
        δν02 = np.average(freq0s[idx] - freq2s[idx], weights=weight[idx])
    return δν02


def ν_g_as(ν, params):
    n = np.arange(np.floor(np.min(1e6/ν)/params['ΔΠ1'])-1, np.floor(np.max(1e6/ν)/params['ΔΠ1']), 1)
    return params['ΔΠ1'] * (n + params['ε_g'])

def ν_p_as(ν, params, constant_ε_p=False, constant_d01=False):
    n = np.arange(np.floor(np.min(ν)/params['Δν'])-1, np.floor(np.max(ν)/params['Δν']), 1)
    return params['Δν'] * (n + 1/2 + ε_p(params['Δν'] * (n + 1/2 + params['ε_p'] + d01(ν, params, constant_d01=constant_d01)), params, constant_ε_p=constant_ε_p) + 
                                      d01(ν, params, constant_d01=constant_d01))

def make_fp(ν, params):
    Theta_g = np.pi*(params['ε_g'] - 1/(ν*1e-6)/params['ΔΠ1'])
    cot = np.tan(Theta_g)**-1.
    csc = 1/np.sin(Theta_g)
    c = 1e6
    num = params['Δν']*(cot*params['q'] - c*np.pi*csc**2.*q(ν, params)/(ν**2.*params['ΔΠ1']))
    den = np.pi*(1 + cot**2. * q(ν, params)**2.)
    return 1 - num/den

def make_fpp(ν, params):
    Theta_g = np.pi*(params['ε_g'] - 1/(ν*1e-6)/params['ΔΠ1'])
    cot = np.tan(Theta_g)**-1.
    csc = 1/np.sin(Theta_g)
    c = 1e6
    # q = q(ν, p)

    term1 = (1 / (ν**4 * params['ΔΠ1']**2)) * 2 * c**2 * np.pi**2
    # term2_common = c / (ν * p.ΔΠ1) - p.ε_g
    term2 = cot * csc**2
    term3 = q(ν, params)
    term4 = - (2 * c * np.pi * csc**2 * params['q_k']) / (ν**2 * params['ΔΠ1'])
    term5 = + (2 * c * np.pi * csc**2 * term3) / (ν**3 * params['ΔΠ1'])
    term6 = 1 + cot**2 * term3**2
    
    first_half = -((params['Δν'] * (term1 * term2 * term3 + term4 + term5)) / (np.pi * term6))
    
    term7 = params['Δν'] * (cot * params['q_k'] - (c * np.pi * csc**2 * term3) / (ν**2 * params['ΔΠ1']))
    term8 = 2 * cot**2 * params['q_k'] * term3 - 1 / (ν**2 * params['ΔΠ1']) * 2 * c * np.pi * cot * csc**2 * term3**2
    term9 = 1 + cot**2 * term3**2
    
    second_half = term7 * term8 / (np.pi * term9**2)
    
    return first_half + second_half