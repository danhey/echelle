from astropy.convolution import convolve, Box1DKernel


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
