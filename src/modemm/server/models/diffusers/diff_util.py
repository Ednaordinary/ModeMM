import numpy as np

def sigma_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    :param t_steps: Sigmas
    :param num_steps: Steps to interpolate to
    :return: List of the new interpolates steps
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

def verify_sigmas(sigmas: list):
    """
    Verifies that a list of sigmas is correct
    :param sigmas: The list of sigmas
    :return: Nothing or an error
    """
    errors = []
    if not all(isinstance(x, float) for x in sigmas):
        errors.append()