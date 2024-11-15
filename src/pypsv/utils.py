import numpy as np
from numpy import ndarray

import jax.numpy as jnp
from pytensor import tensor as pt, shared

from pytensor.tensor.math import Argmax
from pytensor.link.jax.dispatch import jax_funcify

from .calibration_curves import intcal20

REARTH = 6371.2


def matern32(x, y=None, tau=3.7, sigma=1.):
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)

    frac = (x[:, None] - y[None, :]) / tau
    res = sigma**2 * (1 + np.abs(frac)) * np.exp(-np.abs(frac))

    return res.reshape(x.shape[0], y.shape[0])


def reloc_f(F, loc, reloc):
    colat = np.deg2rad(90-loc[0])
    recolat = np.deg2rad(90-reloc[0])
    res = F * np.sqrt(
        np.abs((1 + 3*np.cos(recolat)**2) / (1 + 3*np.cos(colat)**2))
    )
    return res


def gcd(p1, p2):
    hav = np.sin((p2[0] - p1[0])/2)**2 \
        + np.cos(p1[0])*np.cos(p2[0])*np.sin((p2[1] - p1[1])/2)**2
    return 2 * REARTH * np.arcsin(np.sqrt(hav))


def log_prob_t(t, dat, a=3, b=4, calibration_curve=intcal20):
    # Generalized Student's t-distribution, as proposed by Christen and Perez
    # (2009)
    # XXX:not normalized
    mu = np.interp(
        np.atleast_1d(t),
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['14C age'].values,
    )

    sig = np.interp(
        np.atleast_1d(t),
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['Sigma 14C'].values,
    )

    sig = np.sqrt(dat['Sigma 14C']**2 + sig**2)
    df = mu - dat['14C age']
    return -(a+0.5) * np.log(b + 0.5 * (df / sig)**2)


def get_curve(dat, thresh=1e-2, func=log_prob_t, calibration_curve=intcal20):
    _t = 1950 - calibration_curve['CAL BP'].values
    prob = np.exp(func(_t, dat, calibration_curve=calibration_curve))
    prob /= np.sum(prob)
    inds = np.argwhere(thresh*prob.max() <= prob).flatten()

    prob = prob[min(inds):max(inds)]
    prob /= np.sum(prob)

    return _t[min(inds):max(inds)], prob


@jax_funcify.register(Argmax)
def jax_funcify_Argmax(op, **kwargs):
    # Obtain necessary "static" attributes from the Op being converted
    axis = op.axis

    # Create a JAX jit-able function that implements the Op
    def argmax(x, axis=axis):
        if axis is not None:
            if 1 < len(axis):
                raise ValueError("JAX doesn't support tuple axis.")
            axis = axis[0]

        return jnp.argmax(x, axis=axis).astype("int64")

    return argmax


def interp(_x, _xp, _fp):
    """
    Simple equivalent of np.interp to compute a linear interpolation. This
    function will not extrapolate, but use the edge values for points outside
    of _xp.
    """
    if isinstance(_xp, (ndarray, list)):
        xp = shared(_xp)
    else:
        xp = _xp

    if isinstance(_fp, (ndarray, list)):
        fp = shared(_fp)
    else:
        fp = _fp

    # No extrapolation!
    x = pt.clip(_x, xp[0], xp[-1])
    # First we find the nearest neighbour
    ind = pt.argmin((x[:, None] - xp[None, :])**2, axis=1)
    xi = xp[ind]
    # Figure out if we are on the right or the left of nearest
    s = pt.sign(x - xi).astype(int)
    # Perform linear interpolation
    # (1-pt.abs(s))*1e-10) to prevent divide by zero error if we are exactly
    # at the knot points
    a = (fp[ind + s] - fp[ind]) / (
        xp[ind + s] - xp[ind] + (1-pt.abs(s))*1e-10
    )

    b = fp[ind] - a * xp[ind]

    return a * x + b
