import os
from urllib.request import urlretrieve

import warnings

from appdirs import AppDirs

import numpy as np

from scipy.interpolate import BSpline

from .fieldmodel import FieldModel


data_dir = AppDirs("pypsv").user_data_dir

if not os.path.exists(data_dir):
    warnings.warn(
        f"Data folder doesn't exist and will be created at '{data_dir}'.",
        UserWarning,
    )
    os.makedirs(data_dir)

filepath = data_dir + '/' + 'covarch_ensemble.npz'

if not os.path.isfile(filepath):
    warnings.warn(
        "cov-arch ensemble datafile doesn't exist and will be downloaded."
        " This may take some time.",
        UserWarning,
    )
    urlretrieve(
        "https://nextcloud.gfz.de/s/PxrdsbqXmcLReAC/"
        "download/covarch_ensemble.npz",
        filepath,
    )


class CovArch(FieldModel):
    def __init__(self):
        with np.load(filepath) as fh:
            self._knots = fh['knots']

            coeffs = fh['samples'].transpose(1, 0, 2)
            self._coeffs = coeffs

        self._l_max = 10
        self._n_samples = self._coeffs.shape[2]

        self._t_min = -1200
        self._t_max = 2000
        spline_knots = np.concatenate(
            [
                [self._knots[0]],
                self._knots,
                [self._knots[-1]],
            ]
        )

        self._linear_spline = BSpline(
            spline_knots,
            self._coeffs,
            1,
        )

    def __call__(self, t):
        if np.any(t < self._t_min) or np.any(self._t_max < t):
            raise ValueError(
                "At least one epoch is not covered by the model."
            )
        return self._linear_spline(t)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    covarch = CovArch()

    plt.plot(
        covarch.knots,
        covarch.coeffs[:, 0, 0],
    )

    plt.plot(
        covarch.knots,
        covarch(covarch.knots)[:, 0, 0],
    )
    plt.show()
