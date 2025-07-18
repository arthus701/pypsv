import os
from urllib.request import urlretrieve

import warnings

from appdirs import AppDirs

import numpy as np

from scipy.interpolate import BSpline

from pypsv.fieldmodels.fieldmodel import FieldModel


data_dir = AppDirs("pypsv").user_data_dir


class PFM9k2(FieldModel):
    def __init__(self):
        if not os.path.exists(data_dir):
            warnings.warn(
                "Data folder doesn't exist and will be created at "
                f"'{data_dir}'.",
                UserWarning,
            )
            os.makedirs(data_dir)

        filepath = data_dir + '/' + 'pfm9k2_ensemble.npz'

        if not os.path.isfile(filepath):
            warnings.warn(
                "pfm9k.2 ensemble datafile doesn't exist and will be "
                "downloaded. This may take some time. The file will be written"
                f" to {filepath}",
                UserWarning,
            )
            urlretrieve(
                "https://nextcloud.gfz.de/s/ytQTX3Hjcr3YNfZ/"
                "download/pfm9k2_ensemble.npz",
                filepath,
            )
        with np.load(self.filepath) as fh:
            self._knots = fh['knots']

            coeffs = fh['samples'].transpose(1, 0, 2)
            self._coeffs = coeffs

        self._l_max = 5
        self._n_samples = self._coeffs.shape[2]

        self._t_min = -7000
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

    pfm = PFM9k2()

    plt.plot(
        pfm.knots,
        pfm.coeffs[:, 0, 0],
    )

    plt.plot(
        pfm.knots,
        pfm(pfm.knots)[:, 0, 0],
    )
    plt.show()
