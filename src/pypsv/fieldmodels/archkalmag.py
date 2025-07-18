import os
from urllib.request import urlretrieve

import warnings

from appdirs import AppDirs

import numpy as np

from scipy.interpolate import BSpline

from pypsv.fieldmodels.fieldmodel import FieldModel


data_dir = AppDirs("pypsv").user_data_dir


class ArchKalmag8k(FieldModel):
    def __init__(self):
        if not os.path.exists(data_dir):
            warnings.warn(
                "Data folder doesn't exist and will be created at "
                f"'{data_dir}'.",
                UserWarning,
            )
            os.makedirs(data_dir)

        filepath = data_dir + '/' + 'archkalmag8k_coeffs_ensemble.npz'

        if not os.path.isfile(filepath):
            warnings.warn(
                "ArchKalmag14k ensemble datafile doesn't exist and will be "
                " This may take some time. The file will be written to "
                f"{filepath}",
                UserWarning,
            )
            urlretrieve(
                "https://nextcloud.gfz.de/s/ngxdEWndG3kGFZG/"
                "download/archkalmag8k_coeffs_ensemble.npz",
                filepath,
            )
        with np.load(filepath) as fh:
            self._knots = fh['knots'][::-1]

            coeffs = fh['samples'].transpose(1, 0, 2)
            self._coeffs = coeffs[::-1]

        self._l_max = 5
        self._n_samples = self._coeffs.shape[2]

        self._t_min = -6000
        self._t_max = 2000
        spline_knots = np.concatenate(
            [
                [self._knots[-1]],
                self._knots[::-1],
                [self._knots[0]],
            ]
        )

        self._linear_spline = BSpline(
            spline_knots,
            self._coeffs[::-1],
            1,
        )

    def __call__(self, t):
        if np.any(t < self._t_min) or np.any(self._t_max < t):
            raise ValueError(
                "At least one epoch is not covered by the model."
            )
        return self._linear_spline(t)


class ArchKalmag14k(FieldModel):
    def __init__(self):
        if not os.path.exists(data_dir):
            warnings.warn(
                "Data folder doesn't exist and will be created at "
                f"'{data_dir}'.",
                UserWarning,
            )
            os.makedirs(data_dir)

        filepath = data_dir + '/' + 'archkalmag14k_coeffs_ensemble.npz'

        if not os.path.isfile(filepath):
            warnings.warn(
                "ArchKalmag14k ensemble datafile doesn't exist and will be "
                "downloaded. This may take some time. The file will be written"
                f" to {filepath}",
                UserWarning,
            )
            urlretrieve(
                "https://nextcloud.gfz-potsdam.de/s/8yYiDQtyA6WMwde/"
                "download/archkalmag14k_coeffs_ensemble.npz",
                filepath,
            )
        with np.load(filepath) as fh:
            self._knots = fh['knots']

            coeffs = fh['samples'].transpose(2, 1, 0)
            self._coeffs = coeffs[:, :coeffs.shape[1] // 2]

        self._l_max = 5
        self._n_samples = self._coeffs.shape[2]

        self._t_min = -8000
        self._t_max = 2000
        spline_knots = np.concatenate(
            [
                [self._knots[-1]],
                self._knots[::-1],
                [self._knots[0]],
            ]
        )

        self._linear_spline = BSpline(
            spline_knots,
            self._coeffs[::-1],
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

    akm8k = ArchKalmag8k()
    akm14k = ArchKalmag14k()

    plt.plot(
        akm8k.knots,
        akm8k.coeffs[:, 0, :].mean(axis=-1),
    )

    plt.plot(
        akm14k.knots,
        akm14k(akm14k.knots)[:, 0, :].mean(axis=-1),
    )
    plt.show()
