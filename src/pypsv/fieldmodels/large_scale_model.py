import os
from urllib.request import urlretrieve

import warnings

from appdirs import AppDirs

import numpy as np

from scipy.interpolate import BSpline

from pymagglobal.utils import scaling, REARTH, i2lm_l

from pypsv.fieldmodels.fieldmodel import FieldModel
from pypsv.utils import matern32

rng = np.random.default_rng()
data_dir = AppDirs("pypsv").user_data_dir

sigma_ND = 95
tau_ND = 500
R = 2800


class LargeScaleModel(FieldModel):
    def __init__(self, l_max=1):
        if 5 < l_max:
            raise ValueError(
                'Maximum SH degree for large scale model is 5.'
            )
        if l_max < 1:
            raise ValueError(
                'Minimum SH degree for large scale model is 1.'
            )
        if not os.path.exists(data_dir):
            warnings.warn(
                "Data folder doesn't exist and will be created at "
                f"'{data_dir}'.",
                UserWarning,
            )
            os.makedirs(data_dir)

        filepath = data_dir + '/' + 'mixed_coeffs_ensemble.npz'

        if not os.path.isfile(filepath):
            warnings.warn(
                "Mixed model ensemble datafile doesn't exist and will be "
                " This may take some time. The file will be written to "
                f"{filepath}",
                UserWarning,
            )
            urlretrieve(
                "https://nextcloud.gfz.de/s/oMEcr4Ft7oKLm7W/"
                "download/mixed_coeffs_ensemble.npz",
                filepath,
            )
        with np.load(filepath) as fh:
            self._knots = fh['knots']

            coeffs = fh['samples'].transpose(1, 0, 2)

        self._l_max = 5

        mix = 1 / (1 + np.exp(-(self._knots - 1900) / 20))[:, None, None]

        reduced_coeffs = np.zeros_like(coeffs)

        n_coeffs = l_max * (l_max + 2)

        reduced_coeffs[:, :n_coeffs] = coeffs[:, :n_coeffs]
        scl = scaling(R, REARTH, self._l_max)
        for it in range(n_coeffs, coeffs.shape[1]):
            cov = scl[it]**2 * sigma_ND**2 * matern32(
                self._knots,
                tau=tau_ND / i2lm_l(it),
                sigma=1.,
            )
            chol = np.linalg.cholesky(cov + 1e-6 * np.eye(len(self._knots)))
            reduced_coeffs[:, it, :] = chol @ rng.normal(
                size=(len(self._knots), coeffs.shape[2]),
            )
            reduced_coeffs[:, it, :] -= \
                reduced_coeffs[:, it, :].mean(axis=-1)[..., None]

        coeffs = mix * coeffs + (1 - mix) * reduced_coeffs

        self._coeffs = coeffs[::-1]

        self._n_samples = self._coeffs.shape[2]

        self._t_min = self._knots.min()
        self._t_max = self._knots.max()
        spline_knots = np.concatenate(
            [
                [self._knots[0]],
                self._knots,
                [self._knots[-1]],
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

    from pypsv.fieldmodels.archkalmag import ArchKalmag8k

    akm8k = ArchKalmag8k()
    mixed = LargeScaleModel()

    plt.plot(
        mixed.knots,
        mixed.coeffs[:, 0, :].mean(axis=-1),
    )

    plt.plot(
        akm8k.knots,
        akm8k.coeffs[:, 0, :].mean(axis=-1),
        color='black',
        ls='--',
    )

    plt.show()
