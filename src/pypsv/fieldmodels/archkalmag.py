import numpy as np

from scipy.interpolate import BSpline

from .fieldmodel import FieldModel


class ArchKalmag14k(FieldModel):
    def __init__(self):
        with np.load(
            "/media/arthus/Extreme SSD/archeo_prior/dat/"
            "archkalmag_coeffs_ensemble.npz",
        ) as fh:
            self._knots = fh['knots']

            coeffs = fh['samples'].transpose(2, 1, 0)
            self._coeffs = coeffs[:, :coeffs.shape[1] // 2]

        self._l_max = 5
        self._n_samples = self._coeffs.shape[2]

        self._t_min = -12000
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
                "At least one epoch is not covered by the model"
            )
        return self._linear_spline(t)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    akm = ArchKalmag14k()

    plt.plot(
        akm.knots,
        akm.coeffs[:, 0, 0],
    )

    plt.plot(
        akm.knots,
        akm(akm.knots)[:, 0, 0],
    )
    plt.show()
