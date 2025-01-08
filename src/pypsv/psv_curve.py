import numpy as np

from pandas import Series

import pymc as pm
from pytensor import tensor as pt

from pymc.sampling import jax as pmj

from pymagglobal.utils import dsh_basis

from .fieldmodels.fieldmodel import FieldModel
from .utils import get_curve, interp
from .calibration_curves import intcal20, shcal20, marine20

JITTER = 1e-4
AXIAL_DIPOLE_PRIOR = -30
DEFAULT_PRIOR_VALUES = (8.25, 150)
MODEL_INCREASE_FACTOR = 2

age_types = [
    "14C NH",
    "14C SH",
    "14C MA",
    "absolute",
    "uniform",
    "Gaussian",
]


class PSVCurve(object):
    def __init__(
        self,
        loc,
        curve_knots,
        data,
        prior_model=None,
    ):
        self.loc = loc
        self.curve_knots = curve_knots
        if prior_model is None:
            self.prior_model = DEFAULT_PRIOR_VALUES
        else:
            self.prior_model = prior_model

        self.data = self.data_sanity_check(data)

        self.setup_prior()

    def setup_prior(self):
        z_loc = np.atleast_2d(
            [
                90 - self.loc[0],
                self.loc[1],
                6371.2,
            ]
        ).T
        if isinstance(self.prior_model, FieldModel):
            base = dsh_basis(self.prior_model.l_max, z_loc)
            coeffs = self.prior_model(self.curve_knots)

            nez = np.einsum(
                'ij, kil -> jkl',
                base,
                coeffs,
            )

            self.prior_mean = nez.mean(axis=2).T.flatten()
            _prior_cov = np.cov(
                nez.transpose(1, 0, 2).reshape(-1, coeffs.shape[-1])
            )
            self.prior_chol = MODEL_INCREASE_FACTOR * np.linalg.cholesky(
                _prior_cov + JITTER * np.eye(3 * len(self.curve_knots))
            )

        else:
            from .utils import matern32

            kalmag_lmax = 13
            kalmag_2000_coeffs = np.genfromtxt(
                '../dat/Kalmag_2000_CORE_MEAN_Radius_6371.2.txt'
            )
            kalmag_knot = np.atleast_1d(
                np.copy(
                    kalmag_2000_coeffs[0]
                )
            )
            kalmag_2000_coeffs = kalmag_2000_coeffs[
                1:(kalmag_lmax * (kalmag_lmax + 2) + 1)
            ]
            base = dsh_basis(kalmag_lmax, z_loc)

            nez_kalman = kalmag_2000_coeffs @ base / 1e3

            _prior_cor = matern32(
                self.curve_knots,
                kalmag_knot,
                sigma=self.prior_model[0],
                tau=self.prior_model[1],
            )

            nez_mean = AXIAL_DIPOLE_PRIOR * base[0]

            self.prior_mean = (
                nez_mean[:, None]
                + (
                    _prior_cor.flatten()[None, :]
                    * (nez_kalman[:, None] - nez_mean[:, None])
                    / self.prior_model[0]**2
                )
            ).T.flatten()

            _prior_cov = matern32(
                self.curve_knots,
                sigma=self.prior_model[0],
                tau=self.prior_model[1],
            ) - _prior_cor @ _prior_cor.T / self.prior_model[0]**2

            prior_chol = np.linalg.cholesky(
                _prior_cov + JITTER * np.eye(len(self.curve_knots))
            )
            zero_block = np.zeros(
                (len(self.curve_knots), len(self.curve_knots)),
            )
            prior_chol = np.array(
                [
                    [prior_chol, zero_block, zero_block],
                    [zero_block, prior_chol, zero_block],
                    [zero_block, zero_block, prior_chol],
                ],
            )
            self.prior_chol = (
                prior_chol
                .transpose(2, 0, 3, 1)
                .reshape(
                    3 * len(self.curve_knots), 3 * len(self.curve_knots)
                )
            )

    def setup_mcmodel(self):
        with pm.Model() as self.mcModel:
            nez_cent = pm.Normal(
                'nez_cent',
                mu=0,
                sigma=1,
                size=(
                    3 * len(self.curve_knots),
                ),
            )
            nez_at_knots = pm.Deterministic(
                'nez_at_knots',
                pt.reshape(
                    pt.dot(self.prior_chol, nez_cent) + self.prior_mean,
                    (len(self.curve_knots), 3),
                ),
            )

            for idx, row in self.data.iterrows():
                if "14C" in row['Age type']:
                    if row['Age type'] == '14C NH':
                        calibration_curve = intcal20
                    elif row['Age type'] == '14C SH':
                        calibration_curve = shcal20
                    elif row['Age type'] == '14C MA':
                        calibration_curve = marine20

                    x_points, pdf_points = get_curve(
                        Series(
                            data={
                                '14C age': row['Age'],
                                'Sigma 14C': row['dAge'],
                            }
                        ),
                        calibration_curve=calibration_curve,
                    )
                    cdf_points = np.cumsum(pdf_points)

                    t_uni = pm.Uniform(
                        f't_uniform_{idx}',
                        lower=0.,
                        upper=1.,
                        size=1,
                    )

                    t = pm.Deterministic(
                        f't_{idx}',
                        interp(
                            t_uni,
                            cdf_points,
                            x_points,
                        ),
                    )
                elif row['Age type'] == 'absolute':
                    t = pt.as_tensor([row['Age']])
                elif row['Age type'] == 'uniform':
                    t_uni = pm.Uniform(
                        f't_uniform_{idx}',
                        lower=0.,
                        upper=1.,
                        size=1,
                    )

                    upper = 2 * row['dAge']
                    lower = row['Age'] - row['dAge']
                    t = pm.Deterministic(
                        f't_{idx}',
                        t_uni * upper - lower,
                    )
                elif row['Age type'] == 'Gaussian':
                    t_cent = pm.Normal(
                        f't_cent_{idx}',
                        mu=0.,
                        sigma=1.,
                        size=1,
                    )

                    t = pm.Deterministic(
                        f't_{idx}',
                        t_cent * row['dAge'] + row['Age'],
                    )

                nez_at_t = interp(
                    t,
                    self.curve_knots,
                    nez_at_knots,
                )

                if (
                    'D' in self.components and not np.isnan(row['D'])
                ):
                    d_at_t = pm.Deterministic(
                        f'd_at_t_{idx}',
                        pt.rad2deg(
                            pt.arctan2(
                                nez_at_t[:, 1],
                                nez_at_t[:, 0],
                            ),
                        ),
                    )
                    _rD = d_at_t - row['D']
                    rD = pm.Deterministic(
                        f'rD_{idx}',
                        ((_rD - 360 * (_rD > 180) + 360 * (-180 > _rD)))
                        / row['dD'],
                    )

                    pm.StudentT(
                        f'd_obs_{idx}',
                        nu=4,
                        mu=rD,
                        sigma=1.,
                        observed=[0.],
                    )
                if (
                    'I' in self.components and not np.isnan(row['I'])
                ):
                    _h = pt.sqrt(
                        pt.sum(
                            pt.square(nez_at_t[:, 0:2]),
                            axis=1,
                        ),
                    )
                    i_at_t = pm.Deterministic(
                        f'i_at_t_{idx}',
                        pt.rad2deg(
                            pt.arctan(
                                nez_at_t[:, 2]
                                / _h,
                            ),
                        ),
                    )

                    rI = pm.Deterministic(
                        f'rI_{idx}',
                        (i_at_t - row['I']) / row['dI'],
                    )

                    pm.StudentT(
                        f'i_obs_{idx}',
                        nu=4,
                        mu=rI,
                        sigma=1.,
                        observed=[0.],
                    )
                if (
                    'F' in self.components and not np.isnan(row['F'])
                ):
                    f_at_t = pm.Deterministic(
                        f'f_at_t_{idx}',
                        pt.sqrt(
                            pt.sum(
                                pt.square(nez_at_t),
                                axis=1,
                            ),
                        ),
                    )

                    rF = pm.Deterministic(
                        f'rF_{idx}',
                        (f_at_t - row['F']) / row['dF'],
                    )

                    pm.StudentT(
                        f'f_obs_{idx}',
                        nu=4,
                        mu=rF,
                        sigma=1.,
                        observed=[0.],
                    )

    def data_sanity_check(self, _data):
        data = _data.copy()
        self.components = []
        for comp in 'DIF':
            if comp in list(_data.columns):
                self.components.append(comp)

                if f"d{comp}" not in list(data.columns):
                    raise RuntimeError(
                        f"'{comp}' data found, but no 'd{comp}' column."
                    )
            else:
                data[comp] = np.nan
                data[f"d{comp}"] = np.nan

        if len(self.components) == 0:
            raise RuntimeError("No usable data found in DataFrame.")

        for radiocarbon_attribute in [
            'Age',
            'dAge',
            'Age type',
        ]:
            if radiocarbon_attribute not in list(data.columns):
                raise RuntimeError(
                    f"No column '{radiocarbon_attribute}' found in DataFrame."
                )

        for age_type in data['Age type']:
            if age_type not in age_types:
                print(
                    f"'{age_type}' is not a valid age type. Valid types are:"
                )
                for _type in age_types:
                    print(_type)
                exit()

        return data

    def sample(
        self,
        draws=500,
        tune=1000,
        progressbar=True,
        chains=4,
        target_accept=0.95,
        **kwargs,
    ):
        if progressbar:
            print("Setting up PyMC model...")
        self.setup_mcmodel()
        if progressbar:
            print("...model setup done.")
        with self.mcModel:
            iData = pmj.sample_numpyro_nuts(
                draws,
                tune=tune,
                progressbar=progressbar,
                chains=chains,
                target_accept=target_accept,
                postprocessing_backend='cpu',
                **kwargs,
            )

        iData.observed_data['loc'] = np.array(self.loc)
        iData.observed_data['curve_knots'] = self.curve_knots
        iData.observed_data['data_labels'] = self.data.index.values

        absolute_data = self.data[
            self.data['Age type'] == 'absolute'
        ]
        iData.observed_data['absolute_labels'] = absolute_data.index.values
        iData.observed_data['absolute_dates'] = absolute_data['Age'].values

        return iData
