import numpy as np

import pymc as pm
from pytensor import tensor as pt

from pymc.sampling import jax as pmj

from pymagglobal.utils import dsh_basis, nez2dif

from fieldmodels.fieldmodel import FieldModel
from utils import get_curve, interp

JITTER = 1e-4
default_prior_values = {
    'D': (20, 150),
    'I': (10, 150),
    'F': (8.25, 150),
}


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
            self.prior_model = default_prior_values
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
            if 'D' in self.components:
                d = np.rad2deg(
                    np.arctan2(nez[1], nez[0])
                )
                self.prior_mean_d = d.mean(axis=1)
                _prior_cov_d = np.cov(d)
                self.prior_chol_d = np.linalg.cholesky(
                    _prior_cov_d + JITTER * np.eye(len(self.curve_knots))
                )
            if 'I' in self.components:
                i = np.rad2deg(
                    np.arctan2(
                        nez[2],
                        np.sqrt(nez[0]**2 + nez[1]**2),
                    )
                ),
                self.prior_mean_i = i.mean(axis=1)
                _prior_cov_i = np.cov(i)
                self.prior_chol_i = np.linalg.cholesky(
                    _prior_cov_i + JITTER * np.eye(len(self.curve_knots))
                )
            if 'F' in self.components:
                f = np.sqrt(
                    np.sum(
                        nez**2,
                        axis=0,
                    )
                )
                self.prior_mean_f = f.mean(axis=1)
                _prior_cov_f = np.cov(f)
                self.prior_chol_f = np.linalg.cholesky(
                    _prior_cov_f + JITTER * np.eye(len(self.curve_knots))
                )
        else:
            from utils import matern32

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

            nez_kalman = kalmag_2000_coeffs @ base
            d_kalman, i_kalman, f_kalman = nez2dif(*nez_kalman)
            f_kalman /= 1e3

            if 'D' in self.components:
                _prior_cor_d = matern32(
                    self.curve_knots,
                    kalmag_knot,
                    sigma=self.prior_model['D'][0],
                    tau=self.prior_model['D'][1],
                )

                self.prior_mean_d = (
                    _prior_cor_d.flatten()
                    * d_kalman
                    / self.prior_model['I'][0]**2
                )

                _prior_cov_d = matern32(
                    self.curve_knots,
                    sigma=self.prior_model['D'][0],
                    tau=self.prior_model['D'][1],
                ) - _prior_cor_d @ _prior_cor_d.T / self.prior_model['D'][0]**2

                self.prior_chol_d = np.linalg.cholesky(
                    _prior_cov_d + JITTER * np.eye(len(self.curve_knots))
                )
            if 'I' in self.components:
                # Merril, 1998, eq. 3.3.4
                prior_mean_i = (
                    np.deg2rad(
                        np.arctan(2 * np.tan(self.loc[0]))
                    )
                )
                _prior_cor_i = matern32(
                    self.curve_knots,
                    kalmag_knot,
                    sigma=self.prior_model['I'][0],
                    tau=self.prior_model['I'][1],
                )

                self.prior_mean_i = (
                    prior_mean_i * np.ones(len(self.curve_knots))
                )
                self.prior_mean_i += (
                    _prior_cor_i.flatten()
                    * (f_kalman - prior_mean_i)
                    / self.prior_model['I'][0]**2
                )

                _prior_cov_i = matern32(
                    self.curve_knots,
                    sigma=self.prior_model['I'][0],
                    tau=self.prior_model['I'][1],
                ) - _prior_cor_i @ _prior_cor_i.T / self.prior_model['I'][0]**2

                self.prior_chol_i = np.linalg.cholesky(
                    _prior_cov_i + JITTER * np.eye(len(self.curve_knots))
                )
            if 'F' in self.components:
                # Merril, 1998, eq. 3.4.5
                prior_mean_f = 28 * np.sqrt((1 + 3*np.cos(self.loc[0])**2))
                _prior_cor_f = matern32(
                    self.curve_knots,
                    kalmag_knot,
                    sigma=self.prior_model['F'][0],
                    tau=self.prior_model['F'][1],
                )

                self.prior_mean_f = (
                    prior_mean_f * np.ones(len(self.curve_knots))
                )
                self.prior_mean_f += (
                    _prior_cor_f.flatten()
                    * (f_kalman - prior_mean_f)
                    / self.prior_model['F'][0]**2
                )

                _prior_cov_f = matern32(
                    self.curve_knots,
                    sigma=self.prior_model['F'][0],
                    tau=self.prior_model['F'][1],
                ) - _prior_cor_f @ _prior_cor_f.T / self.prior_model['F'][0]**2

                self.prior_chol_f = np.linalg.cholesky(
                    _prior_cov_f + JITTER * np.eye(len(self.curve_knots))
                )

    def setup_mcmodel(self):
        with pm.Model() as self.mcModel:
            if 'D' in self.components:
                d_cent = pm.Normal(
                    'd_cent',
                    mu=0,
                    sigma=1,
                    size=(
                        len(self.curve_knots),
                    ),
                )
                d_at_knots = pm.Deterministic(
                    'd_at_knots',
                    pt.dot(self.prior_chol_d, d_cent) + self.prior_mean_d,
                )
            if 'I' in self.components:
                i_cent = pm.Normal(
                    'i_cent',
                    mu=0,
                    sigma=1,
                    size=(
                        len(self.curve_knots),
                    ),
                )
                i_at_knots = pm.Deterministic(
                    'i_at_knots',
                    pt.dot(self.prior_chol_i, i_cent) + self.prior_mean_i,
                )
            if 'F' in self.components:
                f_cent = pm.Normal(
                    'f_cent',
                    mu=0,
                    sigma=1,
                    size=(
                        len(self.curve_knots),
                    ),
                )
                f_at_knots = pm.Deterministic(
                    'f_at_knots',
                    pt.dot(self.prior_chol_f, f_cent) + self.prior_mean_f,
                )

            for idx, row in self.data.iterrows():
                x_points, pdf_points = get_curve(row)
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
                if (
                    'D' in self.components and not np.isnan(row['D'])
                ):
                    d_at_t = pm.Deterministic(
                        f'd_at_t_{idx}',
                        interp(
                            t,
                            self.curve_knots,
                            d_at_knots,
                        ),
                    )

                    rD = pm.Deterministic(
                        f'rD_{idx}',
                        (d_at_t - row['D']) / row['dI'],
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
                    i_at_t = pm.Deterministic(
                        f'i_at_t_{idx}',
                        interp(
                            t,
                            self.curve_knots,
                            i_at_knots,
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
                        interp(
                            t,
                            self.curve_knots,
                            f_at_knots,
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

    def data_sanity_check(self, data):
        self.components = []
        for comp in 'DIF':
            if comp in list(data.columns):
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
            "14C age",
            "Sigma 14C",
            "14C type",
        ]:
            if radiocarbon_attribute not in list(data.columns):
                raise RuntimeError(
                    f"No column '{radiocarbon_attribute}' found in DataFrame."
                )

        return data

    def sample(
        self,
        draws=500,
        tune=1000,
        progressbar=True,
        chains=4,
        target_accept=0.95,
    ):
        self.setup_mcmodel()
        with self.mcModel:
            idata = pmj.sample_numpyro_nuts(
                draws,
                tune=tune,
                progressbar=progressbar,
                chains=chains,
                target_accept=0.95,
                postprocessing_backend='cpu',
            )

        return idata
