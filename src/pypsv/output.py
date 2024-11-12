import numpy as np
import pandas as pd


def generate_curve_output(
    iData,
    thin=1,
    type='numpy',
):
    knots = iData.observed_data['curve_knots'].values

    samples = {}
    if 'd_at_knots' in iData.posterior.keys():
        d_samps = iData.posterior['d_at_knots'].values
        d_samps = d_samps.reshape(-1, d_samps.shape[-1]).T
        samples['D'] = d_samps[:, ::thin]

    if 'i_at_knots' in iData.posterior.keys():
        i_samps = iData.posterior['i_at_knots'].values
        i_samps = i_samps.reshape(-1, i_samps.shape[-1]).T
        samples['I'] = i_samps[:, ::thin]

    if 'f_at_knots' in iData.posterior.keys():
        f_samps = iData.posterior['f_at_knots'].values
        f_samps = f_samps.reshape(-1, f_samps.shape[-1]).T
        samples['F'] = f_samps[:, ::thin]

    if type == 'numpy':
        return knots, samples
    elif type == 'pandas':
        data = {'t': knots}

        if 'D' in samples.keys():
            for it, _sample in enumerate(samples['D'].T):
                data[f'D_{it}'] = _sample

        if 'I' in samples.keys():
            for it, _sample in enumerate(samples['I'].T):
                data[f'I_{it}'] = _sample

        if 'F' in samples.keys():
            for it, _sample in enumerate(samples['F'].T):
                data[f'F_{it}'] = _sample

        return pd.DataFrame(
            data=data,
        )


def generate_data_output(
    iData,
    thin=1,
    type='numpy',
):
    data = {}
    for label in iData.observed_data['data_labels'].values:
        _this_data = {}
        if label in iData.observed_data['absolute_labels']:
            index = np.argwhere(
                label
                == iData.observed_data['absolute_labels'].values.flatten()
            ).flatten().item()
            _this_data['t'] = \
                iData.observed_data['absolute_dates'].values[index]
        else:
            _this_data['t'] = \
                iData.posterior[f't_{label}'].values.flatten()[::thin]

        if f'd_at_t_{label}' in iData.posterior.keys():
            _this_data['D'] = \
                iData.posterior[f'd_at_t_{label}'].values.flatten()[::thin]

        if f'i_at_t_{label}' in iData.posterior.keys():
            _this_data['I'] = \
                iData.posterior[f'i_at_t_{label}'].values.flatten()[::thin]

        if f'f_at_t_{label}' in iData.posterior.keys():
            _this_data['F'] = \
                iData.posterior[f'f_at_t_{label}'].values.flatten()[::thin]

        data[label] = _this_data

    return data
