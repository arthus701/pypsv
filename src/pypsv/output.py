import numpy as np
import pandas as pd

from pymagglobal.utils import nez2dif


def generate_curve_output(
    iData,
    thin=1,
    type='numpy',
):
    knots = iData.observed_data['curve_knots'].values

    nez_samples = (
        iData
        .posterior['nez_at_knots']
        .values
        .reshape(
            -1,
            len(iData.observed_data['curve_knots']),
            3,
        )
        .transpose(2, 1, 0)
    )[:, :, ::thin]

    d_samples, i_samples, f_samples = nez2dif(*nez_samples)

    samples = {
        'N': nez_samples[0],
        'E': nez_samples[1],
        'Z': nez_samples[2],
        'D': d_samples,
        'I': i_samples,
        'F': f_samples,
    }

    if type == 'numpy':
        return knots, samples
    elif type == 'pandas':
        data = {'t': knots}

        for which in samples.keys():
            for it, _sample in enumerate(samples[which].T):
                data[f'{which}_{it}'] = _sample

            data[f'{which}_mean'] = samples[which].mean(axis=1)
            data[f'{which}_std'] = samples[which].std(axis=1)

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
