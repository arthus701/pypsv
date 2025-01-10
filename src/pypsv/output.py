import numpy as np
import pandas as pd

from pymagglobal.utils import nez2dif


def generate_curve_output(
    iData,
    thin=1,
    type='numpy',
    output_nez=False,
):
    """
    Generate curve output from the posterior samples in the input data.

    Parameters
    ----------
    iData : object
        Input data object containing observed and posterior data.
    thin : int, optional
        Thinning factor for the samples, default is 1.
    type : str, optional
        Type of output to generate, either 'numpy' or 'pandas'. Default is
        'numpy'.
    output_nez : bool, optional
        If True, include N, E, Z components in the output. Default is False.

    Returns
    -------
    knots : ndarray
        Array of knot points from the observed data. Only included if type is
        'numpy'.
    samples : dict or DataFrame
        Dictionary or DataFrame containing the generated samples. The
        dictionary keys are:
        'D', 'I', 'F' for dif components, and optionally 'N', 'E', 'Z' if
        output_nez is True.
        If type is 'pandas', a DataFrame is returned with columns for each
        sample and statistics.
    """
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
        'D': d_samples,
        'I': i_samples,
        'F': f_samples,
    }
    if output_nez:
        samples['N'] = nez_samples[0]
        samples['E'] = nez_samples[1]
        samples['Z'] = nez_samples[2]

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
):
    """
    Generate data output from the posterior samples in the input data.

    Parameters
    ----------
    iData : object
        Input data object containing observed and posterior data.
    thin : int, optional
        Thinning factor for the samples, default is 1.

    Returns
    -------
    data : dict
        Dictionary containing the generated data for each label. Each label's
        data is a dictionary with keys 't', 'D', 'I', 'F' representing the time
        and dif components respectively.
    """
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
