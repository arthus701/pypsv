import numpy as np
import pandas as pd

from .utils import get_curve
from .calibration_curves import intcal20, shcal20, marine20


def add_input_data_to_ax(
    ax,
    input_data,
    which,
    rc_height=1.5,
    **kwargs,
):
    """
    Helper routine to add data to an axes. Depending on the data type, the
    uncertainties will be drawn differently (e.g. violins for Radiocarbon
    dated points).

    Parameters
    ----------
    ax : matplotlib.Axes
        The axes to be drawn into
    input_data : pd.DataFrame
        The input data, formatted as required by psvcurve
    which : string, either 'D', 'I' or 'F'
        Which component to draw
    rc_height : float, optional
        The height of the Radiocarbon violins, default is 1.5
    **kwargs
        Will be passed to the matplotlib routines. Can be used to specify color
        etc.
    """
    if 'color' in kwargs:
        color = kwargs.get('color')
        del kwargs['color']
    else:
        color = 'grey'

    for idx, row in input_data.iterrows():
        if "14C" in row['Age type']:
            if row['Age type'] == '14C NH':
                calibration_curve = intcal20
            elif row['Age type'] == '14C SH':
                calibration_curve = shcal20
            elif row['Age type'] == '14C MA':
                calibration_curve = marine20

            x_points, pdf_points = get_curve(
                pd.Series(
                    data={
                        '14C age': row['Age'],
                        'Sigma 14C': row['dAge'],
                    }
                ),
                calibration_curve=calibration_curve,
            )

            if 0 < rc_height:
                ax.fill_between(
                    x_points,
                    row[which] + rc_height * pdf_points / pdf_points.max(),
                    row[which] - rc_height * pdf_points / pdf_points.max(),
                    alpha=0.3,
                    color=color,
                    **kwargs,
                )
            idx_median = np.argmin(np.abs(np.cumsum(pdf_points) - 0.5))

            x_median = x_points[idx_median]
            ax.errorbar(
                x_median,
                row[which],
                xerr=np.atleast_2d([
                    x_median - x_points.min(),
                    x_points.max() - x_median,
                ]).T,
                yerr=row[f'd{which}'],
                ls='',
                marker='.',
                zorder=5,
                color=color,
                **kwargs,
            )
        elif "uniform" in row['Age type'] or "Gaussian" in row['Age type']:
            ax.errorbar(
                row['Age'],
                row[which],
                xerr=row['dAge'],
                yerr=row[f'd{which}'],
                alpha=0.6,
                ls='',
                marker='.',
                color=color,
                **kwargs,
            )
        elif "absolute" in row['Age type']:
            ax.errorbar(
                row['Age'],
                row[which],
                yerr=row[f'd{which}'],
                ls='',
                marker='.',
                color=color,
                **kwargs,
            )
