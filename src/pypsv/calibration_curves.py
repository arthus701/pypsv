import numpy as np
import pandas as pd

from scipy.interpolate import BSpline


# TODO: auto-download of calibration curves
_curve_names = {
    'IntCal20': "intcal20.14c",
    'Marine20': "marine20.14c",
    'SHCal20': "shcal20.14c",
}

data_dir = '../dat/'

intcal20 = pd.read_csv(
    data_dir + '/' + _curve_names['IntCal20'],
    header=11,
    sep=',',
    names=[
        "CAL BP",
        "14C age",
        "Sigma 14C",
        "Delta 14C",
        "Sigma Delta 14C",
    ]
)

marine20 = pd.read_csv(
    data_dir + '/' + _curve_names['Marine20'],
    header=11,
    sep=',',
    names=[
        "CAL BP",
        "14C age",
        "Sigma 14C",
        "Delta 14C",
        "Sigma Delta 14C",
    ]
)

shcal20 = pd.read_csv(
    data_dir + '/' + _curve_names['SHCal20'],
    header=11,
    sep=',',
    names=[
        "CAL BP",
        "14C age",
        "Sigma 14C",
        "Delta 14C",
        "Sigma Delta 14C",
    ]
)


def spline_curve(calibration_curve):
    knots = np.flip(calibration_curve['CAL BP'].to_numpy())
    knots = np.append(knots, np.array(max(knots)+1))
    knots = np.flip(1950 - knots)
    vals = calibration_curve[['14C age', 'Sigma 14C']].to_numpy()

    return BSpline(
        knots,
        vals,
        1,
    )


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for curve, color, label in zip(
        [intcal20, marine20, shcal20],
        ['C0', 'C1', 'C2'],
        ['intcal20', 'marine20', 'shcal20'],
    ):
        ax.plot(
            1950 - curve['CAL BP'],
            curve['14C age'],
            color=color,
            zorder=5,
            label=label,
        )
        ax.fill_between(
            1950 - curve['CAL BP'],
            curve['14C age'] + curve['Sigma 14C'],
            curve['14C age'] - curve['Sigma 14C'],
            color=color,
            zorder=0,
            alpha=0.4,
        )

    ax.set_xlim(-10e3, 1950)
    ax.set_ylim(None, 11e3)
    ax.set_xlabel('Cal age')
    ax.set_ylabel('14C age')

    ax.legend(frameon=False)

    plt.show()
