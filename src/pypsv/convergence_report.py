import sys
import os
import numpy as np

import arviz as az

from matplotlib import pyplot as plt


def calc_summary_and_print_diagnostics(iData, threshold=1.1):
    summary = az.summary(iData)
    summary.index.names = ['Name']
    summary.reset_index(inplace=True)

    num_samples = (
        iData.posterior.sizes['chain'] *
        iData.posterior.sizes['draw']
    )

    cnt = 0

    names = []
    for it in summary['Name'][
        np.argwhere(summary['r_hat'].values > threshold).flatten()
    ]:
        names.append(it)
        cnt += 1

    num_samples = (
        iData.posterior.sizes['chain'] *
        iData.posterior.sizes['draw']
    )
    bulk = summary['ess_bulk'].values / num_samples
    tail = summary['ess_tail'].values / num_samples

    print(
        "The maximal treedepth was "
        f"{np.max(np.array(iData.sample_stats['tree_depth']))}."
    )
    print(
        f"The chains had {np.sum(np.array(iData.sample_stats['diverging']))} "
        "divergences."
    )
    print(
        f"There were {cnt} random variables with rhat > {threshold:.2f}."
    )
    if 0 < cnt:
        print(
            "Here's a list of the names of the random variables with "
            f"rhat > {threshold:.2f}:"
        )
        print(names)

    print(
        "Effective sample size ratio estimations:\n"
        "            min / mean / max\n"
        "from bulk: "
        f"{np.min(bulk):.2f} / {np.mean(bulk):.2f} / {np.max(bulk):.2f}\n"
        "from tail: "
        f"{np.min(tail):.2f} / {np.mean(tail):.2f} / {np.max(tail):.2f}\n"

    )
    return summary


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except IndexError:
        print(
            "Please provide the path to a .nc file, containing the results of "
            "a model run."
        )
        exit()

    folder = os.path.dirname(fname)
    basename = os.path.basename(fname).removesuffix('.nc')

    iData = az.from_netcdf(fname)

    summary = calc_summary_and_print_diagnostics(iData)

    summary.to_csv(folder + "/" + basename + '_summary.csv', index=False)

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    az.plot_energy(iData, ax=axs[0])

    lp = np.array(iData.sample_stats['lp'])
    n_chains = lp.shape[0]

    for it in range(4):
        axs[1].plot(
            np.arange(lp.shape[1]),
            lp[it],
        )

    axs[1].set_ylabel('log $p$')
    axs[1].set_xlabel('$n$ in chain')
    axs[1].set_xlim(0, lp.shape[1])

    fig.tight_layout(w_pad=1.3)

    plt.show()
