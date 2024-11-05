import sys
import os
import numpy as np

import arviz as az

from matplotlib import pyplot as plt

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

summary = az.summary(iData)
summary.index.names = ['Name']
summary.reset_index(inplace=True)

summary.to_csv(folder + "/" + basename + '_summary.csv', index=False)
cnt = 0

names = []
for it in summary['Name'][
    np.argwhere(summary['r_hat'].values > 1.1).flatten()
]:
    names.append(it)
    cnt += 1

print(
    "The maximal treedepth was "
    f"{np.max(np.array(iData.sample_stats['tree_depth']))}."
)
print(
    f"The chains had {np.sum(np.array(iData.sample_stats['diverging']))} "
    "divergences."
)
print(
    f"There were {cnt} random variables with rhat > 1.1."
)

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
