import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import gaussian_kde
from smcpy.priors import InvWishart


if __name__ == '__main__':

    n_samples = 1000
    cov_dim = 2
    x = np.linspace(-7, 7, 1000)

    fig, axes = plt.subplots(cov_dim, 2)
    for dof in range(cov_dim + 5, 15):
        samples = InvWishart(dof, np.eye(cov_dim) * 5).rvs(n_samples)

        for i, y in enumerate(samples.T):
            ax = axes.flatten()[i]
            kde = gaussian_kde(y)
            ax.plot(x, kde.pdf(x), label=f'{dof}')
            ax.set_xlabel(f'cov{i}')
            ax.set_ylabel('PDF')

        axes[0, 0].legend()

        ax = axes[-1, -1]
        ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.5)
        ax.set_xlabel('cov0')
        ax.set_ylabel('cov1')


    plt.tight_layout()
    plt.show()
