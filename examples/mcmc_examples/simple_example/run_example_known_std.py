import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.utils.plotter import plot_mcmc_chain


TRUE_PARAMS = np.array([[2, 3.5]])
TRUE_STD = 2


def eval_model(theta):
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model, plot=True):
    y_true = eval_model(TRUE_PARAMS)
    noisy_data = y_true + np.random.normal(0, TRUE_STD, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), '-k')
    ax.plot(x.flatten(), noisy_data.flatten(), 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


if __name__ == '__main__':

    np.random.seed(200)

    noisy_data = generate_data(eval_model, plot=False)
    num_samples = 10000
    burnin = 5000
    std_dev = 2
    init_inputs = np.array([[2, 2], [3.5, 3.5]])
    cov = np.eye(2)

    priors = [uniform(0., 6.), uniform(0., 6.)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)

    chain = vector_mcmc.metropolis(init_inputs, num_samples, cov,
                                   adapt_interval=200, adapt_delay=100,
                                   progress_bar=True)

    plot_mcmc_chain(chain, param_labels=['a', 'b'], burnin=burnin,
                    include_kde=True)
