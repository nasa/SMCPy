import matplotlib.pyplot as plt
import numpy as np
import time

from smcpy import MVNormal, ImproperUniform
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.utils.plotter import plot_mcmc_chain, plot_pairwise
from smcpy.priors import InvWishart

NUM_SNAPSHOTS = 10
NUM_FEATURES = 3
TRUE_PARAMS = np.array([[2, 3.5]])
X = np.arange(NUM_FEATURES)
TRUE_COV = np.array([[0.5, 0.25, 0.005],
                     [0.25, 0.25, 0.04],
                     [0.005, 0.04, 1]])

def eval_model(theta):
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * X + b


def gen_data_from_mvn(eval_model, plot=True):
    y_true = eval_model(TRUE_PARAMS)
    mean = [0] * NUM_FEATURES
    noisy_data = np.tile(y_true, (NUM_SNAPSHOTS, 1))
    noisy_data += np.random.multivariate_normal(mean, TRUE_COV, NUM_SNAPSHOTS)
    if plot:
        plot_noisy_data(X, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    for i, nd in enumerate(noisy_data):
        ax.plot(x, nd, '-o', label=f'Noisy Snapshot {i}')
    ax.plot(x, y_true[0], 'k-', linewidth=2, label='True')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.legend(bbox_to_anchor=(1.1, 1.0))
    plt.show()


if __name__ == '__main__':

    np.random.seed(200)

    noisy_data = gen_data_from_mvn(eval_model, plot=True)
    num_samples = 50000
    burnin = int(num_samples / 2)
    init_inputs = np.array([[1, 2]])
    priors = [ImproperUniform(0., 6.), ImproperUniform(0., 6.)]

    idx1, idx2 = np.triu_indices(noisy_data.shape[1])
    log_like_args = [None] * len(idx1) # estimate all variances/covariances
    log_like_func = MVNormal

    init_cov = np.eye(noisy_data.shape[1])[idx1, idx2].reshape(1, -1) * 0.5
    init_inputs = np.concatenate((init_inputs, init_cov), axis=1)
    priors.append(InvWishart(NUM_FEATURES))

    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, log_like_args,
                             log_like_func)

    cov = np.eye(init_inputs.shape[1]) * 0.001
    chain = vector_mcmc.metropolis(init_inputs, num_samples, cov,
                                   adapt_interval=200, adapt_delay=5000,
                                   progress_bar=True)

    #plotting
    ground_truth = np.concatenate((TRUE_PARAMS.flatten(), TRUE_COV[idx1, idx2]))
    labels = ['a', 'b'] + [f'var{i}' for i in range(init_cov.shape[1])]
    plot_mcmc_chain(chain, param_labels=labels, burnin=burnin,
                    include_kde=True)
    plot_pairwise(chain.squeeze().T[burnin+1:], param_labels=labels,
                  true_params=ground_truth)
