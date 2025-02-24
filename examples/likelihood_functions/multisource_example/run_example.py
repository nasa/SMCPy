import matplotlib.pyplot as plt
import numpy as np
import time

from smcpy import MultiSourceNormal, ImproperUniform
from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.utils.plotter import plot_mcmc_chain

NUM_DATA_PTS = 20
TRUE_PARAMS = np.array([[2, 3.5]])
X = np.arange(NUM_DATA_PTS)


def eval_model(theta):
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    y = np.tile(a * X + b, (1, 3))
    return y


def gen_data_from_multi_src(eval_model, std_devs, plot=True):
    y_true = np.array_split(eval_model(TRUE_PARAMS), 3, axis=1)
    noisy_data = [
        y_true[i] + np.random.normal(0, std, y_true[i].shape)
        for i, std in enumerate(std_devs)
    ]
    if plot:
        plot_noisy_data(X, y_true, noisy_data)
    return np.array(noisy_data).flatten()


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    for i, nd in enumerate(noisy_data):
        ax.plot(x.flatten(), y_true[i].flatten(), "-k")
        ax.plot(x.flatten(), nd.flatten(), "o", label=f"std{i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(200)

    true_std_devs = (20, 5, 0.25)
    noisy_data = gen_data_from_multi_src(eval_model, true_std_devs, plot=True)
    num_samples = 50000
    burnin = int(num_samples / 2)
    num_params = 5
    init_inputs = np.array([[3] * num_params, [3] * num_params])
    cov = np.eye(num_params)

    src_num_pts = (NUM_DATA_PTS, NUM_DATA_PTS, NUM_DATA_PTS)
    src_std_devs = (None, None, None)  # estimate all 3 source std devs
    log_like_args = [src_num_pts, src_std_devs]
    log_like_func = MultiSourceNormal

    priors = [ImproperUniform(0.0, 6.0), ImproperUniform(0.0, 6.0)] + [
        ImproperUniform(0, None)
    ] * 3  # priors for all 3 source std devs
    vector_mcmc = VectorMCMC(
        eval_model, noisy_data, priors, log_like_args, log_like_func
    )

    chain = vector_mcmc.metropolis(
        init_inputs,
        num_samples,
        cov,
        adapt_interval=200,
        adapt_delay=200,
        progress_bar=True,
    )

    plot_mcmc_chain(
        chain,
        param_labels=["a", "b", "std1", "std2", "std3"],
        burnin=burnin,
        include_kde=True,
    )
