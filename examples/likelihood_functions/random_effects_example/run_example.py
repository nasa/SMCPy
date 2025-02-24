import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from numpy.random import multivariate_normal as MVN
from pathlib import Path
from scipy.stats import uniform

from smcpy import MVNRandomEffects, AdaptiveSampler, VectorMCMC, VectorMCMCKernel
from smcpy.priors import InvWishart, ImproperCov
from smcpy.paths import GeometricPath
from smcpy.proposals import MultivarIndependent

FILEPATH = Path(__file__).parent / "smc_samples.csv"
NUM_RAND_EFF = 100
NUM_DATA_PTS = 5
TRUE_PARAMS = np.array([[2, 3.5]])
RAND_EFF_NOISE_STD = 2.0
X = np.linspace(2, 8, NUM_DATA_PTS)
TRUE_COV = np.array([[2, -1], [-1, 3]])


def eval_model(theta):
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * X + b


def gen_data_from_mvn(eval_model, plot=True):
    r_effs = MVN(TRUE_PARAMS[0], TRUE_COV, NUM_RAND_EFF)
    print(TRUE_PARAMS)
    print(r_effs)
    y_true = np.zeros((len(r_effs), len(X)))
    noisy_data = np.zeros(y_true.shape)
    for i, random_effect in enumerate(r_effs):
        y_true[i] = eval_model(random_effect.reshape(1, -1))
        noisy_data[i] = y_true[i] + np.random.normal(0, RAND_EFF_NOISE_STD, len(X))
    if plot:
        plot_noisy_data(X, y_true, noisy_data)
    return noisy_data, r_effs


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    for i, nd in enumerate(noisy_data):
        ax.plot(x, nd.flatten(), "o", label=f"Noisy Data {i}")
        c = ax.get_lines()[-1].get_color()
        ax.plot(x, y_true[i].flatten(), "-", c=c, linewidth=2, label="True {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend(bbox_to_anchor=(1.1, 1.0))
    plt.show()


if __name__ == "__main__":
    np.random.seed(200)

    noisy_data, true_rand_eff = gen_data_from_mvn(eval_model, plot=False)
    num_samples = 100

    idx1, idx2 = np.triu_indices(TRUE_COV.shape[0])

    true_rand_eff_std = np.tile(RAND_EFF_NOISE_STD, true_rand_eff.shape[0])
    init_inputs = np.concatenate(
        (
            TRUE_PARAMS.flatten(),
            true_rand_eff.flatten(),
            TRUE_COV[idx1, idx2].flatten(),
            true_rand_eff_std.flatten(),
        )
    )
    init_inputs = init_inputs.reshape(1, -1)

    priors = [uniform(-10.0, 20), uniform(-10.0, 20)]
    priors += [uniform(-10.0, 20) for _ in range(NUM_RAND_EFF * 2)]
    priors += [ImproperCov(TRUE_COV.shape[0])]
    priors += [uniform(0, 100) for _ in range(NUM_RAND_EFF)]

    proposal = MultivarIndependent(
        *priors[: (2 + NUM_RAND_EFF * 2)],
        InvWishart(dof=7, scale=np.eye(TRUE_COV.shape[0]) * 5),
        *priors[-NUM_RAND_EFF:],
    )

    path = GeometricPath(proposal=proposal)

    idx1, idx2 = np.triu_indices(TRUE_PARAMS.shape[1])
    log_like_args = (
        [None] * len(idx1),  # total eff variances/covariances
        [None] * NUM_RAND_EFF,
    )  # rand eff std deviations
    log_like_func = MVNRandomEffects

    param_order = (
        ["a", "b"]
        + [f"re{i}" for i in range(NUM_RAND_EFF * TRUE_PARAMS.shape[1])]
        + [f"oe_cov{i}" for i in range(len(idx1))]
        + [f"re_std{i}" for i in range(NUM_RAND_EFF)]
    )

    vector_mcmc = VectorMCMC(
        eval_model, noisy_data, priors, log_like_args, log_like_func
    )
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order, path)

    smc = AdaptiveSampler(mcmc_kernel)
    step_list, mll_list = smc.sample(num_particles=5000, num_mcmc_samples=5)

    df = pd.DataFrame(dict(zip(param_order, step_list[-1].params.T)))
    df.to_csv(FILEPATH)
    import pdb

    pdb.set_trace()
