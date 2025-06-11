import numpy as np
import matplotlib.pyplot as plt


NUM_RAND_EFF = 100
NUM_DATA_PTS = 5
TRUE_PARAMS = np.array([[2.3, 10.8]])
RAND_EFF_NOISE_STD = 0.1
X = np.linspace(2, 8, NUM_DATA_PTS)
TRUE_COV = np.array([[0.2, -0.1], [-0.1, 0.3]])


def eval_model(theta):
    a = theta[:, [0]]
    b = theta[:, [1]]
    return a * X + b


def gen_data_from_mvn(plot=True, show=False):
    rng = np.random.default_rng(seed=34)

    r_effs = rng.multivariate_normal(TRUE_PARAMS[0], TRUE_COV, NUM_RAND_EFF)

    y_true = eval_model(r_effs)
    noise = rng.normal(0, RAND_EFF_NOISE_STD, y_true.shape)
    noisy_data = y_true + noise

    if plot:
        plot_noisy_data(X, y_true, noisy_data, show)
    return noisy_data, r_effs


def plot_noisy_data(x, y_true, noisy_data, show=True):
    _, ax = plt.subplots(1)
    for i, nd in enumerate(noisy_data):
        ax.plot(x, nd.flatten(), "x")
        c = ax.get_lines()[-1].get_color()
        ax.plot(x, y_true[i].flatten(), "-", c=c, linewidth=2, label="True {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend(["Noisy Data", "Truth"])
    if show:
        plt.show()


if __name__ == "__main__":
    noisy_data, r_effs = gen_data_from_mvn(plot=True, show=True)
