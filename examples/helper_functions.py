import matplotlib.pyplot as plt
import numpy as np
import time


def eval_model(theta):
    time.sleep(0.05)  # artificial slowdown to show off progress bar
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(x_true, eval_model, std_dev, plot=True):
    y_true = eval_model(x_true)
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot:
        plot_noisy_data(x, y_true, noisy_data)
    return noisy_data


def plot_noisy_data(x, y_true, noisy_data):
    fig, ax = plt.subplots(1)
    ax.plot(x.flatten(), y_true.flatten(), "-k")
    ax.plot(x.flatten(), noisy_data.flatten(), "o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
