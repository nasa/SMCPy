import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import uniform

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler

def eval_model(theta):
    import time
    time.sleep(0.1)
    a = theta[:, 0, None]
    b = theta[:, 1, None]
    return a * np.arange(100) + b


def generate_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
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

    std_dev = 2
    noisy_data = generate_data(eval_model, std_dev, plot=False)

    priors = [uniform(0., 6.), uniform(0., 6.)]
    vector_mcmc = VectorMCMC(eval_model, noisy_data, priors, std_dev)
    mcmc_kernel = VectorMCMCKernel(vector_mcmc, param_order=('a', 'b'))

    smc = SMCSampler(mcmc_kernel)
    phi_sequence = np.linspace(0, 1, 20)
    step_list, mll = smc.sample(num_particles=500, num_mcmc_samples=5,
                                phi_sequence=phi_sequence, ess_threshold=1.0,
                                progress_bar=True)

    print('marginal log likelihood = {}'.format(mll))
    print('parameter means = {}'.format(step_list[-1].compute_mean()))
