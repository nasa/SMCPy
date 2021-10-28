import matplotlib.pyplot as plt
import numpy as np
import time

from mpi4py import MPI
from scipy.stats import uniform

from smcpy import AdaptiveSampler, VectorMCMCKernel, ParallelMCMC


def gen_noisy_data(eval_model, std_dev, plot=True):
    y_true = eval_model(np.array([[2, 3.5]]))
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)
    if plot and MPI.COMM_WORLD.Clone().Get_rank() == 0:
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

    x = np.linspace(1, 5, 100)
    def eval_model(theta):
        out = []
        for theta_i in theta:
            a = theta_i[0]
            b = theta_i[1]
            out.append(a * x + b)
        return np.array(out)

    std_dev = 0.5
    noisy_data = gen_noisy_data(eval_model, std_dev, plot=False)

    # configure
    num_particles = 10000
    num_mcmc_samples = 10
    tgt_ess = 0.8
    priors = [uniform(0., 6.), uniform(0., 6.)]

    comm = MPI.COMM_WORLD.Clone()
    parallel_mcmc = ParallelMCMC(eval_model, noisy_data, priors, comm, std_dev)

    phi_sequence = np.linspace(0, 1, num_smc_steps)

    mcmc_kernel = VectorMCMCKernel(parallel_mcmc, param_order=('a', 'b'))
    smc = AdaptiveSampler(mcmc_kernel)
    t0 = time.time()
    step_list, mll_list = smc.sample(num_particles, num_mcmc_samples, tgt_ess)

    print(f'total time = {time.time() - t0}')
    print(f'mean vector = {step_list[-1].compute_mean()}')
