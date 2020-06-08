import numpy as np
import pymc3 as pm
import pytest
import time

from copy import copy

from smcpy.mcmc.pymc3_step_methods import SMCMetropolis
from smcpy.mcmc.pymc3_translator import PyMC3Translator


def run_timed_vanilla_mcmc(num_samples, init_params, pymc3_model):
    time0 = time.time()
    with pymc3_model:
        trace = pm.sample(num_samples, chains=1, cores=1, progressbar=False,
                          discard_tuned_samples=False, step=pm.Metropolis(),
                          tune=False, start=init_params)
        a_trace = trace.get_values('a')
        b_trace = trace.get_values('b')
    time1 = time.time()
    return a_trace, b_trace, time1 - time0


def run_timed_with_mcmc_smc_step_method(mcmc_kernel, num_samples, init_params):
    time0 = time.time()
    mcmc_kernel = copy(mcmc_kernel)
    mcmc_kernel.sample(num_samples, init_params, cov=np.eye(2), phi=1)
    trace = mcmc_kernel.get_all_trace_values()
    time1 = time.time()
    return trace['a'], trace['b'], time1 - time0


def eval_model(a, b):
    x = np.arange(100)
    return a * x + b


@pytest.fixture
def std_dev(std):
    return std


@pytest.fixture
def noisy_data(std_dev):
    y_true = eval_model(a=2, b=3.5)
    return y_true + np.random.normal(0, std_dev, y_true.shape)


@pytest.fixture
def pymc3_model(std_dev, noisy_data):
    pymc3_model = pm.Model()
    with pymc3_model:
        a = pm.Uniform('a', 0., 5., transform=None)
        b = pm.Uniform('b', 0., 5., transform=None)
        mu = eval_model(a, b)
        obs = pm.Normal('obs', mu=mu, sigma=std_dev, observed=noisy_data)
    return pymc3_model


@pytest.fixture
def mcmc_kernel(pymc3_model):
    return PyMC3Translator(copy(pymc3_model), SMCMetropolis)


@pytest.mark.filterwarnings('ignore: The number of samples is too small')
@pytest.mark.parametrize('init_params', [{'a': 1, 'b': 1}, {'a': 2, 'b': 1.5}])
@pytest.mark.parametrize('std', [0.1, 1.0])
def test_translator_vs_vanilla_pymc3(std_dev, mcmc_kernel, pymc3_model,
                                     init_params):
    n_samples = 5

    np.random.seed(100)
    a_pymc, b_pymc, total_time_pymc = \
        run_timed_vanilla_mcmc(n_samples, init_params, pymc3_model)

    np.random.seed(100)
    a_smcpy, b_smcpy, total_time_smcpy = \
        run_timed_with_mcmc_smc_step_method(mcmc_kernel, n_samples, init_params)

    np.testing.assert_array_almost_equal(a_smcpy, a_pymc)
    np.testing.assert_array_almost_equal(b_smcpy, b_pymc)
    assert total_time_smcpy <= total_time_pymc
