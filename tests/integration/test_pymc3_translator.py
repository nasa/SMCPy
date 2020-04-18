import numpy as np
import pymc3 as pm
import pytest

from smcpy.mcmc.pymc3_step_methods import SMCMetropolis
from smcpy.mcmc.pymc3_translator import PyMC3Translator


np.random.seed(2)

class StubModel:

    def __init__(self):
        pass

    def evaluate(self, a, b):
        return np.ones(3) * a / b


def likelihood(a, b, data, sigma, model):
    ssqe = np.linalg.norm(data - model.evaluate(a, b)) ** 2
    likelihood = 1. / (2 * np.pi * sigma ** 2)  ** (len(data) / 2) * \
                 np.exp(-ssqe / (2 * sigma **2))
    return likelihood


def prior(a, b):
    return uniform(0, 1).pdf(a) * uniform(0, 10).pdf(b)


@pytest.fixture
def data():
    return np.ones(3) * 2.


@pytest.fixture
def model():
    return StubModel()


@pytest.fixture
def sigma():
    return 1


@pytest.fixture
def pymc_model(model, sigma, data):
    stoch_model = pm.Model()
    with stoch_model:
        a = pm.Uniform('a', 0, 1, transform=None)
        b = pm.Uniform('b', 0, 1, transform=None)
        obs = pm.Normal('obs', mu=model.evaluate(a, b), sigma=sigma,
                        observed=data)
    return stoch_model


@pytest.fixture
def translator(pymc_model):
    return PyMC3Translator(pymc_model, SMCMetropolis)


def test_get_data(data, translator):
    np.testing.assert_array_equal(translator.get_data(), data)


@pytest.mark.parametrize('a,b', [(0.1, 0.4), (0.5, 0.9)])
def test_get_log_likelihood(a, b, data, sigma, model, translator):
    like = likelihood(a, b, data, sigma, model)
    assert translator.get_log_likelihood({'a': a, 'b': b}) == \
           pytest.approx(np.log(like))


@pytest.mark.parametrize('size', [100, 1000, 2000])
def test_sample_from_prior(translator, size):
    samples = translator.sample_from_prior(size)
    for key in ['a', 'b']:
        assert len(samples[key]) == size
        assert all([s <= 1 and s >= 0 for s in samples[key]])


def test_sample(translator):
    init_params={'a': 0, 'b': 0}
    translator.sample(100, init_params=init_params, phi=1)
    new_params = translator.get_final_trace_values()
    assert len(translator._last_trace.get_values('a')) == 100
    assert new_params.keys() == init_params.keys()
    assert new_params.values() != init_params.values()
