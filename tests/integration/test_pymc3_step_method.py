import numpy as np
import pymc3 as pm
import pytest

from scipy.stats import norm
from pymc3.theanof import floatX

from smcpy.mcmc.pymc3_step_methods import SMCMetropolis


class StubModel:

    def __init__(self):
        pass

    def evaluate(self, a, b):
        return np.ones(3) * a


def likelihood(q, data, sigma, model):
    ssqe = np.linalg.norm(data - model.evaluate(q[0], q[1])) ** 2
    likelihood = 1. / (2 * np.pi * sigma ** 2)  ** (len(data) / 2) * \
                 np.exp(-ssqe / (2 * sigma **2))
    return likelihood


def prior(q):
    return norm(0, 1).pdf(q[0]) * norm(0, 1).pdf(q[1])


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
        a = pm.Normal('a', 0, 1)
        b = pm.Normal('b', 0, 1)
        obs = pm.Normal('obs', mu=model.evaluate(a, b), sigma=sigma,
                        observed=data)
    return stoch_model


@pytest.mark.parametrize('q', ([0, 0], [1, 0]))
def test_posterior_calc(pymc_model, model, q, data, sigma):
    posterior = likelihood(q, data, sigma, model) * prior(q)
    assert pymc_model.logp({'a': q[0], 'b': q[1]}) == np.log(posterior)


def test_step_method_no_phi(pymc_model):
    with pytest.raises(AttributeError):
        with pymc_model:
            SMCMetropolis().calc_acceptance_ratio(floatX(np.array([0, 0])),
                                                  floatX(np.array([1, 1])))


@pytest.mark.parametrize('phi', [0.1, 0.5, 1.0])
@pytest.mark.parametrize('step_method', [SMCMetropolis])
def test_acceptance_ratio_in_smc_step_methods(step_method, phi, data, sigma,
                                              model, pymc_model):
    q0 = floatX(np.array([0, 0]))
    q1 = floatX(np.array([1, 0]))

    acceptance = (prior(q1) * likelihood(q1, data, sigma, model) ** phi) / \
                 (prior(q0) * likelihood(q0, data, sigma, model) ** phi)
    log_acceptance = np.log(acceptance)

    with pymc_model:
        step_method = SMCMetropolis()
        for method in step_method.methods:
            method.phi = phi
            pymc_accept = method.calc_acceptance_ratio(q1, q0)

    np.testing.assert_array_almost_equal(pymc_accept, np.array(log_acceptance))


@pytest.mark.parametrize('phi', [-0.1, 1.1])
@pytest.mark.parametrize('step_method', [SMCMetropolis])
def test_bad_phi(step_method, phi, pymc_model):
    with pytest.raises(ValueError):
        with pymc_model:
            SMCMetropolis().methods[0].phi = phi


@pytest.mark.parametrize('step_method', [SMCMetropolis])
def test_proposal_change_with_std_dev(step_method, pymc_model):
    with pymc_model:
        step_method = SMCMetropolis()
        original_proposal = step_method.methods[0].proposal_dist
        assert original_proposal == step_method.methods[0].proposal_dist
        step_method.methods[0].S = np.array([[1, 0.8], [0.8, 1.1]])
        assert original_proposal is not step_method.methods[0].proposal_dist
