import numpy as np
import pytest

from smcpy.mcmc.pymc3_kernel import PyMC3Kernel
from smcpy.mcmc import SMCStepMethod


class DummyStepMethodObject():

    def __init__(self):
        self.phi = 0
        self.S = None


class DummyStepMethodClass(SMCStepMethod):

    def __init__(self):
        self.methods = [DummyStepMethodObject(), DummyStepMethodObject()]

    @property
    def __bases__(self):
        return [SMCStepMethod]


class RVDummySampler:

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def random(self, size):
        if size == 1:
            return self.value
        return [self.value] * size


@pytest.fixture
def data():
    return np.array([1., 1.])


@pytest.fixture
def stub_pymc3_model(data, mocker):
    stub_pymc3_model = mocker.Mock()
    mocker.patch.object(stub_pymc3_model, 'observed_RVs', return_value=[object])
    stub_pymc3_model.observed_RVs[0].observations = data
    mocker.patch.object(stub_pymc3_model, 'logp', return_value=66)
    mocker.patch.object(stub_pymc3_model.observed_RVs[0], 'logp',
                        return_value=99)
    stub_pymc3_model.__enter__ = mocker.Mock()
    stub_pymc3_model.__exit__ = mocker.Mock()
    
    stub_a = RVDummySampler(1, 'a')
    stub_b = RVDummySampler(2, 'b')
    stub_a__ = RVDummySampler(1, 'a__')
    stub_pymc3_model.vars =[stub_a, stub_b, stub_a__]

    return stub_pymc3_model


@pytest.fixture
def stub_trace(mocker):
    stub_trace = mocker.Mock()
    mocker.patch.object(stub_trace, 'get_values',
                        side_effect=[np.array([1, 1]), np.array([2, 2])])
    stub_trace.varnames = ['a', 'a_interval__', 'b']
    return stub_trace


@pytest.fixture
def kernel(stub_pymc3_model):
    return PyMC3Kernel(stub_pymc3_model, DummyStepMethodClass)


def test_get_data(kernel, data):
    np.testing.assert_array_equal(kernel.get_data(), data)


def test_get_log_likelihood(kernel):
    loglike = kernel.get_log_likelihood({})
    assert loglike == 99


def test_sample_phi_and_cov_are_set(kernel):
    phi = 0.1
    cov = np.array([[4, 1.2], [1.2, 1]])
    kernel.sample(num_samples=100, cov=cov, phi=phi, init_params={})
    assert kernel.step_method.methods[0].phi == phi
    assert kernel.step_method.methods[0].S == 2
    assert kernel.step_method.methods[1].phi == phi
    assert kernel.step_method.methods[1].S == 1


def test_sample_inputs_are_passed(kernel, stub_pymc3_model, mocker):
    num_samples = 100
    phi = 0.1
    cov = np.eye(2)
    init_params = {}

    mocker.patch('pymc3.sampling.sample')
    import pymc3

    kernel.sample(num_samples=num_samples, phi=phi, cov=cov,
                      init_params=init_params)
    pymc3.sampling.sample.assert_called_with(draws=num_samples,
                              step=kernel.step_method,
                              chains=1, cores=1, start=init_params,
                              tune=0, discard_tuned_samples=False,
                              progressbar=False)


def test_sample_checks_step_isinstance_smcstep(kernel, stub_pymc3_model):
    step = object
    with pytest.raises(TypeError):
        PyMC3Kernel(stub_pymc3_model, step)


def test_model_is_copy(kernel, stub_pymc3_model):
    assert kernel.pymc3_model is not stub_pymc3_model


def test_get_final_values_of_last_trace(kernel, stub_trace):
    kernel._last_trace = stub_trace

    values = kernel.get_final_trace_values()
    assert values == {'a': 1, 'b': 2}


def test_get_last_trace(kernel, stub_trace):
    kernel._last_trace = stub_trace

    values = kernel.get_all_trace_values()
    assert values == {'a': [1, 1], 'b': [2, 2]}


def test_sample_from_prior(kernel):
    random_sample = kernel.sample_from_prior(size=1)
    assert random_sample == {'a': 1, 'b': 2}


def test_multiple_samples_from_prior(kernel):
    random_sample = kernel.sample_from_prior(size=10)
    assert random_sample == {'a': [1] * 10, 'b': [2] * 10}
