import numpy as np
import pytest

from smcpy.mcmc import PyMC3Translator
from smcpy.mcmc import SMCStepMethod


class DummyStepMethod(SMCStepMethod):

    def __init__(self, phi):
        self.phi = phi


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
    mocker.patch.object(stub_pymc3_model, 'sample')

    stub_a = mocker.Mock()
    mocker.patch.object(stub_a, 'random', return_value=1)
    stub_b = mocker.Mock()
    mocker.patch.object(stub_b, 'random', return_value=2)
    stub_pymc3_model.named_vars = {'a': stub_a, 'b': stub_b, 'a__': object}

    return stub_pymc3_model


@pytest.fixture
def stub_trace(mocker):
    stub_trace = mocker.Mock()
    mocker.patch.object(stub_trace, 'get_values',
                        side_effect=[np.array([1, 1]), np.array([2, 2])])
    stub_trace.varnames = ['a', 'a_interval__', 'b']
    return stub_trace


@pytest.fixture
def translator(stub_pymc3_model):
    return PyMC3Translator(stub_pymc3_model, DummyStepMethod)


def test_get_data(translator, data):
    np.testing.assert_array_equal(translator.get_data(), data)


def test_get_log_likelihood(translator):
    loglike = translator.get_log_likelihood({})
    assert loglike == 99


def test_sample_phi_is_set(translator):
    phi = 0.1
    translator.sample(samples=100, phi=phi, init_params={})
    assert translator._last_step_method.phi == phi


def test_sample_inputs_are_passed(translator):
    samples = 100
    phi = 0.1
    init_params = {}

    translator.sample(samples=samples, phi=phi, init_params=init_params)
    translator.pymc3_model.sample.assert_called_with(draws=samples,
                                    step=translator._last_step_method,
                                    chains=1, cores=1, start=init_params,
                                    tune=0, discard_tuned_samples=False)


def test_sample_checks_step_isinstance_smcstep(translator, stub_pymc3_model):
    step = object
    with pytest.raises(TypeError):
        PyMC3Translator(stub_pymc3_model, step)


def test_model_is_copy(translator, stub_pymc3_model):
    assert translator.pymc3_model is not stub_pymc3_model


def test_get_final_values_of_last_trace(translator, stub_trace):
    translator._last_trace = stub_trace

    values = translator.get_final_trace_values()
    assert values == {'a': 1, 'b': 2}


def test_sample_from_prior(translator):
    random_sample = translator.sample_from_prior()
    assert random_sample == {'a': 1, 'b': 2}
