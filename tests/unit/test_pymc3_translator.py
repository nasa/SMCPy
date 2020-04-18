import numpy as np
import pytest

from smcpy.mcmc import PyMC3Translator
from smcpy.mcmc import SMCStepMethod


class RVDummySampler:

    def __init__(self, value):
        self.value = value

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
    mocker.patch.object(stub_pymc3_model, 'sample')

    stub_a = RVDummySampler(1)
    stub_b = RVDummySampler(2)
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
    return PyMC3Translator(stub_pymc3_model)


def test_get_data(translator, data):
    np.testing.assert_array_equal(translator.get_data(), data)


def test_get_log_likelihood(translator):
    loglike = translator.get_log_likelihood({})
    assert loglike == 99


def test_sample_inputs_are_passed(translator, mocker):
    samples = 100
    phi = 01.
    init_params = {}
    step = mocker.Mock(SMCStepMethod)

    translator.sample(samples=samples, phi=phi, init_params=init_params,
                      step_method=step)
    translator.pymc3_model.sample.assert_called_with(draws=samples, step=step,
                                    chains=1, cores=1, start=init_params,
                                    tune=0, discard_tuned_samples=False)
    assert step.phi == phi


def test_sample_checks_step_isinstance_smcstep(translator):
    step = object
    with pytest.raises(TypeError):
        translator.sample(step_method=step)


def test_model_is_copy(translator, stub_pymc3_model):
    assert translator.pymc3_model is not stub_pymc3_model


def test_get_final_values_of_last_trace(translator, stub_trace):
    translator._last_trace = stub_trace

    values = translator.get_final_trace_values()
    assert values == {'a': 1, 'b': 2}


def test_sample_from_prior(translator):
    random_sample = translator.sample_from_prior(size=1)
    assert random_sample == {'a': 1, 'b': 2}


def test_multiple_samples_from_prior(translator):
    random_sample = translator.sample_from_prior(size=10)
    assert random_sample == {'a': [1] * 10, 'b': [2] * 10}
