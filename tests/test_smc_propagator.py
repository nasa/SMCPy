from numpy.testing import assert_array_almost_equal
import pytest
from smcpy.smc.smc_propagator import SMCPropagator

@pytest.fixture
def expected_mean():
    return {'output_0': 1., 'output_1': 2., 'output_2': 3., 'output_3': 4.}


class StubModel():
    def __init__(self, expected_mean):
        self.expected_mean = expected_mean

    def evaluate(self, params):
        return self.expected_mean.values()


@pytest.fixture
def stub_model(expected_mean):
    stub_model = StubModel(expected_mean)
    return stub_model


def test_smc_propagator_returns_multi_dim_output(stub_model, filled_step,
                                                 expected_mean):
    smc_propagator = SMCPropagator(stub_model)
    prop_smc_step = smc_propagator.propagate(filled_step)
    mean = prop_smc_step.get_mean()
    assert all(key in mean.keys() for key in expected_mean.keys())
    assert_array_almost_equal([mean[key] for key in sorted(mean.keys())],
                              expected_mean.values())
