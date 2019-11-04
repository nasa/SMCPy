import numpy as np
import pytest

from smcpy.model.base_model import BaseModel

class ImplementedModel(BaseModel):

    def __init__(self):
        pass

    def evaluate(self, *args, **kwargs):
        params = self.process_args(args, kwargs)
        return np.array(params.values())


@pytest.fixture
def implemented_model():
    return ImplementedModel()


def test_evaluate_accepts_kwargs(implemented_model):
    assert implemented_model.evaluate(test=1) == [1]


def test_evaluate_accepts_args(implemented_model):
    assert implemented_model.evaluate({'test': 1}) == [1]


def test_evaluate_non_dict_args(implemented_model):
    with pytest.raises(TypeError):
        implemented_model.evaluate(1)


def test_generate_noisy_data_with_zero_stdv(implemented_model):
    params = {'test': 1}
    stdv = 0
    noisy_data = implemented_model.generate_noisy_data_with_model(stdv, params)
    assert noisy_data[0] == 1


def test_generate_noisy_data_with_one_stdv(implemented_model):
    np.random.seed(1)
    params = {'test': 1}
    stdv = 1
    noise = np.random.normal(0, stdv, (1,))
    noisy_data = implemented_model.generate_noisy_data_with_model(stdv, params)
    assert noisy_data[0] != 1 + noise
