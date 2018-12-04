import pytest
import numpy as np
from smcpy.model.base_model import BaseModel
from smcpy.mcmc.mcmc_sampler import MCMCSampler

'''
Unit and regression tests for the smc_sampler.

NOTE:
At this time, this should only be run in serial (not sure what seed = 20 will
do if that is set on all processors). To extend to parallel, would want to
have seed be a function of rank.
'''

class Model(BaseModel):

    def __init__(self, x):
        self.x = np.array(x)

    def evaluate(self, *args, **kwargs):
        params = self.process_args(args, kwargs)
        a = params['a']
        b = params['b']
        return a * self.x + b


# set seed
np.random.seed(20)

std_dev = 0.6
y_noisy = np.genfromtxt('noisy_data.txt')

param_priors = {'a': ['Uniform', -5.0, 5.0],
                'b': ['Uniform', -5.0, 5.0]}


@pytest.fixture
def x_space():
    return np.arange(50)


@pytest.fixture
def model(x_space):
    return Model(x_space)


def test_sample():
    pass


def test_save_particle_chain():
    pass


def test_load_particle_chain():
    pass


def test_setup_mcmc_sampler(model):
    assert isinstance(model._mcmc, MCMCSampler)


def test_sample_value_error_restart_time_out_of_range():
    pass


def test_set_proposal_distribution_value_error_bad_proposal_def():
    pass

