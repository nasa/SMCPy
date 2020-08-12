import numpy as np

from .mcmc_base import MCMCBase


class VectorMCMC(MCMCBase):
    
    def __init__(self, model, data, priors, std_dev=None, debug=False):
        '''
        :param model: maps inputs to outputs
        :type model: callable
        :param data: data corresponding to model outputs
        :type data: 1D array
        :param priors: random variable objects with a pdf and rvs method (e.g.
            scipy stats random variable objects)
        :type priors: list of objects
        :param std_dev: Gaussian additive noise standard deviation
        :type std_dev: float
        '''
        super().__init__(model, data, priors, std_dev, debug)

    def evaluate_model(self, inputs):
        return self._eval_model(inputs)
