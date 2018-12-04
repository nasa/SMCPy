import numpy as np
from smcpy.model.base_model import BaseModel

class Model(BaseModel):
    """
    Defines a simple linear model for MCMC and fit tests.
    """

    def __init__(self, x):
        """
        Class is instanced with a support, x, defining points where the 
        model will be evaluated (i.e., y(x))
        """
        self.x = np.array(x)


    def evaluate(self, *args, **kwargs):
        """
        Example linear model y = a*x+b where input can either be a single
        paramMap => {'a':a, 'b':b} or individual kwargs => a=a, b=b or
        individual args => a, b.
        """
        params = self.process_args(args, kwargs)
        a = params['a']
        b = params['b']
        return a * self.x + b
