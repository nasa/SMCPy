import numpy as np

class model:
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
        a, b = self.check_inputs(*args, **kwargs)
        return a * self.x + b

    def generate_noisy_data(self, paramMap, std_dev):
        """
        Obtains model output using evaluate() method, then adds gaussian
        white noise using std_dev.
        """
        a = paramMap['a']
        b = paramMap['b']
        y = a * self.x + b
        yn = y + np.random.normal(0.0, std_dev, len(y))
        return yn

    def check_inputs(self, *args, **kwargs):
        """
        Used to catch errors with input format.
        """
        if len(args) == 1 and type(args[0]) == dict:
            a = args[0]['a']
            b = args[0]['b']
        elif len(args) == 2:
            a = args[0]
            b = args[1]
        elif len(args) == 0:
            a = kwargs['a']
            b = kwargs['b']
        elif len(args) > 2:
            error_msg = 'If inputs are individual args, len(inputs) == 2. ' +\
                        'If input is paramMap, must be a single dictionary.'
            raise IOError(error_msg)
        else:
            error_msg = 'Check inputs: 2 are required as either two args - ' +\
                        'evaluate(a, b) - a set of two kwargs - ' +\
                        'evaluate(a=a, b=b) - or a paramMap dictionary -' +\
                        'evaluate({"a":a, "b":b}).'
            raise IOError(error_msg)
        return (a, b)
