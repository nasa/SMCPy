import numpy as np
import pandas as pd
import seaborn as sns

from smcpy.priors import ConstrainedUniform
from smcpy import VectorMCMC, VectorMCMCKernel, AdaptiveSampler

if __name__ == "__main__":
    """
    Define a circular uniform prior with radius = 1. Defining the model as
    independent of x and y results in SMC returning samples from the prior.
    """

    data = np.random.default_rng().normal(1, 1, 10)
    model = lambda x: np.ones((x.shape[0], 10))
    bounds = np.array([[-1, -1], [1, 1]])
    constraint_func = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2 <= 1
    p = ConstrainedUniform(bounds=bounds, constraint_function=constraint_func)

    mcmc = VectorMCMC(model, data, priors=[p], log_like_args=1)
    kernel = VectorMCMCKernel(mcmc, param_order=["x", "y"])
    smc = AdaptiveSampler(kernel)
    steps, _ = smc.sample(10000, 2)

    sns.pairplot(pd.DataFrame(steps[-1].param_dict))
    sns.mpl.pyplot.show()
