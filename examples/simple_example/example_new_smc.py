import numpy as np
import pymc3 as pm

from scipy.optimize import minimize

from smcpy.mcmc.pymc3_step_methods import SMCMetropolis
from smcpy.mcmc.pymc3_translator import PyMC3Translator
from smcpy.smc.initializer import Initializer
from smcpy.smc.mutator import Mutator

from model import Model

def perform_sampling(num_particles, num_steps, num_mcmc_samples, phi_sequence,
                     pymc3_model):

    mcmc_kernel = PyMC3Translator(pymc3_model, SMCMetropolis)
    initializer = Initializer(mcmc_kernel, phi_sequence[1])
    mutator = Mutator(mcmc_kernel)

    smc_step = initializer.initialize_particles_from_prior(num_particles)
    step_list = [smc_step]
    for phi in phi_sequence[2:]:
        smc_step.update_weights(phi)
        smc_step.resample_if_needed(ess_threshold=num_particles * 0.75)
        smc_step = mutator.mutate(smc_step, num_mcmc_samples, phi)
        step_list.append(smc_step)

    print(smc_step.get_mean())


if __name__ == '__main__':

    np.random.seed(100)

    # instance model / set up ground truth / add noise
    x = np.arange(100)
    #my_model = Model(x)
    eval_model = lambda a, b: a * x + b
    std_dev = 1.
    #y_true = my_model.evaluate(a=2, b=3.5) 
    y_true = eval_model(2, 3.5)
    noisy_data = y_true + np.random.normal(0, std_dev, y_true.shape)

    # setup pymc3 model
    pymc3_model = pm.Model()
    with pymc3_model:
        a = pm.Uniform('a', 0., 5., transform=None)
        b = pm.Uniform('b', 0., 5., transform=None)
        std_dev = pm.Uniform('std_dev', 0, 5, transform=None)

        #mu = my_model.evaluate(a=a, b=b)
        mu = eval_model(a, b)

        obs = pm.Normal('obs', mu=mu, sigma=std_dev, observed=noisy_data)

    # run smc
    num_particles = 100 
    num_steps = 5 
    num_mcmc_samples = 1
    phi_sequence = np.linspace(0, 1, num_steps)

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    perform_sampling(num_particles, num_steps, num_mcmc_samples, phi_sequence,
                     pymc3_model)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
