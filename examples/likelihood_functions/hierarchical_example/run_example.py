from scipy.stats import uniform
from pathlib import Path

from smcpy import VectorMCMC, VectorMCMCKernel, AdaptiveSampler
from smcpy.priors import ImproperCov, InvWishart
from smcpy.paths import GeometricPath
from smcpy.proposals import MultivarIndependent
from smcpy.hierarch_log_likelihoods import *
from smcpy.utils.storage import HDF5Storage

from generate_data import *

HDF5_FILE = Path(__file__).parent / "smc.h5"
N_PARTICLES = 1000
N_PARTICLES_TOP = 500
N_MCMC_SAMPLES = 10
COV_PROPOSAL = InvWishart(8, 1.5 * np.eye(2))

if __name__ == "__main__":
    # random effects
    noisy_data, _ = gen_data_from_mvn()
    param_order = ["a", "b", "std"]
    priors = [uniform(0, 20), uniform(0, 20), uniform(0, 10)]
    posterior_samples = np.zeros((NUM_RAND_EFF, N_PARTICLES, 2))
    prior_logpdf = np.zeros((NUM_RAND_EFF, N_PARTICLES, 1))
    mlls = []

    for i, data in enumerate(noisy_data):
        print(f"calibrating REs: {i/NUM_RAND_EFF * 100}% complete")
        mcmc = VectorMCMC(eval_model, data, priors)
        kernel = VectorMCMCKernel(mcmc, param_order)
        smc = AdaptiveSampler(kernel)
        steps, mll = smc.sample(N_PARTICLES, N_MCMC_SAMPLES)

        posterior_samples[i, :, :] = steps[-1].params[:, :-1]  # drop std
        prior_logpdf[i, :, :] = kernel.get_log_priors(steps[-1].param_dict)
        mlls.append(mll[-1])

    # overall effects
    data = posterior_samples
    ll_args = (mlls, prior_logpdf.squeeze())
    param_order = ["a", "b", "cov11", "cov12", "cov22"]
    priors = [uniform(0, 20), uniform(0, 20), ImproperCov(2)]
    proposal = MultivarIndependent(priors[0], priors[1], COV_PROPOSAL)
    path = GeometricPath(proposal=proposal)
    mcmc = VectorMCMC(MVNHierarchModel, data, priors, ll_args, ApproxHierarch)
    kernel = VectorMCMCKernel(mcmc, param_order, path=path)

    with HDF5Storage(HDF5_FILE, "w"):
        smc = AdaptiveSampler(kernel)
        steps, mll = smc.sample(N_PARTICLES_TOP, N_MCMC_SAMPLES)
